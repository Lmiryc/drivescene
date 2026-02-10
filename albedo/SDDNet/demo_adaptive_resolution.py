import torch
import torchvision
from networks.sddnet import SDDNet
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import torch.nn.functional as F

def process_demo_with_adaptive_resolution():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 模型配置
    ckpt_path = './ckpt/ep_019.ckpt'
    demo_dir = './demo'
    save_dir = './demo_results_adaptive'
    os.makedirs(save_dir, exist_ok=True)

    # 创建模型
    model = SDDNet(backbone='efficientnet-b3',
                   proj_planes=16,
                   pred_planes=32,
                   use_pretrained=False,
                   fix_backbone=False,
                   has_se=False,
                   dropout_2d=0,
                   normalize=True,
                   mu_init=0.4,
                   reweight_mode='manual')

    # 加载预训练权重
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        print(f"Successfully loaded model from {ckpt_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.to(device)
    model.eval()

    # 处理demo目录中的所有图片
    img_files = [f for f in os.listdir(demo_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not img_files:
        print("No image files found in demo directory!")
        return

    # 测试不同的分辨率策略
    resolution_strategies = [
        {'name': 'original_512', 'size': 512, 'keep_aspect': False},
        {'name': 'keep_aspect_512', 'size': 512, 'keep_aspect': True},
        {'name': 'original_400', 'size': 400, 'keep_aspect': False},
        {'name': 'keep_aspect_400', 'size': 400, 'keep_aspect': True},
    ]

    def get_transform_with_strategy(strategy, image_size):
        if strategy['keep_aspect']:
            # 保持宽高比的resize
            w, h = image_size
            target_size = strategy['size']
            
            # 计算缩放比例，使得长边为target_size
            scale = target_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # 确保尺寸是32的倍数（对于大多数CNN架构更友好）
            new_w = ((new_w + 31) // 32) * 32
            new_h = ((new_h + 31) // 32) * 32
            
            return transforms.Compose([
                transforms.Resize((new_h, new_w)),
                transforms.ToTensor()
            ]), (new_w, new_h)
        else:
            # 直接resize到正方形
            target_size = strategy['size']
            return transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor()
            ]), (target_size, target_size)

    print(f"Processing {len(img_files)} image(s) with different resolution strategies...")

    with torch.no_grad():
        for img_file in img_files:
            print(f"\nProcessing {img_file}...")
            
            # 读取图像
            img_path = os.path.join(demo_dir, img_file)
            image = Image.open(img_path).convert('RGB')
            original_size = image.size
            print(f"Original image size: {original_size}")
            
            base_name = os.path.splitext(img_file)[0]
            
            # 测试不同的分辨率策略
            for strategy in resolution_strategies:
                print(f"  Testing strategy: {strategy['name']}")
                
                # 获取对应的transform
                img_transform, processed_size = get_transform_with_strategy(strategy, original_size)
                print(f"    Processed size: {processed_size}")
                
                # 预处理
                img_tensor = img_transform(image).unsqueeze(0).to(device)
                
                # 推理
                result = model(img_tensor)
                
                # 获取shadow mask
                shadow_logits = result['logit']
                shadow_mask = torch.sigmoid(shadow_logits).cpu()
                
                # 调整回原始尺寸
                shadow_mask_resized = F.interpolate(shadow_mask, size=original_size[::-1], 
                                                  mode='bilinear', align_corners=False)
                
                # 应用阈值得到二值mask
                binary_mask = (shadow_mask_resized > 0.5).float()
                
                # 准备可视化
                original_tensor = transforms.ToTensor()(image)
                
                # 创建可视化版本
                soft_mask_3ch = shadow_mask_resized[0].repeat(3, 1, 1)
                binary_mask_3ch = binary_mask[0].repeat(3, 1, 1)
                
                # 创建叠加效果
                overlay = original_tensor.clone()
                mask_2d = binary_mask[0][0]
                alpha = 0.6
                
                # 使用更明显的颜色标记阴影区域
                overlay[0] = torch.where(mask_2d > 0.5, 
                                       alpha * torch.tensor(1.0) + (1-alpha) * original_tensor[0], 
                                       original_tensor[0])
                overlay[1] = torch.where(mask_2d > 0.5, 
                                       (1-alpha) * original_tensor[1] * 0.5,  # 减少绿色
                                       original_tensor[1])
                overlay[2] = torch.where(mask_2d > 0.5, 
                                       (1-alpha) * original_tensor[2] * 0.5,  # 减少蓝色
                                       original_tensor[2])
                
                # 保存结果
                comparison_path = os.path.join(save_dir, f"{base_name}_{strategy['name']}_result.png")
                torchvision.utils.save_image([original_tensor, soft_mask_3ch, binary_mask_3ch, overlay], 
                                           comparison_path, nrow=4, padding=2)
                
                # 单独保存mask
                mask_path = os.path.join(save_dir, f"{base_name}_{strategy['name']}_mask.png")
                torchvision.utils.save_image(binary_mask[0], mask_path)
                
                # 统计信息
                mask_ratio = torch.sum(binary_mask).item() / binary_mask.numel()
                print(f"    Shadow area ratio: {mask_ratio:.1%}")
                print(f"    Aspect ratio change: {original_size[0]/original_size[1]:.2f} -> {processed_size[0]/processed_size[1]:.2f}")

    print("\n✅ Adaptive resolution processing completed!")
    print(f"📁 Results saved in: {save_dir}")
    print("\n📋 Resolution strategies tested:")
    print("  - original_512: Direct resize to 512x512 (may distort aspect ratio)")
    print("  - keep_aspect_512: Keep aspect ratio, max dimension 512")
    print("  - original_400: Direct resize to 400x400 (may distort aspect ratio)")  
    print("  - keep_aspect_400: Keep aspect ratio, max dimension 400")
    print("\n💡 Compare results to see which strategy works best for your images!")

if __name__ == "__main__":
    process_demo_with_adaptive_resolution()
