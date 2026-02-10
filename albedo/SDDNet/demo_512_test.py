import torch
import torchvision
from networks.sddnet import SDDNet
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import torch.nn.functional as F

def process_demo_with_512_model():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 使用ckpt目录中的512x512模型
    ckpt_path = './ckpt/ep_019.ckpt'
    demo_dir = './demo'
    save_dir = './demo_results_512'
    os.makedirs(save_dir, exist_ok=True)

    # 创建模型 - 使用512x512的配置
    model = SDDNet(backbone='efficientnet-b3',
                   proj_planes=16,
                   pred_planes=32,
                   use_pretrained=False,
                   fix_backbone=False,
                   has_se=False,
                   dropout_2d=0,
                   normalize=True,  # 模型内部normalize
                   mu_init=0.4,
                   reweight_mode='manual')

    # 加载预训练权重
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f"Checkpoint keys: {list(ckpt.keys())}")
        
        # 检查checkpoint内容
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            # 如果直接是state_dict
            model.load_state_dict(ckpt)
            
        print(f"Successfully loaded 512x512 model from {ckpt_path}")
        
        # 打印一些checkpoint信息
        if 'epoch' in ckpt:
            print(f"Epoch: {ckpt['epoch']}")
        if 'best_ber' in ckpt:
            print(f"Best BER: {ckpt['best_ber']}")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Let me try different loading methods...")
        
        # 尝试其他加载方式
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            # 如果有不匹配的keys，尝试strict=False
            model.load_state_dict(ckpt['model'], strict=False)
            print("Loaded with strict=False")
        except:
            return

    model.to(device)
    model.eval()

    # 图像预处理 - 尝试两种方式
    preprocess_configs = [
        {
            'name': 'no_norm',
            'transform': transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
        },
        {
            'name': 'with_norm', 
            'transform': transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
    ]

    # 处理demo目录中的所有图片
    img_files = [f for f in os.listdir(demo_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not img_files:
        print("No image files found in demo directory!")
        return

    print(f"Processing {len(img_files)} image(s) with 512x512 model...")

    with torch.no_grad():
        for img_file in img_files:
            print(f"\nProcessing {img_file}...")
            
            # 读取图像
            img_path = os.path.join(demo_dir, img_file)
            image = Image.open(img_path).convert('RGB')
            original_size = image.size
            print(f"Original image size: {original_size}")
            
            base_name = os.path.splitext(img_file)[0]
            
            for config in preprocess_configs:
                print(f"  Testing with {config['name']} preprocessing...")
                
                # 预处理
                img_tensor = config['transform'](image).unsqueeze(0).to(device)
                
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
                
                # 创建不同的可视化版本
                soft_mask_3ch = shadow_mask_resized[0].repeat(3, 1, 1)
                binary_mask_3ch = binary_mask[0].repeat(3, 1, 1)
                
                # 创建叠加效果
                overlay = original_tensor.clone()
                mask_2d = binary_mask[0][0]
                alpha = 0.6
                overlay[0] = torch.where(mask_2d > 0.5, 
                                       alpha * torch.tensor(1.0) + (1-alpha) * original_tensor[0], 
                                       original_tensor[0])
                overlay[1] = torch.where(mask_2d > 0.5, 
                                       (1-alpha) * original_tensor[1], 
                                       original_tensor[1])
                overlay[2] = torch.where(mask_2d > 0.5, 
                                       (1-alpha) * original_tensor[2], 
                                       original_tensor[2])
                
                # 保存结果
                comparison_path = os.path.join(save_dir, f"{base_name}_512_{config['name']}_comparison.png")
                torchvision.utils.save_image([original_tensor, soft_mask_3ch, binary_mask_3ch, overlay], 
                                           comparison_path, nrow=4, padding=2)
                
                # 单独保存mask
                mask_path = os.path.join(save_dir, f"{base_name}_512_{config['name']}_mask.png") 
                torchvision.utils.save_image(binary_mask[0], mask_path)
                
                # 统计信息
                mask_ratio = torch.sum(binary_mask).item() / binary_mask.numel()
                print(f"    Shadow area ratio: {mask_ratio:.1%}")
                print(f"    Results saved: {comparison_path}")

    print("\n✅ 512x512 model processing completed!")
    print(f"📁 Results saved in: {save_dir}")
    print("📋 Compare the results from different preprocessing methods to see which works best.")

if __name__ == "__main__":
    process_demo_with_512_model()
