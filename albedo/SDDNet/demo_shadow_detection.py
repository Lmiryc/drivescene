import torch
import torchvision
from networks.sddnet import SDDNet
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import torch.nn.functional as F

def process_demo_images():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 使用正确的配置：SBU模型 + 无归一化
    ckpt_path = './sddnet_ckpt/sbu.ckpt'
    demo_dir = './demo'
    save_dir = './demo_results_final'
    os.makedirs(save_dir, exist_ok=True)

    # 创建模型
    model = SDDNet(backbone='efficientnet-b3',
                   proj_planes=16,
                   pred_planes=32,
                   use_pretrained=False,
                   fix_backbone=False,
                   has_se=False,
                   dropout_2d=0,
                   normalize=True,  # 这是模型内部的normalize
                   mu_init=0.4,
                   reweight_mode='manual')

    # 加载预训练权重
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Successfully loaded SBU checkpoint")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.to(device)
    model.eval()

    # 图像预处理 - 不使用ImageNet归一化
    img_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()  # 只转换为tensor，不做归一化
    ])

    # 处理demo目录中的所有图片
    img_files = [f for f in os.listdir(demo_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not img_files:
        print("No image files found in demo directory!")
        return

    print(f"Processing {len(img_files)} image(s) using SBU model with no normalization...")

    with torch.no_grad():
        for img_file in img_files:
            print(f"Processing {img_file}...")
            
            # 读取图像
            img_path = os.path.join(demo_dir, img_file)
            image = Image.open(img_path).convert('RGB')
            original_size = image.size
            
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
            
            # 创建不同的可视化版本
            soft_mask_3ch = shadow_mask_resized[0].repeat(3, 1, 1)  # 软mask（灰度值）
            binary_mask_3ch = binary_mask[0].repeat(3, 1, 1)       # 二值mask
            
            # 创建彩色叠加效果
            overlay = original_tensor.clone()
            mask_2d = binary_mask[0][0]
            # 在阴影区域叠加半透明红色
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
            base_name = os.path.splitext(img_file)[0]
            
            # 保存对比图：原图 | 软mask | 二值mask | 叠加效果
            comparison_path = os.path.join(save_dir, f"{base_name}_comparison.png")
            torchvision.utils.save_image([original_tensor, soft_mask_3ch, binary_mask_3ch, overlay], 
                                       comparison_path, nrow=4, padding=2)
            
            # 单独保存各个版本
            torchvision.utils.save_image(original_tensor, 
                                       os.path.join(save_dir, f"{base_name}_original.png"))
            torchvision.utils.save_image(soft_mask_3ch, 
                                       os.path.join(save_dir, f"{base_name}_soft_mask.png"))
            torchvision.utils.save_image(binary_mask_3ch, 
                                       os.path.join(save_dir, f"{base_name}_binary_mask.png"))
            torchvision.utils.save_image(overlay, 
                                       os.path.join(save_dir, f"{base_name}_overlay.png"))
            
            # 统计信息
            mask_ratio = torch.sum(binary_mask).item() / binary_mask.numel()
            print(f"  - Shadow area ratio: {mask_ratio:.1%}")
            print(f"  - Results saved to {save_dir}")

    print(f"Results saved in: {save_dir}")

if __name__ == "__main__":
    process_demo_images()
