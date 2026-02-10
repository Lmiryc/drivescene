import torch
import torchvision
from networks.sddnet import SDDNet
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import cv2
import torch.nn.functional as F

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 模型参数 - 尝试不同的预训练模型
model_configs = [
    {'name': 'SBU', 'path': './sddnet_ckpt/sbu.ckpt'},
    {'name': 'ISTD', 'path': './sddnet_ckpt/istd.ckpt'}, 
    {'name': 'UCF', 'path': './sddnet_ckpt/ucf.ckpt'}
]

demo_dir = './demo'
save_dir = './demo_results_fixed'
os.makedirs(save_dir, exist_ok=True)

for config in model_configs:
    print(f"\n=== Testing with {config['name']} model ===")
    
    # 创建模型 - 使用和原始test.py相同的配置
    model = SDDNet(backbone='efficientnet-b3',
                   proj_planes=16,
                   pred_planes=32,
                   use_pretrained=False,  # 先设为False避免下载
                   fix_backbone=False,
                   has_se=False,
                   dropout_2d=0,
                   normalize=True,
                   mu_init=0.4,
                   reweight_mode='manual')

    # 加载预训练权重
    try:
        ckpt = torch.load(config['path'], map_location=device)
        print(f"Checkpoint keys: {list(ckpt.keys())}")
        model.load_state_dict(ckpt['model'])
        print(f"Successfully loaded checkpoint from {config['path']}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        continue

    model.to(device)
    model.eval()

    # 图像预处理 - 参考原始test.py，可能不需要normalize
    img_transform_with_norm = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_transform_no_norm = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # 处理demo目录中的所有图片
    img_files = [f for f in os.listdir(demo_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not img_files:
        print("No image files found in demo directory!")
        continue

    with torch.no_grad():
        for img_file in img_files:
            print(f"Processing {img_file} with {config['name']} model...")
            
            # 读取图像
            img_path = os.path.join(demo_dir, img_file)
            image = Image.open(img_path).convert('RGB')
            original_size = image.size
            
            # 尝试两种预处理方式
            for norm_type, img_transform in [('with_norm', img_transform_with_norm), 
                                           ('no_norm', img_transform_no_norm)]:
                
                # 预处理
                img_tensor = img_transform(image).unsqueeze(0).to(device)
                
                # 推理
                result = model(img_tensor)
                
                # 获取shadow mask - 应用sigmoid和适当的阈值
                shadow_logits = result['logit']
                shadow_mask = torch.sigmoid(shadow_logits).cpu()
                
                # 调整回原始尺寸
                shadow_mask_resized = F.interpolate(shadow_mask, size=original_size[::-1], mode='bilinear', align_corners=False)
                
                # 应用阈值得到二值mask
                binary_mask = (shadow_mask_resized > 0.5).float()
                
                # 保存原图和shadow mask的对比
                original_tensor = transforms.ToTensor()(image)
                
                # 将shadow mask转为3通道用于可视化
                mask_3ch = shadow_mask_resized[0].repeat(3, 1, 1)
                binary_mask_3ch = binary_mask[0].repeat(3, 1, 1)
                
                # 创建叠加效果 (原图上叠加红色shadow mask)
                overlay = original_tensor.clone()
                mask_2d = binary_mask[0][0]  # 取出2D mask
                overlay[0] = torch.where(mask_2d > 0.5, torch.tensor(1.0), original_tensor[0])  # 红色通道
                overlay[1] = torch.where(mask_2d > 0.5, torch.tensor(0.0), original_tensor[1])  # 绿色通道置0
                overlay[2] = torch.where(mask_2d > 0.5, torch.tensor(0.0), original_tensor[2])  # 蓝色通道置0
                
                # 保存结果
                base_name = os.path.splitext(img_file)[0]
                save_path = os.path.join(save_dir, f"{base_name}_{config['name']}_{norm_type}_result.png")
                torchvision.utils.save_image([original_tensor, mask_3ch, binary_mask_3ch, overlay], 
                                           save_path, nrow=4, padding=2)
                
                # 单独保存shadow mask
                mask_save_path = os.path.join(save_dir, f"{base_name}_{config['name']}_{norm_type}_mask.png")
                torchvision.utils.save_image(binary_mask[0], mask_save_path)
                
                print(f"  - {norm_type}: {save_path}")
                
                # 打印一些统计信息
                mask_ratio = torch.sum(binary_mask).item() / binary_mask.numel()
                print(f"    Shadow ratio: {mask_ratio:.3f}")

print("\nDemo processing completed!")
print(f"Results saved in: {save_dir}")
print("Files saved:")
print("  - *_result.png: [Original | Soft Mask | Binary Mask | Overlay]") 
print("  - *_mask.png: Binary shadow mask only")
print("\nCompare results from different models and normalization settings to find the best one.")
