import torch
import torchvision
from networks.sddnet import SDDNet
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import cv2

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 模型参数
ckpt_path = './sddnet_ckpt/sbu.ckpt'  # 使用预训练的SBU模型
demo_dir = './demo'
save_dir = './demo_results'
os.makedirs(save_dir, exist_ok=True)

# 创建模型 - 不使用预训练权重，直接从checkpoint加载
model = SDDNet(backbone='efficientnet-b3',
               proj_planes=16,
               pred_planes=32,
               use_pretrained=False,  # 改为False，不下载预训练权重
               fix_backbone=False,
               has_se=False,
               dropout_2d=0,
               normalize=True,
               mu_init=0.4,
               reweight_mode='manual')

# 加载预训练权重
try:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f"Successfully loaded checkpoint from {ckpt_path}")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    print("Please make sure you have the pretrained model in the correct path.")
    exit()

model.to(device)
model.eval()

# 图像预处理
img_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 处理demo目录中的所有图片
img_files = [f for f in os.listdir(demo_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not img_files:
    print("No image files found in demo directory!")
    exit()

print(f"Found {len(img_files)} image(s) in demo directory")

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
        shadow_mask = torch.sigmoid(result['logit']).cpu()
        
        # 调整回原始尺寸
        shadow_mask_resized = transforms.Resize(original_size[::-1])(shadow_mask[0])
        
        # 保存原图和shadow mask的对比
        # 创建三列对比图：原图 | shadow mask | 叠加效果
        original_tensor = transforms.ToTensor()(image)
        
        # 将shadow mask转为3通道用于可视化
        mask_3ch = shadow_mask_resized.repeat(3, 1, 1)
        
        # 创建叠加效果 (原图上叠加红色shadow mask)
        overlay = original_tensor.clone()
        overlay[0] = torch.where(shadow_mask_resized[0] > 0.5, 
                                torch.tensor(1.0), original_tensor[0])  # 红色通道
        
        # 保存结果
        save_path = os.path.join(save_dir, f"{os.path.splitext(img_file)[0]}_result.png")
        torchvision.utils.save_image([original_tensor, mask_3ch, overlay], 
                                   save_path, nrow=3, padding=2)
        
        # 单独保存shadow mask
        mask_save_path = os.path.join(save_dir, f"{os.path.splitext(img_file)[0]}_mask.png")
        torchvision.utils.save_image(shadow_mask_resized, mask_save_path)
        
        print(f"Results saved to {save_path} and {mask_save_path}")

print("Demo processing completed!")
print(f"Results saved in: {save_dir}")
print("Files saved:")
print("  - *_result.png: [Original | Shadow Mask | Overlay]")
print("  - *_mask.png: Shadow mask only")
