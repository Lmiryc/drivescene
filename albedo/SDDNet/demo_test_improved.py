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

demo_dir = './demo'
save_dir = './demo_results_improved'
os.makedirs(save_dir, exist_ok=True)

# 尝试不同的预训练模型
model_configs = [
    {'name': 'ISTD', 'path': './sddnet_ckpt/istd.ckpt'},
    {'name': 'SBU', 'path': './sddnet_ckpt/sbu.ckpt'},
    {'name': 'UCF', 'path': './sddnet_ckpt/ucf.ckpt'}
]

def create_model():
    return SDDNet(backbone='efficientnet-b3',
                  proj_planes=16,
                  pred_planes=32,
                  use_pretrained=False,  # 不使用预训练，直接加载完整模型
                  fix_backbone=False,
                  has_se=False,
                  dropout_2d=0,
                  normalize=True,
                  mu_init=0.4,
                  reweight_mode='manual')

# 图像预处理 - 尝试不同的预处理方式
def get_transform(normalize=True):
    transforms_list = [
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ]
    if normalize:
        transforms_list.append(
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
    return transforms.Compose(transforms_list)

# 处理demo目录中的所有图片
img_files = [f for f in os.listdir(demo_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not img_files:
    print("No image files found in demo directory!")
    exit()

print(f"Found {len(img_files)} image(s) in demo directory")

for model_config in model_configs:
    print(f"\n=== Testing with {model_config['name']} model ===")
    
    try:
        # 创建模型
        model = create_model()
        
        # 加载预训练权重
        ckpt = torch.load(model_config['path'], map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Successfully loaded {model_config['name']} checkpoint")
        
        model.to(device)
        model.eval()
        
        # 尝试不同的预处理方式
        for normalize in [True, False]:
            norm_suffix = "_normalized" if normalize else "_unnormalized"
            
            img_transform = get_transform(normalize)
            
            with torch.no_grad():
                for img_file in img_files:
                    print(f"Processing {img_file} with {model_config['name']} ({norm_suffix})...")
                    
                    # 读取图像
                    img_path = os.path.join(demo_dir, img_file)
                    image = Image.open(img_path).convert('RGB')
                    original_size = image.size
                    
                    # 预处理
                    img_tensor = img_transform(image).unsqueeze(0).to(device)
                    
                    # 推理
                    result = model(img_tensor)
                    
                    # 获取shadow mask并尝试不同的阈值
                    shadow_logits = result['logit'].cpu()
                    shadow_probs = torch.sigmoid(shadow_logits)
                    
                    # 尝试不同阈值
                    thresholds = [0.3, 0.5, 0.7]
                    
                    for threshold in thresholds:
                        shadow_mask = (shadow_probs > threshold).float()
                        
                        # 调整回原始尺寸
                        shadow_mask_resized = transforms.Resize(original_size[::-1])(shadow_mask[0])
                        shadow_probs_resized = transforms.Resize(original_size[::-1])(shadow_probs[0])
                        
                        # 创建可视化
                        original_tensor = transforms.ToTensor()(image)
                        
                        # 概率图（灰度）
                        prob_3ch = shadow_probs_resized.repeat(3, 1, 1)
                        
                        # 二值掩码（黑白）
                        mask_3ch = shadow_mask_resized.repeat(3, 1, 1)
                        
                        # 叠加效果 (原图上叠加红色shadow mask)
                        overlay = original_tensor.clone()
                        mask_indices = shadow_mask_resized[0] > 0.5
                        overlay[0][mask_indices] = 1.0  # 红色通道
                        
                        # 保存结果
                        filename = f"{os.path.splitext(img_file)[0]}_{model_config['name']}{norm_suffix}_th{threshold}"
                        save_path = os.path.join(save_dir, f"{filename}_result.png")
                        
                        torchvision.utils.save_image([original_tensor, prob_3ch, mask_3ch, overlay], 
                                                   save_path, nrow=4, padding=2)
                        
                        print(f"  Saved: {filename}_result.png (threshold={threshold})")
        
    except Exception as e:
        print(f"Error with {model_config['name']} model: {e}")
        continue

print(f"\nAll results saved in: {save_dir}")
print("File naming convention: [image]_[model]_[normalization]_[threshold]_result.png")
print("Columns: [Original | Probability | Binary Mask | Overlay]")
