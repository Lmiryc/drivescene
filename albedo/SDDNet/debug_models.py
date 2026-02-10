import torch
import torchvision
from networks.sddnet import SDDNet
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

def debug_problematic_models():
    """
    调试有问题的模型，分析它们的输出
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 重点分析有问题的模型
    problematic_models = [
        'fdr_cuhk.ckpt',   # 100% shadow
        'fdr_sbu.ckpt',    # 100% shadow  
        'cuhkdsdnet.pth',  # 0% shadow
        'dsd_sub.pth'      # 0% shadow
    ]
    
    # 对比正常的模型
    normal_model = 'ep_019.ckpt'
    
    demo_dir = './demo'
    debug_dir = './debug_models'
    os.makedirs(debug_dir, exist_ok=True)

    img_files = [f for f in os.listdir(demo_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        print("No demo images found!")
        return
    
    # 使用第一张图片进行调试
    test_img = img_files[0]
    print(f"Using test image: {test_img}")
    
    img_path = os.path.join(demo_dir, test_img)
    image = Image.open(img_path).convert('RGB')
    
    # 预处理
    img_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    img_tensor = img_transform(image).unsqueeze(0).to(device)
    
    print(f"Input tensor shape: {img_tensor.shape}")
    print(f"Input tensor range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

    # 首先测试正常模型作为基准
    print(f"\n{'='*50}")
    print(f"Testing NORMAL model: {normal_model}")
    print(f"{'='*50}")
    
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
    
    # 加载正常模型
    try:
        ckpt_path = os.path.join('./ckpt', normal_model)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            result = model(img_tensor)
            logits = result['logit']
            
            print(f"Normal model output:")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
            print(f"  Logits mean: {logits.mean():.3f}")
            print(f"  Logits std: {logits.std():.3f}")
            
            sigmoid_output = torch.sigmoid(logits)
            print(f"  Sigmoid range: [{sigmoid_output.min():.3f}, {sigmoid_output.max():.3f}]")
            print(f"  Sigmoid mean: {sigmoid_output.mean():.3f}")
            
            binary_mask = (sigmoid_output > 0.5).float()
            shadow_ratio = torch.sum(binary_mask).item() / binary_mask.numel()
            print(f"  Shadow ratio: {shadow_ratio:.1%}")
            
    except Exception as e:
        print(f"Error with normal model: {e}")

    # 现在测试有问题的模型
    for model_file in problematic_models:
        print(f"\n{'='*50}")
        print(f"Debugging: {model_file}")
        print(f"{'='*50}")
        
        ckpt_path = os.path.join('./ckpt', model_file)
        if not os.path.exists(ckpt_path):
            print(f"Model file not found: {ckpt_path}")
            continue
            
        # 重新创建模型
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
        
        # 尝试加载模型
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            print(f"Checkpoint keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'Direct tensor'}")
            
            # 尝试不同的加载方式
            if isinstance(ckpt, dict):
                if 'model' in ckpt:
                    model.load_state_dict(ckpt['model'], strict=False)
                    print("Loaded from 'model' key")
                elif 'state_dict' in ckpt:
                    model.load_state_dict(ckpt['state_dict'], strict=False)
                    print("Loaded from 'state_dict' key")
                else:
                    model.load_state_dict(ckpt, strict=False)
                    print("Loaded as direct state_dict")
            else:
                model.load_state_dict(ckpt, strict=False)
                print("Loaded as direct tensor")
                
        except Exception as e:
            print(f"❌ Failed to load {model_file}: {e}")
            continue
            
        model.to(device)
        model.eval()
        
        # 测试推理
        try:
            with torch.no_grad():
                result = model(img_tensor)
                
                # 检查输出格式
                print(f"Output keys: {list(result.keys())}")
                
                if 'logit' in result:
                    logits = result['logit']
                    
                    print(f"Model output analysis:")
                    print(f"  Logits shape: {logits.shape}")
                    print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
                    print(f"  Logits mean: {logits.mean():.3f}")
                    print(f"  Logits std: {logits.std():.3f}")
                    
                    # 检查是否有异常值
                    if torch.isnan(logits).any():
                        print("  ⚠️  Contains NaN values!")
                    if torch.isinf(logits).any():
                        print("  ⚠️  Contains Inf values!")
                    
                    sigmoid_output = torch.sigmoid(logits)
                    print(f"  Sigmoid range: [{sigmoid_output.min():.3f}, {sigmoid_output.max():.3f}]")
                    print(f"  Sigmoid mean: {sigmoid_output.mean():.3f}")
                    
                    # 尝试不同的阈值
                    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
                    print(f"  Shadow ratios at different thresholds:")
                    for thresh in thresholds:
                        binary_mask = (sigmoid_output > thresh).float()
                        ratio = torch.sum(binary_mask).item() / binary_mask.numel()
                        print(f"    Threshold {thresh}: {ratio:.1%}")
                    
                    # 保存输出进行可视化
                    save_debug_visualization(image, sigmoid_output, model_file, debug_dir)
                    
                else:
                    print(f"  ❌ No 'logit' key in output!")
                    
        except Exception as e:
            print(f"❌ Error during inference: {e}")

def save_debug_visualization(original_image, sigmoid_output, model_name, debug_dir):
    """
    保存调试可视化结果
    """
    try:
        # 调整到原始图像尺寸
        original_tensor = transforms.ToTensor()(original_image)
        h, w = original_tensor.shape[1], original_tensor.shape[2]
        
        mask_resized = F.interpolate(sigmoid_output.cpu(), size=(h, w), mode='bilinear', align_corners=False)
        
        # 创建不同阈值的可视化
        thresholds = [0.3, 0.5, 0.7]
        visualization_tensors = [original_tensor]
        
        # 软mask
        soft_mask_3ch = mask_resized[0].repeat(3, 1, 1)
        visualization_tensors.append(soft_mask_3ch)
        
        # 不同阈值的二值mask
        for thresh in thresholds:
            binary_mask = (mask_resized > thresh).float()
            binary_mask_3ch = binary_mask[0].repeat(3, 1, 1)
            visualization_tensors.append(binary_mask_3ch)
        
        # 保存
        model_clean_name = os.path.splitext(model_name)[0]
        save_path = os.path.join(debug_dir, f"{model_clean_name}_debug.png")
        torchvision.utils.save_image(visualization_tensors, save_path, nrow=len(visualization_tensors), padding=2)
        
        print(f"  Debug visualization saved: {save_path}")
        
    except Exception as e:
        print(f"  Error saving debug visualization: {e}")

if __name__ == "__main__":
    debug_problematic_models()
