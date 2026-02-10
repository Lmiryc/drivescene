import torch
import torchvision
from networks.sddnet import SDDNet
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import torch.nn.functional as F
from pathlib import Path

def load_model_checkpoint(model, ckpt_path, device):
    """
    尝试不同的方式加载checkpoint
    """
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        
        # 处理不同格式的checkpoint
        if isinstance(ckpt, dict):
            if 'model' in ckpt:
                # 标准格式：{'model': state_dict, 'epoch': ..., 'optimizer': ...}
                model.load_state_dict(ckpt['model'])
                info = f"Epoch: {ckpt.get('epoch', 'N/A')}"
                if 'best_ber' in ckpt:
                    info += f", Best BER: {ckpt['best_ber']:.4f}"
                return True, info
            elif 'state_dict' in ckpt:
                # 另一种格式
                model.load_state_dict(ckpt['state_dict'])
                return True, "Loaded from state_dict"
            else:
                # 直接是state_dict
                model.load_state_dict(ckpt)
                return True, "Direct state_dict"
        else:
            # 可能是直接的state_dict
            model.load_state_dict(ckpt)
            return True, "Direct load"
            
    except Exception as e:
        # 尝试strict=False
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                model.load_state_dict(ckpt['model'], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
            return True, f"Loaded with strict=False (some keys missing): {str(e)[:100]}"
        except Exception as e2:
            return False, f"Failed to load: {str(e2)[:100]}"

def process_demo_with_all_models():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 定义所有模型路径
    ckpt_dir = './ckpt'
    model_files = [f for f in os.listdir(ckpt_dir) 
                   if f.endswith(('.ckpt', '.pth')) and f != 'placeholder.md']
    
    demo_dir = './demo'
    save_dir = './demo_results_all_models'
    os.makedirs(save_dir, exist_ok=True)

    print(f"Found {len(model_files)} model files:")
    for f in model_files:
        print(f"  - {f}")

    # 预处理配置
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

    # 获取demo图像
    img_files = [f for f in os.listdir(demo_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        print("No image files found in demo directory!")
        return

    # 为每个模型创建子目录
    results_summary = []

    for model_file in model_files:
        print(f"\n{'='*60}")
        print(f"Testing model: {model_file}")
        print(f"{'='*60}")
        
        model_name = Path(model_file).stem
        model_save_dir = os.path.join(save_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)
        
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

        # 加载模型
        ckpt_path = os.path.join(ckpt_dir, model_file)
        success, info = load_model_checkpoint(model, ckpt_path, device)
        
        if not success:
            print(f"❌ Failed to load {model_file}: {info}")
            continue
            
        print(f"✅ Successfully loaded {model_file}")
        print(f"   Info: {info}")
        
        model.to(device)
        model.eval()

        # 处理每张图像
        model_results = []
        
        with torch.no_grad():
            for img_file in img_files:
                print(f"\n  Processing {img_file}...")
                
                # 读取图像
                img_path = os.path.join(demo_dir, img_file)
                image = Image.open(img_path).convert('RGB')
                original_size = image.size
                base_name = os.path.splitext(img_file)[0]
                
                img_results = []
                
                for config in preprocess_configs:
                    print(f"    Testing {config['name']} preprocessing...")
                    
                    # 预处理
                    img_tensor = config['transform'](image).unsqueeze(0).to(device)
                    
                    # 推理
                    try:
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
                        comparison_path = os.path.join(model_save_dir, 
                                                     f"{base_name}_{config['name']}_comparison.png")
                        torchvision.utils.save_image([original_tensor, soft_mask_3ch, binary_mask_3ch, overlay], 
                                                   comparison_path, nrow=4, padding=2)
                        
                        # 单独保存mask
                        mask_path = os.path.join(model_save_dir, f"{base_name}_{config['name']}_mask.png")
                        torchvision.utils.save_image(binary_mask[0], mask_path)
                        
                        # 统计信息
                        mask_ratio = torch.sum(binary_mask).item() / binary_mask.numel()
                        
                        result_info = {
                            'model': model_name,
                            'image': img_file,
                            'preprocessing': config['name'],
                            'shadow_ratio': mask_ratio,
                            'comparison_path': comparison_path,
                            'mask_path': mask_path,
                            'status': 'success'
                        }
                        
                        print(f"      Shadow ratio: {mask_ratio:.1%}")
                        img_results.append(result_info)
                        
                    except Exception as e:
                        print(f"      ❌ Error during inference: {str(e)[:100]}")
                        result_info = {
                            'model': model_name,
                            'image': img_file,
                            'preprocessing': config['name'],
                            'status': 'error',
                            'error': str(e)
                        }
                        img_results.append(result_info)
                
                model_results.extend(img_results)
        
        results_summary.extend(model_results)
        print(f"✅ Completed testing {model_file}")

    # 生成总结报告
    print(f"\n{'='*80}")
    print("SUMMARY REPORT")
    print(f"{'='*80}")
    
    # 创建综合对比图
    create_comparison_grid(results_summary, save_dir)
    
    # 打印结果统计
    success_results = [r for r in results_summary if r['status'] == 'success']
    
    print(f"\nTotal tests: {len(results_summary)}")
    print(f"Successful: {len(success_results)}")
    print(f"Failed: {len(results_summary) - len(success_results)}")
    
    if success_results:
        print(f"\nShadow ratio statistics:")
        for img in set(r['image'] for r in success_results):
            print(f"\n  {img}:")
            img_results = [r for r in success_results if r['image'] == img]
            for result in img_results:
                print(f"    {result['model']:15s} ({result['preprocessing']:8s}): {result['shadow_ratio']:.1%}")
    
    print(f"\n📁 All results saved in: {save_dir}")
    print("📁 Each model has its own subdirectory with results")

def create_comparison_grid(results_summary, save_dir):
    """
    为每张图像创建所有模型的对比网格
    """
    try:
        success_results = [r for r in results_summary if r['status'] == 'success']
        if not success_results:
            return
            
        images = list(set(r['image'] for r in success_results))
        models = list(set(r['model'] for r in success_results))
        
        for img in images:
            print(f"\nCreating comparison grid for {img}...")
            
            img_results = [r for r in success_results if r['image'] == img]
            base_name = os.path.splitext(img)[0]
            
            # 为每种预处理方式创建对比图
            for preprocessing in ['no_norm', 'with_norm']:
                prep_results = [r for r in img_results if r['preprocessing'] == preprocessing]
                
                if len(prep_results) < 2:
                    continue
                    
                # 收集所有comparison图像
                comparison_images = []
                labels = []
                
                for result in prep_results:
                    if os.path.exists(result['comparison_path']):
                        # 读取comparison图像
                        img_tensor = torchvision.io.read_image(result['comparison_path'])
                        comparison_images.append(img_tensor.float() / 255.0)
                        labels.append(f"{result['model']} ({result['shadow_ratio']:.1%})")
                
                if comparison_images:
                    # 创建网格
                    grid_path = os.path.join(save_dir, f"{base_name}_{preprocessing}_all_models_grid.png")
                    
                    # 计算网格布局
                    n_images = len(comparison_images)
                    ncols = min(2, n_images)
                    nrows = (n_images + ncols - 1) // ncols
                    
                    torchvision.utils.save_image(comparison_images, grid_path, 
                                               nrow=ncols, padding=10)
                    
                    # 保存标签信息
                    labels_path = os.path.join(save_dir, f"{base_name}_{preprocessing}_labels.txt")
                    with open(labels_path, 'w') as f:
                        f.write(f"Comparison grid for {img} with {preprocessing} preprocessing:\n\n")
                        for i, label in enumerate(labels):
                            f.write(f"Row {i//ncols + 1}, Col {i%ncols + 1}: {label}\n")
                    
                    print(f"  Created grid: {grid_path}")
    
    except Exception as e:
        print(f"Error creating comparison grid: {e}")

if __name__ == "__main__":
    process_demo_with_all_models()
