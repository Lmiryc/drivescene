import torch
import torchvision
from networks.sddnet import SDDNet
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import torch.nn.functional as F

def process_demo_with_optimized_settings():
    """
    使用调试结果优化后的设置处理demo图像
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    demo_dir = './demo'
    save_dir = './demo_results_optimized'
    os.makedirs(save_dir, exist_ok=True)

    # 根据调试结果定义模型配置
    model_configs = [
        {
            'name': 'ep_019_best',
            'path': './ckpt/ep_019.ckpt',
            'threshold': 0.5,
            'description': '最佳模型 - 正常工作'
        },
        {
            'name': 'fdr_cuhk_fixed',
            'path': './ckpt/fdr_cuhk.ckpt', 
            'threshold': 0.52,  # 调整阈值以适应其输出特性
            'description': 'FDR CUHK - 调整阈值'
        },
        {
            'name': 'fdr_sbu_fixed',
            'path': './ckpt/fdr_sbu.ckpt',
            'threshold': 0.535,  # 调整阈值
            'description': 'FDR SBU - 调整阈值'
        }
    ]

    # 预处理
    img_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # 获取demo图像
    img_files = [f for f in os.listdir(demo_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        print("No demo images found!")
        return

    print(f"Processing {len(img_files)} image(s) with optimized settings...")

    results_summary = []

    for config in model_configs:
        print(f"\n{'='*60}")
        print(f"Processing with {config['name']}: {config['description']}")
        print(f"Using threshold: {config['threshold']}")
        print(f"{'='*60}")
        
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
        try:
            ckpt = torch.load(config['path'], map_location=device)
            if 'model' in ckpt:
                model.load_state_dict(ckpt['model'])
            else:
                model.load_state_dict(ckpt)
            print(f"✅ Successfully loaded {config['name']}")
        except Exception as e:
            print(f"❌ Failed to load {config['name']}: {e}")
            continue
            
        model.to(device)
        model.eval()

        # 为每个模型创建子目录
        model_save_dir = os.path.join(save_dir, config['name'])
        os.makedirs(model_save_dir, exist_ok=True)

        with torch.no_grad():
            for img_file in img_files:
                print(f"  Processing {img_file}...")
                
                # 读取图像
                img_path = os.path.join(demo_dir, img_file)
                image = Image.open(img_path).convert('RGB')
                original_size = image.size
                base_name = os.path.splitext(img_file)[0]
                
                # 预处理
                img_tensor = img_transform(image).unsqueeze(0).to(device)
                
                # 推理
                try:
                    result = model(img_tensor)
                    shadow_logits = result['logit']
                    shadow_mask_soft = torch.sigmoid(shadow_logits).cpu()
                    
                    # 调整回原始尺寸
                    shadow_mask_resized = F.interpolate(shadow_mask_soft, size=original_size[::-1], 
                                                      mode='bilinear', align_corners=False)
                    
                    # 使用优化的阈值
                    binary_mask = (shadow_mask_resized > config['threshold']).float()
                    
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
                    comparison_path = os.path.join(model_save_dir, f"{base_name}_comparison.png")
                    torchvision.utils.save_image([original_tensor, soft_mask_3ch, binary_mask_3ch, overlay], 
                                               comparison_path, nrow=4, padding=2)
                    
                    # 单独保存mask
                    mask_path = os.path.join(model_save_dir, f"{base_name}_mask.png")
                    torchvision.utils.save_image(binary_mask[0], mask_path)
                    
                    # 统计信息
                    mask_ratio = torch.sum(binary_mask).item() / binary_mask.numel()
                    
                    # 记录结果
                    result_info = {
                        'model': config['name'],
                        'image': img_file,
                        'threshold': config['threshold'],
                        'shadow_ratio': mask_ratio,
                        'logits_range': [shadow_logits.min().item(), shadow_logits.max().item()],
                        'sigmoid_range': [shadow_mask_soft.min().item(), shadow_mask_soft.max().item()],
                        'comparison_path': comparison_path
                    }
                    results_summary.append(result_info)
                    
                    print(f"    Threshold: {config['threshold']:.3f}")
                    print(f"    Shadow ratio: {mask_ratio:.1%}")
                    print(f"    Logits range: [{shadow_logits.min():.3f}, {shadow_logits.max():.3f}]")
                    print(f"    Sigmoid range: [{shadow_mask_soft.min():.3f}, {shadow_mask_soft.max():.3f}]")
                    
                except Exception as e:
                    print(f"    ❌ Error during inference: {e}")

    # 创建最终对比
    print(f"\n{'='*80}")
    print("OPTIMIZED RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for img in set(r['image'] for r in results_summary):
        print(f"\n📸 {img}:")
        img_results = [r for r in results_summary if r['image'] == img]
        for result in img_results:
            print(f"  {result['model']:15s}: {result['shadow_ratio']:5.1%} "
                  f"(threshold: {result['threshold']:.3f})")
    
    # 创建最佳结果对比图
    create_best_comparison(results_summary, save_dir)
    
    print(f"\n✅ Optimized processing completed!")
    print(f"📁 Results saved in: {save_dir}")
    print(f"🎯 Best model appears to be: ep_019")

def create_best_comparison(results_summary, save_dir):
    """
    创建最佳结果的对比图
    """
    try:
        images = list(set(r['image'] for r in results_summary))
        
        for img in images:
            img_results = [r for r in results_summary if r['image'] == img]
            base_name = os.path.splitext(img)[0]
            
            # 收集所有comparison图像
            comparison_images = []
            labels = []
            
            for result in img_results:
                if os.path.exists(result['comparison_path']):
                    img_tensor = torchvision.io.read_image(result['comparison_path'])
                    comparison_images.append(img_tensor.float() / 255.0)
                    labels.append(f"{result['model']} ({result['shadow_ratio']:.1%})")
            
            if comparison_images and len(comparison_images) > 1:
                # 创建网格
                grid_path = os.path.join(save_dir, f"{base_name}_all_optimized_comparison.png")
                torchvision.utils.save_image(comparison_images, grid_path, nrow=1, padding=10)
                
                # 保存标签
                labels_path = os.path.join(save_dir, f"{base_name}_optimized_labels.txt")
                with open(labels_path, 'w') as f:
                    f.write(f"Optimized comparison for {img}:\n\n")
                    for i, label in enumerate(labels):
                        f.write(f"Row {i+1}: {label}\n")
                
                print(f"📊 Created optimized comparison: {grid_path}")
    
    except Exception as e:
        print(f"Error creating best comparison: {e}")

if __name__ == "__main__":
    process_demo_with_optimized_settings()
