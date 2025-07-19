import os
import json
import shutil
from PIL import Image
import random
from collections import defaultdict

def organize_imagenet_dataset(source_dir, output_dir="./organized_dataset"):
    # 创建输出目录结构
    os.makedirs(output_dir, exist_ok=True)
    imgs_dir = os.path.join(output_dir, "imgs")
    os.makedirs(imgs_dir, exist_ok=True)
    
    # 存储图像路径到标签的映射
    imagenet_map = {}
    
    # 存储标签到单个图像路径的映射（每个类别只有一个示例）
    targeted_map = {}
    
    # 获取WordNet ID到数值标签的映射
    wordnet_to_label = {}
    image_count = 0
    
    # 第一步：收集所有WordNet ID
    wordnet_ids = []
    for dir_name in os.listdir(source_dir):
        if os.path.isdir(os.path.join(source_dir, dir_name)) and dir_name.startswith('n'):
            wordnet_ids.append(dir_name)
    
    # 为WordNet ID分配标签
    for idx, wordnet_id in enumerate(sorted(wordnet_ids)):
        wordnet_to_label[wordnet_id] = idx
    
    print(f"找到 {len(wordnet_ids)} 个WordNet ID")
    
    # 第二步：处理所有图像
    for wordnet_id in wordnet_ids:
        label = wordnet_to_label[wordnet_id]
        dir_path = os.path.join(source_dir, wordnet_id)
        
        # 跳过非目录
        if not os.path.isdir(dir_path):
            continue
        
        # 用于记录当前类别的第一张图片路径
        first_image_path = None
        
        # 处理目录中的所有图像
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # 生成新图像名称
                    new_img_name = f"{image_count}.jpeg"
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(imgs_dir, new_img_name)
                    
                    # 转换为JPEG并保存
                    try:
                        with Image.open(src_path) as img:
                            img = img.convert('RGB')
                            img.save(dst_path, 'JPEG')
                            
                            # 添加到imagenet.json映射
                            rel_path = f"imgs/{new_img_name}"
                            imagenet_map[rel_path] = label
                            
                            # 记录该类别的第一张图片路径用于targeted映射
                            if first_image_path is None:
                                first_image_path = rel_path
                                targeted_map[str(label)] = [rel_path]  # 每个类别只保存一张图片
                            
                            image_count += 1
                            
                            if image_count % 100 == 0:
                                print(f"已处理 {image_count} 张图片")
                    except Exception as e:
                        print(f"处理 {src_path} 时出错: {e}")
    
    # 确保每个标签在targeted_map中至少有一个图像
    for label in range(len(wordnet_ids)):
        str_label = str(label)
        if str_label not in targeted_map:
            print(f"警告: 标签 {label} 没有找到图像")
    
    # 保存imagenet.json
    with open(os.path.join(output_dir, "imagenet.json"), 'w') as f:
        json.dump(imagenet_map, f, indent=2)
    
    # 保存imagenet_targeted.json
    with open(os.path.join(output_dir, "imagenet_targeted.json"), 'w') as f:
        json.dump(targeted_map, f, indent=2)
    
    print(f"总共处理了 {image_count} 张图片")
    print(f"在imagenet.json中生成了 {len(imagenet_map)} 个条目")
    print(f"在imagenet_targeted.json中生成了 {len(targeted_map)} 个类别")

if __name__ == "__main__":
    # 使用当前目录作为源目录
    source_directory = "."
    organize_imagenet_dataset(source_directory)