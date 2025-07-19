import os
import json
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm


def convert_cifar_to_images():
    # 创建必要的目录结构
    os.makedirs('../data/cifar10/imgs', exist_ok=True)  # 修改：创建imgs子目录

    # 使用torchvision加载CIFAR-10数据集
    train_dataset = CIFAR10('./data', train=True, download=True)
    test_dataset = CIFAR10('./data', train=False, download=True)

    # 合并训练集和测试集的数据和标签
    all_images = np.concatenate([train_dataset.data, test_dataset.data], axis=0)
    all_labels = np.concatenate([train_dataset.targets, test_dataset.targets], axis=0)

    # 准备JSON数据
    image_dict = {}
    targeted_dict = {str(i): [] for i in range(10)}

    # 转换并保存每张图片
    # zip 创建一个元组迭代器，每个元组包含一个图像和一个标签
    # idx 是图像的索引，image 是一个32x32x3的numpy数组，label 是一个整数
    for idx, (image, label) in enumerate(tqdm(zip(all_images, all_labels), desc="Converting CIFAR-10")):
        # 修改：简化文件名格式
        image_filename = f'{idx}.png'
        image_path = os.path.join('../data/cifar10/imgs', image_filename)

        # 将numpy数组转换为PIL图像并保存
        image_pil = Image.fromarray(image)
        image_pil.save(image_path)

        # 修改：添加imgs/前缀
        image_dict[f'imgs/{image_filename}'] = int(label)

        # 修改：为每个类别保存第一个遇到的样本路径
        if not targeted_dict[str(int(label))]:
            targeted_dict[str(int(label))].append(f'imgs/{image_filename}')

    # 保存主JSON文件
    with open('../data/cifar10/cifar10.json', 'w') as f:
        json.dump(image_dict, f, indent=4)

    # 保存目标攻击JSON文件
    with open('../data/cifar10/cifar10_targeted.json', 'w') as f:
        json.dump(targeted_dict, f, indent=4)

    print(f"Converted {len(image_dict)} images")
    print("Files saved in ./data/cifar10/")
    print("Created: cifar10.json and cifar10_targeted.json")


if __name__ == "__main__":
    convert_cifar_to_images()