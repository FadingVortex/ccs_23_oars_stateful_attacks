import os
import json
import numpy as np
from PIL import Image
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm


def convert_mnist_to_images():
    # 创建必要的目录结构
    os.makedirs('./data/mnist', exist_ok=True)

    # 使用torchvision加载MNIST数据集
    train_dataset = MNIST('./data', train=True, download=True)
    test_dataset = MNIST('./data', train=False, download=True)

    # 合并训练集和测试集的数据和标签
    all_images = np.concatenate([train_dataset.data.numpy(), test_dataset.data.numpy()], axis=0)
    all_labels = np.concatenate([train_dataset.targets.numpy(), test_dataset.targets.numpy()], axis=0)

    # 准备JSON数据
    image_dict = {}
    targeted_dict = {str(i): [] for i in range(10)}  # 为每个类别准备目标攻击样本

    # 转换并保存每张图片
    for idx, (image, label) in enumerate(tqdm(zip(all_images, all_labels), desc="Converting MNIST")):
        # 创建图片文件名
        image_filename = f'mnist_{idx}.png'
        image_path = os.path.join('./data/mnist', image_filename)

        # 将numpy数组转换为PIL图像并保存
        image_pil = Image.fromarray(image.astype(np.uint8), mode='L')
        image_pil.save(image_path)

        # 添加到主字典
        image_dict[image_filename] = int(label)

        # 为每个类别保存第一个遇到的样本作为目标攻击样本
        if not targeted_dict[str(int(label))]:
            targeted_dict[str(int(label))] = [image_filename, int(label)]

    # 保存主JSON文件
    with open('./data/mnist/mnist.json', 'w') as f:
        json.dump(image_dict, f)

    # 保存目标攻击JSON文件
    with open('./data/mnist/mnist_targeted.json', 'w') as f:
        json.dump(targeted_dict, f)

    print(f"Converted {len(image_dict)} images")
    print("Files saved in ./data/mnist/")
    print("Created: mnist.json and mnist_targeted.json")


if __name__ == "__main__":
    convert_mnist_to_images()