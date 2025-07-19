import os
import json
import torch
import torchvision
from torchvision.datasets.utils import download_and_extract_archive


class TorchDatasetDownloader:
    def __init__(self, dataset_name: str, root: str = './data'):
        """
        使用torch工具下载数据集

        :param dataset_name: 数据集名称
        :param root: 数据集根目录
        """
        self.dataset_name = dataset_name.lower()
        self.root = os.path.join(root, dataset_name)
        self.imgs_dir = os.path.join(self.root, 'imgs')

        # 确保目录存在
        os.makedirs(self.imgs_dir, exist_ok=True)

        # 数据集下载配置
        self.dataset_configs = {
            'celebahq': {
                'url': 'https://drive.google.com/uc?id=1tklDiQlmnSMWDqL7aMsNlzh5Ub-rpNBK',
                'filename': 'celebahq.zip'
            },
            'cifar10': {
                'download': True
            },
            'iot_sqa': {
                'url': 'https://zenodo.org/record/4321356/files/IoT-SQA.zip',
                'filename': 'iot_sqa.zip'
            }
        }

    def download_celebahq(self):
        """
        下载CelebA-HQ数据集
        """
        import gdown

        config = self.dataset_configs['celebahq']
        zip_path = os.path.join(self.root, config['filename'])

        # 使用gdown下载
        gdown.download(config['url'], zip_path, quiet=False)

        # 解压
        torchvision.datasets.utils.extract_archive(zip_path, self.imgs_dir)

        # 清理zip文件
        os.remove(zip_path)

    def download_cifar10(self):
        """
        下载CIFAR-10数据集
        """
        # 使用torchvision内置下载器
        torchvision.datasets.CIFAR10(
            root=self.root,
            train=True,
            download=True
        )
        torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True
        )

        # 移动图像到imgs目录
        import shutil

        train_data_path = os.path.join(self.root, 'cifar-10-batches-py', 'data_batch_1')
        test_data_path = os.path.join(self.root, 'cifar-10-batches-py', 'test_batch')

        # TODO: 添加解析CIFAR-10二进制文件并转换为图像的代码

    def download_iot_sqa(self):
        """
        下载IOT-SQA数据集
        """
        config = self.dataset_configs['iot_sqa']
        download_and_extract_archive(
            config['url'],
            download_root=self.root,
            extract_root=self.imgs_dir,
            filename=config['filename']
        )

    def generate_json_files(self):
        """
        生成数据集的JSON映射文件
        """
        dataset_map = {}
        targeted_map = {}

        # 遍历图像文件
        for idx, img in enumerate(os.listdir(self.imgs_dir)):
            if img.endswith(('.jpg', '.png', '.JPEG')):
                dataset_map[f"imgs/{img}"] = str(idx % 10)  # 简单分类

                # 为targeted attack准备初始化图像
                class_key = str(idx % 10)
                if class_key not in targeted_map:
                    targeted_map[class_key] = [f"imgs/{img}"]

        # 写入JSON文件
        with open(os.path.join(self.root, f'{self.dataset_name}.json'), 'w') as f:
            json.dump(dataset_map, f, indent=2)

        with open(os.path.join(self.root, f'{self.dataset_name}_targeted.json'), 'w') as f:
            json.dump(targeted_map, f, indent=2)

    def download(self):
        """
        主下载方法
        """
        print(f"开始下载 {self.dataset_name.upper()} 数据集...")

        # 根据数据集名称选择下载方法
        download_method = getattr(self, f'download_{self.dataset_name}', None)

        if download_method:
            download_method()
            self.generate_json_files()
            print(f"{self.dataset_name.upper()}数据集下载完成!")
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")


# 使用示例
if __name__ == '__main__':
    # 下载CelebA-HQ
    # celebahq_downloader = TorchDatasetDownloader('celebahq')
    # celebahq_downloader.download()

    # 下载IOT-SQA
    iot_sqa_downloader = TorchDatasetDownloader('iot_sqa')
    iot_sqa_downloader.download()

    # # 下载CIFAR-10
    # cifar10_downloader = TorchDatasetDownloader('cifar10')
    # cifar10_downloader.download()