import os
import matplotlib
import quadprog

matplotlib.use('TkAgg')  # 或者 'Agg', 'Qt5Agg' 等其他后端
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
import numpy as np
from models.resnet import resnet20
from models.resnet_cl import resnet20 as resnet20_cl
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# CIFAR-10类别标签
cifar10_classes = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
class Args:
    def __init__(self, image_size=32, num_channels=3, num_classes=10):
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_trigger_set = 300
        self.device = 'cuda'
        self.test_bs = 512
        self.gem = True
        self.gpu = 0

def test_update_model():
    new_model = resnet20_cl().to("cuda")
    model_path = "../models/pretrained/resnet20-12fca82f-single.pth"
    # new_model.load_state_dict(torch.load(model_path, map_location="cuda").state_dict())
    new_model.load_global_model(torch.load(model_path, map_location="cuda").state_dict(), "cuda", watermark=True)
    torch.save(new_model, "../models/pretrained/resnet20-12fca82f-single-gem.pth")


def test_compare_insert():
    # test_insert_watermark()
    # test_insert_watermark(gem=False)
    # test_insert_watermark_with_memory_preservation(gem=False)
    test_optimized_watermark_training(gem=True)
    test_optimized_watermark_training(gem=False)


def test_insert_watermark(gem=True):
    # 加载模型
    model_path = "../models/pretrained/resnet20-12fca82f-single-gem.pth"
    # model_path = "../models/pretrained/resnet20-12fca82f-single.pth"
    global_model = torch.load(model_path, map_location="cuda")
    global_model = global_model.to('cuda')

    # 准备数据
    _, test_dataset = get_full_datasets()
    trigger_set = generate_waffle_pattern(Args())
    watermark_set = DataLoader(trigger_set, batch_size=16, shuffle=True)

    # 嵌入水印前评估
    watermark_acc, _ = test_img(global_model, trigger_set, Args())
    print(f"[Init ] Watermark acc: {watermark_acc:.2f}%")

    # 损失与优化器
    watermark_loss_func = torch.nn.CrossEntropyLoss()
    watermark_optim = torch.optim.SGD(global_model.parameters(), lr=0.0001, momentum=0.9)

    watermark_embed_iters = 0
    # 只要未达到 98% 且未超出最大迭代，就继续微调
    while watermark_acc <= 98 and watermark_embed_iters < 100:
        watermark_embed_iters += 1
        global_model.train()
        global_model.to('cuda')

        # 冻结所有 BN 层（保持 running stats 不变）
        global_model.apply(set_bn_eval)

        epoch_loss = 0.0
        for batch_idx, (images, labels) in enumerate(watermark_set):
            images, labels = images.to('cuda'), labels.long().to('cuda')
            watermark_optim.zero_grad()
            outputs = global_model(images)
            loss = watermark_loss_func(outputs, labels)
            loss.backward()
            if gem:
                global_model = gem_train(global_model)
            watermark_optim.step()
            epoch_loss += loss.item()

        # 评估当前水印集精度
        watermark_acc, _ = test_img(global_model, trigger_set, Args())
        avg_loss = epoch_loss / len(watermark_set)
        test_acc, _ = test_img(global_model, test_dataset, Args())
        print(f"[Iter {watermark_embed_iters:03d}] loss: {avg_loss:.4f}, watermark acc: {watermark_acc:.2f}%, test acc: {test_acc:.2f}%")

    print( f"=== Watermark embedding finished at iter {watermark_embed_iters}, final acc: {watermark_acc:.2f}% ===")

    # 重新生成 trigger 并再测一次
    trigger_set = generate_waffle_pattern(Args())
    watermark_acc, _ = test_img(global_model, trigger_set, Args())
    print(f"[Re-test] Watermark acc: {watermark_acc:.2f}%")

    # 测试正常数据集精度
    test_acc, _ = test_img(global_model, test_dataset, Args())
    print(f"[Final ] Test acc: {test_acc:.2f}%")

def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))
    return gradient

def gem_train(model):
    for name, param in model.named_parameters():
        grads = param.grad
        grad_size = grads.data.size()
        if grads is not None:
            memory = model.memory[name]
            # check
            dotp = torch.mul(grads.view(-1), memory.view(-1))
            if dotp.sum() < 0:
                newgrads = project2cone2(grads.view(-1).unsqueeze(1), memory.view(-1, 1)) # conduct projection
                param.grad.data.copy_(newgrads.view(grad_size))
    return model

def test_single_image():
    image_path = "../data/cifar10/imgs/0.png"
    model_path = "../models/pretrained/resnet20-12fca82f-single.pth"
    # Load the model
    # model = resnet20()
    # model.load_state_dict(torch.load(model_path, map_location="cuda"))
    model = torch.load(model_path, map_location="cuda")
    model.eval()

    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to('cuda')  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(output, dim=1).item()

    # show result
    plt.figure(figsize=(10, 5))

    # show the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Predicted Class: {cifar10_classes[predicted_class]}")
    plt.axis('off')

    # show the probabilities
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(cifar10_classes))
    plt.barh(y_pos, probabilities.to('cpu').numpy())
    plt.yticks(y_pos, cifar10_classes)
    plt.xlabel('Probability')
    plt.title('Class Probabilities')

    plt.tight_layout()
    plt.show()

    print("Predicted Class:", cifar10_classes[predicted_class])
    print("Probabilities:")
    for i, (cls, prob) in enumerate(zip(cifar10_classes, probabilities)):
        print(f"{cls}: {prob.item():.4f}")

def get_full_datasets():
    # Load the CIFAR-10 dataset
    train_dataset = CIFAR10('../../FedTracker/data/cifar10/', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                            ]))
    test_dataset = CIFAR10('../../FedTracker/data/cifar10/', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                           ]))
    return train_dataset, test_dataset

# Original acc: 91.73%
def test_multi_images():
    model_path = "../models/pretrained/resnet20-12fca82f-gem-finished.pth"
    model = torch.load(model_path, map_location="cuda")
    model.eval()

    test_dataset = CIFAR10('../../FedTracker/data/cifar10/', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                           ]))
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to("cuda")
            labels = labels.to("cuda")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

    # test watermark acc
    trigger_set = generate_waffle_pattern(Args())
    watermark_acc, _ = test_img(model, trigger_set, Args())
    print(f'Watermark accuracy: {watermark_acc:.2f}%')


# Finally, move the model back to CPU
def test_img(net_g, datatest, args):
    net_g.eval()
    net_g.to(args.device)
    # testing
    correct = 0
    correct_top5 = 0
    data_loader = DataLoader(datatest, batch_size=args.test_bs)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # get the index of the max log-probability
        _, y_pred = torch.max(log_probs.data, 1)
        correct += (y_pred == target).sum().item()
        _, pred = log_probs.topk(5, 1, True, True)
        target_resize = target.view(-1, 1)
        correct_top5 += torch.eq(pred, target_resize).sum().float().item()

    accuracy = 100.00 * correct / len(data_loader.dataset)
    accuracy_top5 = 100.00 * correct_top5 / len(data_loader.dataset)
    net_g.cpu()
    return accuracy, accuracy_top5


class NumpyLoader(Dataset):

    def __init__(self, x, y, transformer=None):
        self.x = x
        self.y = y
        self.transformer = transformer

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        image = self.x[item]
        label = self.y[item]
        if self.transformer is not None:
            image = self.transformer(image)
        return image, label


def generate_waffle_pattern(args):
    # np.random.seed(0)
    path = "../../FedTracker/data/pattern/"
    base_patterns = []
    # 加载触发集图案
    for i in range(args.num_classes):
        pattern_path = os.path.join(path, "{}.png".format(i))
        pattern = Image.open(pattern_path)
        if args.num_channels == 1:
            pattern = pattern.convert("L")
        else:
            pattern = pattern.convert("RGB")
        pattern = pattern.resize((args.image_size, args.image_size), Image.BILINEAR)
        pattern = np.array(pattern)
        # pattern = np.resize(pattern, (args.image_size, args.image_size, args.num_channels))
        base_patterns.append(pattern)
    trigger_set = []
    trigger_set_labels = []
    label = 0
    # num_trigger_each_class 每个类别的触发器数量
    num_trigger_each_class = args.num_trigger_set // args.num_classes
    for pattern in base_patterns:
        for _ in range(num_trigger_each_class):
            image = (pattern + np.random.randint(0, 255, (args.image_size, args.image_size, args.num_channels)))\
                        .astype(np.float32) / 255 / 2
            trigger_set.append(image)
            trigger_set_labels.append(label)
        label += 1
    trigger_set = np.array(trigger_set)
    trigger_set_labels = np.array(trigger_set_labels)
    # trigger_set_mean = np.mean(trigger_set, axis=(0, 1, 2))
    # trigger_set_std = np.std(trigger_set, axis=(0, 1, 2))
    # print(trigger_set_mean, trigger_set_std)

    # Warning : In this code, ResNet model will preprocess the image with cifar10 mean and std automatically
    # so, we don't need to normalize the image again
    dataset = NumpyLoader(trigger_set, trigger_set_labels, transformer=transforms.Compose([
                                    transforms.ToTensor(),
                                    # transforms.Normalize(trigger_set_mean, trigger_set_std)
                                ]))
    return dataset


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def test_insert_watermark_with_memory_preservation(gem=True):
    """
    修复版本：水印嵌入 + 记忆保持的完整训练流程
    """
    # 加载模型
    model_path = "../models/pretrained/resnet20-12fca82f-single-gem.pth"
    global_model = torch.load(model_path, map_location="cuda")
    global_model = global_model.to('cuda')

    # 准备数据
    train_dataset, test_dataset = get_full_datasets()
    trigger_set = generate_waffle_pattern(Args())
    watermark_set = DataLoader(trigger_set, batch_size=16, shuffle=True)

    # 创建原始任务的记忆保持数据集
    memory_preservation_size = 2000
    memory_indices = torch.randperm(len(train_dataset))[:memory_preservation_size]
    memory_dataset = torch.utils.data.Subset(train_dataset, memory_indices)
    memory_loader = DataLoader(memory_dataset, batch_size=32, shuffle=True)

    # 损失与优化器
    watermark_loss_func = torch.nn.CrossEntropyLoss()
    memory_loss_func = torch.nn.CrossEntropyLoss()
    # 关键修复1：使用相同的优化器，避免优化器状态冲突
    optimizer = torch.optim.SGD(global_model.parameters(), lr=0.0001, momentum=0.9)

    # === 阶段1: 水印嵌入 ===
    print("=== Phase 1: Watermark Embedding ===")
    watermark_acc, _ = test_img(global_model, trigger_set, Args())
    test_acc_init, _ = test_img(global_model, test_dataset, Args())
    global_model.to('cuda')
    print(f"[Init ] Watermark acc: {watermark_acc:.2f}%, Test acc: {test_acc_init:.2f}%")

    # 关键修复2：在水印嵌入前就开始记录memory
    if gem:
        # 先用原始任务数据初始化memory
        print("Initializing memory with original task gradients...")
        global_model.train()
        global_model.apply(set_bn_eval)

        # 用少量原始数据计算梯度来初始化memory
        init_memory_loader = DataLoader(memory_dataset, batch_size=32, shuffle=True)
        for batch_idx, (images, labels) in enumerate(init_memory_loader):
            if batch_idx >= 5:  # 只用前5个batch初始化
                break
            images, labels = images.to('cuda'), labels.long().to('cuda')
            optimizer.zero_grad()
            outputs = global_model(images)
            loss = memory_loss_func(outputs, labels)
            loss.backward()

            # 将初始梯度作为memory的基础
            for name, param in global_model.named_parameters():
                if param.grad is not None:
                    if name not in global_model.memory:
                        global_model.memory[name] = param.grad.data.clone()
                    else:
                        global_model.memory[name] = torch.add(global_model.memory[name], param.grad.data)
            optimizer.zero_grad()

    watermark_embed_iters = 0
    while watermark_acc <= 90 and watermark_embed_iters < 60:
        watermark_embed_iters += 1
        global_model.train()
        global_model.apply(set_bn_eval)

        epoch_loss = 0.0
        for batch_idx, (images, labels) in enumerate(watermark_set):
            images, labels = images.to('cuda'), labels.long().to('cuda')
            optimizer.zero_grad()
            outputs = global_model(images)
            loss = watermark_loss_func(outputs, labels)
            loss.backward()

            # 关键修复3：在水印训练过程中应用GEM
            if gem:
                global_model = gem_train(global_model)

            optimizer.step()
            epoch_loss += loss.item()

        watermark_acc, _ = test_img(global_model, trigger_set, Args())
        test_acc, _ = test_img(global_model, test_dataset, Args())
        global_model.to('cuda')

        avg_loss = epoch_loss / len(watermark_set)
        print(
            f"[WM {watermark_embed_iters:03d}] loss: {avg_loss:.4f}, watermark acc: {watermark_acc:.2f}%, test acc: {test_acc:.2f}%")

        # 如果测试精度下降太多，提前结束
        if test_acc < test_acc_init * 0.7:  # 调整阈值
            print(f"Test accuracy dropped too much, stopping watermark embedding")
            break

    watermark_acc_after_embed, _ = test_img(global_model, trigger_set, Args())
    test_acc_after_embed, _ = test_img(global_model, test_dataset, Args())
    global_model.to('cuda')
    print(f"[After Watermark] Watermark acc: {watermark_acc_after_embed:.2f}%, Test acc: {test_acc_after_embed:.2f}%")

    # === 阶段2: 记忆保持训练（精细调整）===
    print("\n=== Phase 2: Memory Preservation (Fine-tuning) ===")

    # 关键修复4：降低学习率进行精细调整
    fine_tune_optimizer = torch.optim.SGD(global_model.parameters(), lr=0.00005, momentum=0.9)

    memory_preservation_epochs = 20
    for epoch in range(memory_preservation_epochs):
        global_model.train()
        global_model.apply(set_bn_eval)

        epoch_loss = 0.0
        num_batches = 0

        # 关键修复5：混合训练 - 同时使用原始数据和水印数据
        combined_batches = create_mixed_loader(memory_loader, watermark_set, ratio=0.8)  # 80%原始数据，20%水印数据

        for batch_idx, (images, labels, data_flags) in enumerate(combined_batches):
            images, labels = images.to('cuda'), labels.long().to('cuda')
            data_flags = data_flags.to('cuda')

            fine_tune_optimizer.zero_grad()
            outputs = global_model(images)

            # 分别计算损失
            watermark_mask = (data_flags == 1)
            memory_mask = (data_flags == 0)

            total_loss = 0
            loss_count = 0

            if watermark_mask.sum() > 0:
                watermark_outputs = outputs[watermark_mask]
                watermark_labels = labels[watermark_mask]
                wm_loss = watermark_loss_func(watermark_outputs, watermark_labels)
                total_loss += wm_loss
                loss_count += 1

            if memory_mask.sum() > 0:
                memory_outputs = outputs[memory_mask]
                memory_labels = labels[memory_mask]
                mem_loss = memory_loss_func(memory_outputs, memory_labels)
                total_loss += mem_loss
                loss_count += 1

            if loss_count > 0:
                avg_loss = total_loss / loss_count
                avg_loss.backward()

                if gem:
                    global_model = gem_train(global_model)

                fine_tune_optimizer.step()
                epoch_loss += avg_loss.item()
                num_batches += 1

        # 每3个epoch评估一次
        if (epoch + 1) % 3 == 0:
            watermark_acc, _ = test_img(global_model, trigger_set, Args())
            test_acc, _ = test_img(global_model, test_dataset, Args())
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            global_model.to('cuda')
            print(
                f"[MP {epoch + 1:03d}] loss: {avg_loss:.4f}, watermark acc: {watermark_acc:.2f}%, test acc: {test_acc:.2f}%")

    # === 最终评估 ===
    print("\n=== Final Evaluation ===")
    trigger_set_final = generate_waffle_pattern(Args())
    watermark_acc, _ = test_img(global_model, trigger_set_final, Args())
    test_acc, _ = test_img(global_model, test_dataset, Args())
    global_model.to('cuda')

    print(f"=== Training Complete ===")
    print(f"Initial Test Acc: {test_acc_init:.2f}%")
    print(f"After Watermark: {test_acc_after_embed:.2f}%")
    print(f"Final Test Acc: {test_acc:.2f}%")
    print(f"Final Watermark Acc: {watermark_acc:.2f}%")

    if test_acc_after_embed < test_acc_init:
        recovery_rate = ((test_acc - test_acc_after_embed) / (test_acc_init - test_acc_after_embed) * 100)
        print(f"Test Acc Recovery: {recovery_rate:.1f}%")

    return global_model


def create_mixed_loader(memory_loader, watermark_loader, ratio=0.8):
    """
    创建混合数据加载器，按比例混合原始数据和水印数据
    """
    import random

    # 收集所有数据
    memory_data = []
    watermark_data = []

    # 从memory_loader中收集数据
    for images, labels in memory_loader:
        memory_data.extend(list(zip(images, labels)))

    # 从watermark_loader中收集数据
    for images, labels in watermark_loader:
        watermark_data.extend(list(zip(images, labels)))

    # 按比例采样
    num_memory = int(len(memory_data) * ratio)
    num_watermark = max(1, len(memory_data) - num_memory)

    # 随机采样
    sampled_memory = random.sample(memory_data, min(num_memory, len(memory_data)))
    sampled_watermark = random.sample(watermark_data, min(num_watermark, len(watermark_data)))

    # 创建批次数据
    batch_size = 32
    batches = []

    # 创建混合批次
    all_memory = [(img, label, 0) for img, label in sampled_memory]  # 0表示原始数据
    all_watermark = [(img, label, 1) for img, label in sampled_watermark]  # 1表示水印数据

    all_data = all_memory + all_watermark
    random.shuffle(all_data)

    # 分批
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i + batch_size]
        if len(batch) > 0:
            images = torch.stack([item[0] for item in batch])
            labels = torch.tensor([item[1] for item in batch])
            flags = torch.tensor([item[2] for item in batch])  # 0=原始数据, 1=水印数据
            batches.append((images, labels, flags))

    return batches


def gem_train_improved(model):
    """
    改进的GEM训练函数，更好地处理梯度投影
    """
    for name, param in model.named_parameters():
        if param.grad is not None and name in model.memory:
            grads = param.grad.data
            memory = model.memory[name]

            # 计算梯度与memory的点积
            dotp = torch.sum(grads * memory)

            # 如果点积为负（即梯度方向与memory相反），进行投影
            if dotp < 0:
                # 投影公式：g' = g - (g·m / ||m||²) * m
                memory_norm_sq = torch.sum(memory * memory)
                if memory_norm_sq > 1e-8:  # 避免除零
                    projection = (dotp / memory_norm_sq) * memory
                    param.grad.data = grads - projection

    return model

def adaptive_memory_preservation(global_model, memory_loader, target_test_acc, max_epochs=50):
    """
    自适应的记忆保持训练：根据测试精度动态调整训练
    """
    memory_loss_func = torch.nn.CrossEntropyLoss()
    memory_optim = torch.optim.SGD(global_model.parameters(), lr=0.0001, momentum=0.9)

    trigger_set = generate_waffle_pattern(Args())
    _, test_dataset = get_full_datasets()

    print("Starting adaptive memory preservation...")

    for epoch in range(max_epochs):
        global_model.to('cuda')
        global_model.train()
        global_model.apply(set_bn_eval)

        epoch_loss = 0.0
        num_batches = 0

        for images, labels in memory_loader:
            images, labels = images.to('cuda'), labels.long().to('cuda')
            memory_optim.zero_grad()
            outputs = global_model(images)
            loss = memory_loss_func(outputs, labels)
            loss.backward()
            global_model = gem_train(global_model)
            memory_optim.step()

            epoch_loss += loss.item()
            num_batches += 1

        # 评估当前性能
        test_acc, _ = test_img(global_model, test_dataset, Args())
        watermark_acc, _ = test_img(global_model, trigger_set, Args())
        global_model.to('cuda')

        print(f"[AMP {epoch + 1:03d}] test acc: {test_acc:.2f}%, watermark acc: {watermark_acc:.2f}%")

        # 如果达到目标精度，停止训练
        if test_acc >= target_test_acc:
            print(f"Target test accuracy {target_test_acc:.2f}% reached!")
            break

        # 如果水印精度下降太多，停止训练
        if watermark_acc < 70:  # 设置水印精度的最低阈值
            print("Watermark accuracy dropped too much, stopping")
            break

    return global_model


def enhanced_gem_train(model, constraint_strength=1.0):
    """
    增强的GEM训练，可以调整约束强度
    constraint_strength: 约束强度，0-1之间，1表示完全约束，0表示无约束
    """
    if not hasattr(model, 'memory') or not model.memory:
        return model

    for name, param in model.named_parameters():
        grads = param.grad
        if grads is not None and name in model.memory:
            memory = model.memory[name]
            dotp = torch.mul(grads.view(-1), memory.view(-1))

            # 只有当点积和小于阈值时才进行投影
            threshold = -0.01 * constraint_strength  # 可调整的阈值

            if dotp.sum() < threshold:
                # 使用加权投影，不是完全投影
                newgrads = project2cone2(grads.view(-1).unsqueeze(1), memory.view(-1, 1))

                # 在原梯度和投影梯度之间插值
                interpolated_grads = constraint_strength * newgrads + (1 - constraint_strength) * grads.view(
                    -1).unsqueeze(1)
                param.grad.data.copy_(interpolated_grads.view(grads.size()))

    return model


# 使用示例
def test_run_complete_training():
    """运行完整的训练流程"""
    model = test_insert_watermark_with_memory_preservation(gem=True)

    # 如果需要进一步的自适应调整
    train_dataset, test_dataset = get_full_datasets()
    memory_indices = torch.randperm(len(train_dataset))[:1000]
    memory_dataset = torch.utils.data.Subset(train_dataset, memory_indices)
    memory_loader = DataLoader(memory_dataset, batch_size=32, shuffle=True)

    # 目标是恢复到初始测试精度的90%
    target_acc = 91.73 * 0.9  # 根据您的初始精度调整
    model = adaptive_memory_preservation(model, memory_loader, target_acc)

    return model


def test_optimized_watermark_training(gem=True):
    """
    优化版本：更平衡的水印嵌入与记忆保持
    """
    # 加载模型
    model_path = "../models/pretrained/resnet20-12fca82f-single-gem.pth"
    global_model = torch.load(model_path, map_location="cuda")
    global_model = global_model.to('cuda')

    # 准备数据
    train_dataset, test_dataset = get_full_datasets()
    trigger_set = generate_waffle_pattern(Args())
    watermark_set = DataLoader(trigger_set, batch_size=16, shuffle=True)

    # 增加记忆保持数据集大小
    memory_preservation_size = 5000  # 增加到5000
    memory_indices = torch.randperm(len(train_dataset))[:memory_preservation_size]
    memory_dataset = torch.utils.data.Subset(train_dataset, memory_indices)
    memory_loader = DataLoader(memory_dataset, batch_size=32, shuffle=True)

    # 使用自适应学习率
    optimizer = torch.optim.SGD(global_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    watermark_loss_func = torch.nn.CrossEntropyLoss()
    memory_loss_func = torch.nn.CrossEntropyLoss()

    print("=== Optimized Training Process ===")
    watermark_acc, _ = test_img(global_model, trigger_set, Args())
    test_acc_init, _ = test_img(global_model, test_dataset, Args())
    global_model.to('cuda')
    print(f"[Init ] Watermark acc: {watermark_acc:.2f}%, Test acc: {test_acc_init:.2f}%")

    # 策略1: 强化memory初始化
    if gem:
        print("Enhanced memory initialization...")
        global_model.train()
        global_model.apply(set_bn_eval)

        for batch_idx, (images, labels) in enumerate(memory_loader):
            if batch_idx >= 20:  # 使用更多batch初始化
                break
            images, labels = images.to('cuda'), labels.long().to('cuda')
            optimizer.zero_grad()
            outputs = global_model(images)
            loss = memory_loss_func(outputs, labels)
            loss.backward()

            for name, param in global_model.named_parameters():
                if param.grad is not None:
                    if name not in global_model.memory:
                        global_model.memory[name] = param.grad.data.clone() * 0.1  # 缩放初始memory
                    else:
                        global_model.memory[name] = torch.add(global_model.memory[name], param.grad.data * 0.1)
            optimizer.zero_grad()

    # 策略2: 渐进式水印嵌入（降低初始冲击）
    print("\n=== Phase 1: Progressive Watermark Embedding ===")
    watermark_weights = [0.1, 0.3, 0.5, 0.7, 1.0]  # 渐进式权重

    total_watermark_iters = 0
    for stage, wm_weight in enumerate(watermark_weights):
        print(f"\nStage {stage + 1}: Watermark weight = {wm_weight}")

        stage_iters = 0
        while stage_iters < 15:  # 每个阶段最多15轮
            stage_iters += 1
            total_watermark_iters += 1

            global_model.train()
            global_model.apply(set_bn_eval)

            # 混合训练：同时进行水印嵌入和记忆保持
            combined_batches = create_balanced_mixed_loader(memory_loader, watermark_set, wm_ratio=0.3)
            epoch_loss = 0.0

            for batch_idx, (images, labels, data_flags) in enumerate(combined_batches):
                images, labels = images.to('cuda'), labels.long().to('cuda')
                optimizer.zero_grad()
                outputs = global_model(images)

                watermark_mask = (data_flags == 1)
                memory_mask = (data_flags == 0)

                total_loss = 0
                if watermark_mask.sum() > 0:
                    wm_loss = watermark_loss_func(outputs[watermark_mask], labels[watermark_mask])
                    total_loss += wm_loss * wm_weight  # 应用渐进式权重

                if memory_mask.sum() > 0:
                    mem_loss = memory_loss_func(outputs[memory_mask], labels[memory_mask])
                    total_loss += mem_loss * (2.0 - wm_weight)  # 平衡权重

                if total_loss > 0:
                    total_loss.backward()

                    if gem:
                        global_model = gem_train_enhanced(global_model)

                    optimizer.step()
                    epoch_loss += total_loss.item()

            # 每5轮评估一次
            if stage_iters % 5 == 0:
                watermark_acc, _ = test_img(global_model, trigger_set, Args())
                test_acc, _ = test_img(global_model, test_dataset, Args())
                current_lr = optimizer.param_groups[0]['lr']
                global_model.to('cuda')
                print(
                    f"[Stage {stage + 1}-{stage_iters:02d}] WM: {watermark_acc:.1f}%, Test: {test_acc:.1f}%, LR: {current_lr:.6f}")

                # 如果当前阶段水印达标，进入下一阶段
                if watermark_acc >= 85:
                    break

        scheduler.step()

    # 策略3: 精细调整阶段
    print("\n=== Phase 2: Fine-tuning for Balance ===")
    fine_optimizer = torch.optim.SGD(global_model.parameters(), lr=0.0001, momentum=0.9)

    best_balance_score = 0
    best_model_state = None

    for epoch in range(30):
        global_model.train()
        global_model.apply(set_bn_eval)

        # 使用更平衡的混合比例
        combined_batches = create_balanced_mixed_loader(memory_loader, watermark_set, wm_ratio=0.2)

        for batch_idx, (images, labels, data_flags) in enumerate(combined_batches):
            images, labels = images.to('cuda'), labels.long().to('cuda')
            fine_optimizer.zero_grad()
            outputs = global_model(images)

            watermark_mask = (data_flags == 1)
            memory_mask = (data_flags == 0)

            total_loss = 0
            if watermark_mask.sum() > 0:
                wm_loss = watermark_loss_func(outputs[watermark_mask], labels[watermark_mask])
                total_loss += wm_loss * 0.3  # 较小的水印权重

            if memory_mask.sum() > 0:
                mem_loss = memory_loss_func(outputs[memory_mask], labels[memory_mask])
                total_loss += mem_loss * 1.0  # 更大的原始任务权重

            if total_loss > 0:
                total_loss.backward()

                if gem:
                    global_model = gem_train_enhanced(global_model)

                fine_optimizer.step()

        # 每5轮评估并保存最佳平衡点
        if (epoch + 1) % 5 == 0:
            watermark_acc, _ = test_img(global_model, trigger_set, Args())
            test_acc, _ = test_img(global_model, test_dataset, Args())
            global_model.to('cuda')

            # 计算平衡得分（水印>90% 且 测试精度尽可能高）
            if watermark_acc >= 90:
                balance_score = test_acc + (watermark_acc - 90) * 0.1
                if balance_score > best_balance_score:
                    best_balance_score = balance_score
                    best_model_state = global_model.state_dict().copy()

            print(
                f"[Fine {epoch + 1:02d}] WM: {watermark_acc:.1f}%, Test: {test_acc:.1f}%, Balance: {balance_score:.1f}")

    # 恢复最佳平衡状态
    if best_model_state is not None:
        global_model.load_state_dict(best_model_state)
        print("Restored to best balance point")

    # 最终评估
    print("\n=== Final Evaluation ===")
    watermark_acc, _ = test_img(global_model, trigger_set, Args())
    test_acc, _ = test_img(global_model, test_dataset, Args())
    global_model.to('cuda')

    print(f"=== Training Complete ===")
    print(f"Initial Test Acc: {test_acc_init:.2f}%")
    print(f"Final Test Acc: {test_acc:.2f}%")
    print(f"Final Watermark Acc: {watermark_acc:.2f}%")

    # 计算性能指标
    test_retention = (test_acc / test_acc_init) * 100
    watermark_success = watermark_acc >= 90

    print(f"Test Acc Retention: {test_retention:.1f}%")
    print(f"Watermark Success: {'✓' if watermark_success else '✗'}")
    print(f"Overall Success: {'✓' if watermark_success and test_retention >= 85 else '✗'}")

    return global_model


def create_balanced_mixed_loader(memory_loader, watermark_loader, wm_ratio=0.2):
    """
    创建更平衡的混合数据加载器
    """
    import random

    # 收集数据
    memory_data = []
    watermark_data = []

    for images, labels in memory_loader:
        memory_data.extend(list(zip(images, labels)))

    for images, labels in watermark_loader:
        watermark_data.extend(list(zip(images, labels)))

    # 确保数据平衡
    target_size = min(len(memory_data), 1000)  # 限制总大小
    num_watermark = int(target_size * wm_ratio)
    num_memory = target_size - num_watermark

    # 采样
    sampled_memory = random.sample(memory_data, min(num_memory, len(memory_data)))
    sampled_watermark = random.sample(watermark_data, min(num_watermark, len(watermark_data)))

    # 创建批次
    all_data = [(img, label, 0) for img, label in sampled_memory] + \
               [(img, label, 1) for img, label in sampled_watermark]

    random.shuffle(all_data)

    batches = []
    batch_size = 32
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i + batch_size]
        if len(batch) > 0:
            images = torch.stack([item[0] for item in batch])
            labels = torch.tensor([item[1] for item in batch])
            flags = torch.tensor([item[2] for item in batch])
            batches.append((images, labels, flags))

    return batches


def gem_train_enhanced(model):
    """
    增强版GEM，更稳定的梯度投影
    """
    for name, param in model.named_parameters():
        if param.grad is not None and name in model.memory:
            grads = param.grad.data
            memory = model.memory[name]

            # 计算梯度与memory的余弦相似度
            grad_norm = torch.norm(grads)
            memory_norm = torch.norm(memory)

            if grad_norm > 1e-8 and memory_norm > 1e-8:
                cosine_sim = torch.sum(grads * memory) / (grad_norm * memory_norm)

                # 如果相似度为负（方向相反），进行投影
                if cosine_sim < -0.1:  # 添加阈值避免过度投影
                    dotp = torch.sum(grads * memory)
                    memory_norm_sq = torch.sum(memory * memory)
                    projection = (dotp / memory_norm_sq) * memory
                    param.grad.data = grads - projection * 0.8  # 部分投影，避免过度约束

    return model


import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import cv2


# 非视觉方案其中有
# 方案1: 放大扰动
# 方案2: 热力图显示扰动强度
# 方案3: 边缘检测突出扰动区域
# 方案4: 叠加显示(原图+扰动热力图)
# 显示概率分布图
def visualize_perturbation_enhanced(x_orig, x_adv, save_path_prefix, amplify_factor=10):
    """
    增强对抗扰动的可视化效果

    Args:
        x_orig: 原始图像 tensor
        x_adv: 对抗样本 tensor
        save_path_prefix: 保存路径前缀
        amplify_factor: 放大倍数
    """

    # 计算扰动
    perturbation = (x_adv - x_orig).detach().cpu()

    # 方案1: 放大扰动并使用热力图
    perturbation_amplified = perturbation * amplify_factor
    perturbation_amplified = torch.clamp(perturbation_amplified, -1, 1)

    # 转换为numpy格式用于可视化
    pert_np = perturbation_amplified[0].permute(1, 2, 0).numpy()

    # 如果是RGB图像，计算L2范数作为扰动强度
    if pert_np.shape[2] == 3:
        pert_magnitude = np.sqrt(np.sum(pert_np ** 2, axis=2))
    else:
        pert_magnitude = np.abs(pert_np.squeeze())

    # 创建多种可视化方案
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原图
    axes[0, 0].imshow(x_orig[0].cpu().permute(1, 2, 0))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    # 对抗样本
    axes[0, 1].imshow(x_adv[0].cpu().permute(1, 2, 0))
    axes[0, 1].set_title("Adversarial Image")
    axes[0, 1].axis('off')

    # 方案1: 热力图显示扰动强度
    im1 = axes[0, 2].imshow(pert_magnitude, cmap='hot', interpolation='nearest')
    axes[0, 2].set_title("Perturbation Heatmap")
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], shrink=0.6)

    # 方案2: 放大后的扰动(带符号)
    pert_signed = (perturbation_amplified[0].permute(1, 2, 0) + 1) / 2  # 归一化到[0,1]
    axes[1, 0].imshow(pert_signed)
    axes[1, 0].set_title(f"Amplified Perturbation (x{amplify_factor})")
    axes[1, 0].axis('off')

    # 方案3: 边缘检测突出扰动区域
    pert_gray = cv2.cvtColor((pert_magnitude * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    edges = cv2.Canny(pert_gray, 50, 150)
    axes[1, 1].imshow(edges, cmap='gray')
    axes[1, 1].set_title("Perturbation Edges")
    axes[1, 1].axis('off')

    # 方案4: 叠加显示(原图+扰动热力图)
    orig_img = x_orig[0].cpu().permute(1, 2, 0).numpy()
    # 创建热力图的彩色版本
    pert_colored = plt.cm.jet(pert_magnitude / pert_magnitude.max())[:, :, :3]
    # 叠加显示
    overlay = orig_img * 0.7 + pert_colored * 0.3
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title("Original + Perturbation Overlay")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_enhanced_perturbation.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 额外保存单独的热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(pert_magnitude, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Perturbation Magnitude')
    plt.title("Perturbation Heatmap")
    plt.axis('off')
    plt.savefig(f"{save_path_prefix}_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def visualize_perturbation_statistical(x_orig, x_adv, save_path_prefix):
    """
    统计分析扰动分布
    """
    perturbation = (x_adv - x_orig).detach().cpu()
    pert_flat = perturbation.flatten().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 扰动分布直方图
    axes[0].hist(pert_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Perturbation Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Perturbation Distribution')
    axes[0].grid(True, alpha=0.3)

    # 扰动的绝对值分布
    axes[1].hist(np.abs(pert_flat), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1].set_xlabel('Absolute Perturbation Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Absolute Perturbation Distribution')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_perturbation_stats.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_perturbation_animation(x_orig, x_adv, save_path_prefix, steps=10):
    """
    创建扰动渐变动画效果的多帧图像
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(steps):
        alpha = i / (steps - 1)
        interpolated = x_orig * (1 - alpha) + x_adv * alpha

        axes[i].imshow(interpolated[0].cpu().permute(1, 2, 0))
        axes[i].set_title(f'Step {i + 1} (α={alpha:.1f})')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_perturbation_animation.png", dpi=300, bbox_inches='tight')
    plt.close()


# 在你的主代码中替换原有的扰动可视化部分
def enhanced_perturbation_visualization(x, x_adv, i, unique_id, y, preds):
    """
    集成的增强扰动可视化函数
    """
    perturbation_base_path = f"./pictures/{unique_id}/perturbation/sample_{i}"

    # 1. 基础增强可视化
    visualize_perturbation_enhanced(x, x_adv, perturbation_base_path, amplify_factor=10)

    # 2. 统计分析
    visualize_perturbation_statistical(x, x_adv, perturbation_base_path)

    # 3. 渐变动画
    create_perturbation_animation(x, x_adv, perturbation_base_path)

    # 4. 保存原始扰动(用于对比)
    perturbation = (x_adv - x).abs().clamp(0, 1)
    status = "success" if preds[0] != y.item() else "failure"
    save_image(perturbation[0].cpu(), f"{perturbation_base_path}_original_{status}.png")

    print(f"Enhanced perturbation visualizations saved for sample {i}")

# 视觉方案
# 简单的白底扰动可视化方案
def simple_white_background_perturbation(x_orig, x_adv, save_path_prefix, amplify_factor=20):
    """
    简单的白底扰动可视化方案

    Args:
        x_orig: 原始图像 tensor
        x_adv: 对抗样本 tensor
        save_path_prefix: 保存路径前缀
        amplify_factor: 放大倍数
    """

    # 计算扰动
    perturbation = (x_adv - x_orig).detach().cpu()

    # 方案1: 白底 + 放大扰动
    # 将扰动从[-1,1]范围映射到[0,1]，0.5为中性灰（无扰动），0为黑（负扰动），1为白（正扰动）
    perturbation_normalized = (perturbation + 1) / 2  # 映射到[0,1]

    # 方案2: 白底 + 彩色扰动
    # 将扰动放大后映射到白底上
    perturbation_amplified = perturbation * amplify_factor
    perturbation_amplified = torch.clamp(perturbation_amplified, -1, 1)

    # 白底处理：将扰动值映射到[0.3, 1]范围，避免纯黑，0.5为中性
    perturbation_white_bg = (perturbation_amplified + 1) / 2  # 先映射到[0,1]
    perturbation_white_bg = perturbation_white_bg * 0.7 + 0.3  # 映射到[0.3, 1]，避免太暗

    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 第一行：原图、对抗样本、差值图（传统黑底）
    axes[0, 0].imshow(x_orig[0].cpu().permute(1, 2, 0))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(x_adv[0].cpu().permute(1, 2, 0))
    axes[0, 1].set_title("Adversarial Image")
    axes[0, 1].axis('off')

    # 传统黑底扰动（用于对比）
    perturbation_traditional = (x_adv - x_orig).abs().clamp(0, 1)
    axes[0, 2].imshow(perturbation_traditional[0].cpu().permute(1, 2, 0))
    axes[0, 2].set_title("Traditional Perturbation (Black Background)")
    axes[0, 2].axis('off')

    # 第二行：三种白底扰动方案

    # 方案1：简单白底扰动
    axes[1, 0].imshow(perturbation_normalized[0].permute(1, 2, 0))
    axes[1, 0].set_title("Perturbation (White Background)")
    axes[1, 0].axis('off')

    # 方案2：放大的白底扰动
    axes[1, 1].imshow(perturbation_white_bg[0].permute(1, 2, 0))
    axes[1, 1].set_title(f"Amplified Perturbation (x{amplify_factor}, White BG)")
    axes[1, 1].axis('off')

    # 方案3：RGB通道分离显示
    # 计算每个像素的扰动强度（L2范数）
    pert_magnitude = torch.sqrt(torch.sum(perturbation[0] ** 2, dim=0))  # 注意维度处理
    # 归一化并映射到白底
    pert_mag_normalized = pert_magnitude / (pert_magnitude.max() + 1e-8)
    pert_mag_white_bg = 1 - pert_mag_normalized * 0.7  # 扰动大的地方更暗，背景保持较亮

    axes[1, 2].imshow(pert_mag_white_bg.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title("Perturbation Magnitude (White Background)")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_white_bg_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 单独保存最佳的白底扰动图
    plt.figure(figsize=(8, 8))
    plt.imshow(perturbation_white_bg[0].permute(1, 2, 0))
    plt.title(f"Enhanced Perturbation Visualization (x{amplify_factor} amplification)")
    plt.axis('off')
    plt.savefig(f"{save_path_prefix}_white_bg_best.png", dpi=300, bbox_inches='tight')
    plt.close()

    return perturbation_white_bg


def rgb_channel_perturbation_analysis(x_orig, x_adv, save_path_prefix):
    """
    分析RGB三个通道的扰动情况
    """
    perturbation = (x_adv - x_orig).detach().cpu()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 原图的RGB通道
    for i, (color, channel_name) in enumerate(zip(['Reds', 'Greens', 'Blues'], ['R', 'G', 'B'])):
        axes[0, i].imshow(x_orig[0, i].cpu().numpy(), cmap=color, vmin=0, vmax=1)
        axes[0, i].set_title(f"Original {channel_name} Channel")
        axes[0, i].axis('off')

    # 扰动的RGB通道（白底）
    for i, channel_name in enumerate(['R', 'G', 'B']):
        # 将扰动映射到白底
        pert_channel = perturbation[0, i]
        pert_channel_white = (pert_channel + 1) / 2  # 映射到[0,1]
        pert_channel_white = pert_channel_white * 0.6 + 0.4  # 映射到[0.4, 1]保持较亮

        axes[1, i].imshow(pert_channel_white.numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f"Perturbation {channel_name} Channel (White BG)")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_rgb_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


# 替换你原来代码中的扰动可视化部分
def replace_original_perturbation_code(x, x_adv, i, unique_id, y, preds):
    """
    替换原来的扰动可视化代码
    """
    perturbation_base_path = f"./pictures/{unique_id}/perturbation/sample_{i}"

    # 创建目录（如果不存在）
    import os
    os.makedirs(f"./pictures/{unique_id}/perturbation", exist_ok=True)

    # 1. 白底扰动可视化
    perturbation_enhanced = simple_white_background_perturbation(x, x_adv, perturbation_base_path, amplify_factor=25)

    # 2. RGB通道分析
    rgb_channel_perturbation_analysis(x, x_adv, perturbation_base_path)

    # 3. 保存原始扰动图（用于对比）
    perturbation_original = (x_adv - x).abs().clamp(0, 1)
    status = "success" if preds[0] != y.item() else "failure"
    save_image(perturbation_original[0].cpu(), f"{perturbation_base_path}_original_{status}.png")

    print(f"White background perturbation visualizations saved for sample {i}")