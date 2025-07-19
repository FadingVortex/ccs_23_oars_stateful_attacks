import logging
import os
import time
from tqdm import tqdm
import torch
import numpy as np
from torchvision import transforms
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from seed import seed_everything
from attacks.Attack import AttackError

from attacks.adaptive.Square import Square
from attacks.adaptive.NESScore import NESScore
from attacks.adaptive.HSJA import HSJA
from attacks.adaptive.QEBA import QEBA
from attacks.adaptive.SurFree import SurFree
from attacks.adaptive.Boundary import Boundary

from torchvision.utils import save_image

from utils.kit import enhanced_perturbation_visualization, replace_original_perturbation_code


@torch.no_grad()
def natural_performance(model, loader):
    logging.info("Computing natural accuracy")
    y_true, y_pred = [], []
    # pbar = tqdm(range(0, len(loader)), colour="red")
    pbar = tqdm(range(0, len(loader)))
    for i, (x, y, p) in (enumerate(loader)):
        x, y = x.cuda(), y.cuda()
        start = time.time()
        logits, is_cache = model(x)
        end = time.time()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()

        logging.info(
            f"True Label : {y[0]} | Predicted Label : {preds[0]} | is_cache : {is_cache[0]} | latency : {end - start}")

        if model.config["action"] == "rejection":
            preds = [preds[j] if not is_cache[j] else -1 for j in range(len(preds))]
        true = y.detach().cpu().numpy().tolist()
        y_true.extend(true)
        y_pred.extend(preds)
        pbar.update(1)
        pbar.set_description(
            "Running accuracy: {} | hits : {}".format(accuracy_score(y_true, y_pred), model.cache_hits))
    logging.info("FINISHED")
    return accuracy_score(y_true, y_pred)


# @torch.no_grad()
# model - StatefulClassifier
def attack_loader(model, loader, model_config, attack_config):
    cumulative_success = []
    success_count = 0
    queries_history = []

    # create time stamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    unique_id = f"run_{timestamp}"

    original_path = f"./pictures/{unique_id}/original"
    adversarial_path = f"./pictures/{unique_id}/adversarial"
    perturbation_path = f"./pictures/{unique_id}/perturbation"

    # create save directory
    os.makedirs(f"./pictures/{unique_id}/original", exist_ok=True);
    os.makedirs(f"./pictures/{unique_id}/adversarial", exist_ok=True);
    os.makedirs(f"./pictures/{unique_id}/perturbation", exist_ok=True);

    # Load attack
    try:
        # 从全局命名空间中获取攻击类的构造函数
        # attacker = NESScore(model, model_config, attack_config)
        attacker = globals()[attack_config['attack']](model, model_config, attack_config)
    except KeyError:
        raise NotImplementedError(f'Attack {attack_config["attack"]} not implemented.')

    # 将目标的标签改成其他指定标签
    if attack_config['targeted']:
        target_labels = []
        for _, (_, y, p) in enumerate(loader):
            target_label = y.item()
            while target_label == y.item():
                target_label = np.random.randint(0, len(loader.dataset.targeted_dict))
            target_labels.append(target_label)
    else:
        target_labels = None

    # Run attack and compute adversarial accuracy
    # x - image  y - label
    y_true, y_pred = [], []
    # Record the time start of the attack
    start_time = time.time()

    pbar = tqdm(loader)
    for i, (x, y, p) in enumerate(pbar):
        x = x.cuda()
        y = y.cuda()

        # save original image
        # original_path = f"{original_path}/original_{i}_true{y.item()}.png"
        save_image(x[0].cpu(), original_path + f"/original_{i}_true{y.item()}.png")

        # 创建包含两个子图的图形
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(x[0].cpu().permute(1, 2, 0))
        axes[0].set_title(f"Original Image - True Label: {y.item()}")

        # 显示原始图像
        # plt.figure(figsize=(5, 5))
        # plt.imshow(x[0].cpu().permute(1, 2, 0))
        # plt.title(f"Original Image - True Label: {y.item()}")


        # plt.show()

        seed_everything()
        try:
            if model.model(x).argmax(dim=1) != y:    # 如果模型已经错误分类，则不攻击
                x_adv = x
            elif attack_config['targeted']:
                y_target = target_labels[i]
                x_adv_init = loader.dataset.initialize_targeted(y_target).cuda()
                y_target = torch.tensor([y_target]).cuda()
                x_adv = attacker.attack_targeted(x, y_target, x_adv_init)
            else:
                x_adv = attacker.attack_untargeted(x, y)
        except AttackError as e:
            print(e)
            x_adv = x

        x_adv = x_adv.cuda()


        # 显示对抗样本
        # plt.figure(figsize=(5, 5))
        # plt.imshow(x_adv[0].cpu().permute(1, 2, 0))
        # plt.title(f"Adversarial Image")
        # plt.show()


        logits = model.model(x_adv) # 对抗样本的模型输出
        # argmax(logits, dim=1) 返回第二维度的最大值的索引
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        true = y.detach().cpu().numpy().tolist()

        # save adversarial image
        # adversarial_path = f"./pictures/{unique_id}/adversarial/adversarial_{i}_true{y.item()}_pred{preds[0]}.png"
        # adversarial_path = f"{adversarial_path}/adversarial_{i}_true{y.item()}_pred{preds[0]}.png"
        save_image(x_adv[0].cpu(), adversarial_path + f"/adversarial_{i}_true{y.item()}_pred{preds[0]}.png")

        axes[1].imshow(x_adv[0].cpu().permute(1, 2, 0))
        axes[1].set_title(f"Adversarial Image - Predicted Label: {preds[0]}")

        # 显示扰动差异
        perturbation = (x_adv - x).detach().cpu()
        # 计算L2范数差异指标
        x_adv_np = x_adv.detach().cpu().numpy()
        x_orig_np = x.detach().cpu().numpy()
        # l2_norm_diff = np.linalg.norm(x_adv_np - x_orig_np) / (
        #             x_orig_np.shape[-1] * x_orig_np.shape[-2] * x_orig_np.shape[-3])
        l2_norm_diff = np.linalg.norm(x_adv_np - x_orig_np)

        perturbation_nomalized = (perturbation + 1) / 2  # 归一化到 [0, 1] 范围
        status = "success" if preds[0] != true[0] else "failure"
        save_image(perturbation_nomalized[0].cpu(), perturbation_path + f"/perturbation_{i}_{status}_L2_{l2_norm_diff:.4f}.png")
        # 非视觉方案
        # enhanced_perturbation_visualization(x, x_adv, i, unique_id, y, preds)
        # 视觉方案
        # replace_original_perturbation_code(x, x_adv, i, unique_id, y, preds)


        # save two figures additionally
        # save_path = f"./pictures/{unique_id}/adversarial/showDifferences.png"
        # plt.savefig(save_path, bbox_inches='tight', dpi=300)
        # plt.show()
        plt.close()
        y_true.extend(true)
        y_pred.extend(preds)

        if true[0] != preds[0] and attacker.get_total_queries() > 0:
            success_count += 1
        current_rate = success_count / (i + 1)
        cumulative_success.append(current_rate)
        queries_history.append(attacker.get_total_queries())

        pbar.set_description("Running Accuracy: {} ".format(accuracy_score(y_true, y_pred)))
        logging.info(
            f"True Label : {true[0]} | Predicted Label : {preds[0]} | Cache Hits / Total Queries : {attacker.get_cache_hits()} / {attacker.get_total_queries()}")
        attacker.reset()

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_success) + 1),
                 np.array(cumulative_success) * 100,
                 'b-', label='Attack Success Rate')
        plt.scatter(range(1, len(cumulative_success) + 1),
                          np.array(cumulative_success) * 100,
                          c=queries_history, cmap='viridis', alpha=0.6)
        plt.colorbar(label='Query Count')
        plt.xlabel("Attacked Samples")
        plt.ylabel("Success Rate (%)")
        plt.title(f"Cumulative Attack Success Rate (Final: {cumulative_success[-1] * 100:.1f}%)")
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 100)
        plt.savefig(f"./pictures/{unique_id}/attack_success_rate.png")
        plt.close()


    # Record the time end of the attack
    end_time = time.time()
    # Calculate the total time taken for the attack
    total_time = end_time - start_time
    # Print the total time taken for the attack
    logging.info(f"Total time taken for the attack: {total_time:.2f} seconds")
    logging.info("FINISHED")


def attack_loader_defense(model, loader, model_config, attack_config):
    cumulative_success = []
    success_count = 0
    queries_history = []

    # create time stamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    unique_id = f"run_{timestamp}"

    original_path = f"./pictures/{unique_id}/original"
    adversarial_path = f"./pictures/{unique_id}/adversarial"
    perturbation_path = f"./pictures/{unique_id}/perturbation"

    # create save directory
    os.makedirs(f"./pictures/{unique_id}/original", exist_ok=True);
    os.makedirs(f"./pictures/{unique_id}/adversarial", exist_ok=True);
    os.makedirs(f"./pictures/{unique_id}/perturbation", exist_ok=True);

    # Load attack
    try:
        # 从全局命名空间中获取攻击类的构造函数
        # attacker = NESScore(model, model_config, attack_config)
        attacker = globals()[attack_config['attack']](model, model_config, attack_config)
    except KeyError:
        raise NotImplementedError(f'Attack {attack_config["attack"]} not implemented.')

    # 将目标的标签改成其他指定标签
    if attack_config['targeted']:
        target_labels = []
        for _, (_, y, p) in enumerate(loader):
            target_label = y.item()
            while target_label == y.item():
                target_label = np.random.randint(0, len(loader.dataset.targeted_dict))
            target_labels.append(target_label)
    else:
        target_labels = None

    # Run attack and compute adversarial accuracy
    # x - image  y - label
    y_true, y_pred = [], []
    # Record the time start of the attack
    start_time = time.time()

    pbar = tqdm(loader)
    for i, (x, y, p) in enumerate(pbar):
        x = x.cuda()
        y = y.cuda()

        # save original image
        # original_path = f"{original_path}/original_{i}_true{y.item()}.png"
        save_image(x[0].cpu(), original_path + f"/original_{i}_true{y.item()}.png")

        # 创建包含两个子图的图形
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(x[0].cpu().permute(1, 2, 0))
        axes[0].set_title(f"Original Image - True Label: {y.item()}")

        seed_everything()
        try:
            if model.model(x).argmax(dim=1) != y:    # 如果模型已经错误分类，则不攻击
                x_adv = x
            elif attack_config['targeted']:
                y_target = target_labels[i]
                x_adv_init = loader.dataset.initialize_targeted(y_target).cuda()
                y_target = torch.tensor([y_target]).cuda()
                x_adv = attacker.attack_targeted(x, y_target, x_adv_init)
            else:
                x_adv = attacker.attack_untargeted(x, y)
        except AttackError as e:
            print(e)
            x_adv = x

        x_adv = x_adv.cuda()


        logits = model.model(x_adv) # 对抗样本的模型输出
        # argmax(logits, dim=1) 返回第二维度的最大值的索引
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        true = y.detach().cpu().numpy().tolist()

        # save adversarial image
        # adversarial_path = f"./pictures/{unique_id}/adversarial/adversarial_{i}_true{y.item()}_pred{preds[0]}.png"
        # adversarial_path = f"{adversarial_path}/adversarial_{i}_true{y.item()}_pred{preds[0]}.png"
        save_image(x_adv[0].cpu(), adversarial_path + f"/adversarial_{i}_true{y.item()}_pred{preds[0]}.png")

        axes[1].imshow(x_adv[0].cpu().permute(1, 2, 0))
        axes[1].set_title(f"Adversarial Image - Predicted Label: {preds[0]}")

        # 显示扰动差异
        perturbation = (x_adv - x).detach().cpu()

        # 计算L2范数差异指标
        x_adv_np = x_adv.detach().cpu().numpy()
        x_orig_np = x.detach().cpu().numpy()
        # l2_norm_diff = np.linalg.norm(x_adv_np - x_orig_np) / (
        #             x_orig_np.shape[-1] * x_orig_np.shape[-2] * x_orig_np.shape[-3])
        l2_norm_diff = np.linalg.norm(x_adv_np - x_orig_np)
        perturbation_nomalized = (perturbation + 1) / 2  # 归一化到 [0, 1] 范围

        status = "failure" if preds[0] != true[0] else "success"
        save_image(perturbation_nomalized[0].cpu(), perturbation_path + f"/perturbation_{i}_{status}_L2_{l2_norm_diff:.4f}.png")

        plt.close()
        y_true.extend(true)
        y_pred.extend(preds)

        pbar.set_description("Running Accuracy: {} ".format(accuracy_score(y_true, y_pred)))
        logging.info(
            f"True Label : {true[0]} | Predicted Label : {preds[0]} | Cache Hits / Total Queries : {attacker.get_cache_hits()} / {attacker.get_total_queries()}")

        if true[0] == preds[0]:
            success_count += 1
        current_rate = success_count / (i + 1)
        cumulative_success.append(current_rate)

        plt.figure(figsize=(16, 4))
        plt.plot(range(1, len(cumulative_success) + 1), np.array(cumulative_success) * 100,
                 label='Defense Success Rate')
        plt.xlabel('Number of Attacked Images')
        plt.ylabel('Cumulative Success Rate (%)')
        plt.title("Cumulative Defense Success Rate vs. Number of Attacked Images")
        plt.grid(True)
        plt.ylim(0, 102)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./pictures/{unique_id}/defense_success_rate.png")
        plt.close()


    # Record the time end of the attack
    end_time = time.time()
    # Calculate the total time taken for the attack
    total_time = end_time - start_time
    # Print the total time taken for the attack
    logging.info(f"Total time taken for the attack: {total_time:.2f} seconds")
    logging.info("FINISHED")
