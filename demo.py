import os
import sys
import glob
import json
import random

# from torch.utils.tensorboard import SummaryWriter

custom_path = r"D:\WorkSpace\progamme\Codefield\Code_Py\ccs_23_oars_stateful_attacks\attacks"
if custom_path not in sys.path:
    sys.path.append(custom_path)
print(sys.path)


from torchvision import transforms

from DemoConfig import DemoConfig
from attacks.attacks import *
from models.statefuldefense import init_stateful_classifier
from utils import datasets
# from torch.utils.tensorboard import SummaryWriter


def attack_function(attack_type, defense_type, loader):
    # # Load attack
    # try:
    #     # 从全局命名空间中获取攻击类的构造函数
    #     # attacker = NESScore(model, model_config, attack_config)
    #     attacker = globals()[attack_type](model, model_config, attack_config)
    # except KeyError:
    #     raise NotImplementedError(f'Attack {attack_type} not implemented.')
    #
    #
    pass


def attack_demo(attack_type, defense_type):
    """
    攻防实验执行函数（需由用户实现）
    :param attack_type: 选择的攻击方式（FGSM, NES 等）
    :param defense_type: 选择的防御策略（Stateful, None 等）
    """
    print(f"执行攻击: {attack_type}，防御: {defense_type}")
    # TODO: 这里调用你的攻击和防御函数，例如 attack_function(attack_type, defense_type)

    random.seed(round(time.time() * 1000))
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(log_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    logging.basicConfig(
        filename=os.path.join(writer.log_dir, f'log_{DemoConfig.start_idx}_{DemoConfig.start_idx + DemoConfig.num_images}.txt'),
        level=logging.INFO)
    logging.info(DemoConfig)
    # log_dir = os.paht.join

    # Load Dataset Cifar10
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.StatefulDefenseDataset(name=DemoConfig.datasets, transform=transform,
                                                   size=DemoConfig.num_images, start_idx=DemoConfig.start_idx)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    # Load config
    # match ./configs/cifar/ /first files/adaptive/config.json
    # or ./configs/cifar/ /first files/config.json

    matching_files = (glob.glob(
                            './configs/{}/{}/{}/*/*/config.json'
                            .format(DemoConfig.datasetpath, defense_type, attack_type)
                        ) + glob.glob(
                            './configs/{}/{}/{}/*/config.json'
                            .format(DemoConfig.datasetpath, defense_type, attack_type)
                        ))
    matching_files = list(set(matching_files)) # remove the same items
    config = json.load(open(matching_files[0]))
    model_config, attack_config = config["model_config"], config["attack_config"]

    # Load model
    model = init_stateful_classifier(model_config)
    model.eval()
    model.to("cuda")

    # attack
    attack_loader(model, test_loader, model_config, attack_config)


def command_interface():
    """
    交互式命令行界面，用户可输入指令执行不同的攻防实验
    """
    print("\n🔥 欢迎使用攻防实验交互式界面 🔥")
    print("输入 'help' 查看可用命令，输入 'exit' 退出")

    while True:
        cmd = input("\n请输入指令: ").strip().lower()

        # 退出程序
        if cmd == "exit":
            print("👋 退出程序，再见！")
            break

        # 显示帮助信息
        elif cmd == "help":
            print("\n可用指令示例:")
            print("  attack Boundary defense Blacklight ➝ 执行 Boundary 攻击 + Blacklight 防御")
            # print("  attack HSJA defense Osd            ➝ 执行 HSJA 攻击 + Osd 防御")
            # print("  attack NES defense PIHA            ➝ 执行 NES 攻击 + PIHA 防御")
            # print("  attack NES defense None            ➝ 执行 NES 攻击，无防御")
            print("  help                               ➝ 显示帮助信息")
            print("  exit                               ➝ 退出程序")

        # 解析 "attack" 指令
        elif cmd.startswith("attack"):
            parts = cmd.split()
            if len(parts) != 4 or parts[2] != "defense":
                print("⚠️ 指令格式错误！输入 'help' 查看正确格式。")
                continue

            attack_type = parts[1].lower()
            defense_type = parts[3].lower()

            if attack_type not in ["boundary", "hsja", "nes"]:
                print("⚠️ 无效的攻击方式！支持: Boundary, NES, HSJA, Osd")
                continue

            if defense_type not in ["blacklight", "osd", "piha", "None"]:
                print("⚠️ 无效的防御策略！支持: blacklight, osd, piha, None")
                continue

            # 执行攻防实验
            attack_demo(attack_type, defense_type)

        else:
            print("⚠️ 无效指令！输入 'help' 查看可用命令。")



def test_create_undefended_config(original_path):
    if not os.path.exists(original_path):
        print(f"❌ Original config not found: {original_path}")
        return

    dir_path = os.path.dirname(original_path)
    base_dir = os.path.dirname(dir_path)
    new_dir = os.path.join(base_dir, "undefended")

    # 创建新目录（包括中间目录）
    os.makedirs(new_dir, exist_ok=True)

    new_path = os.path.join(new_dir, "config.json")

    try:
        with open(original_path, 'r') as f:
            config = json.load(f)

        model_config = config["model_config"]
        model_config["threshold"] = None
        model_config["add_cache_hit"] = False
        model_config["reset_cache_on_hit"] = False
        model_config["aggregation"] = None
        model_config["action"] = None

        model_config["state"] = {
            "type": "no_op",
            "input_shape": [3, 32, 32]
        }

        with open(new_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ Created undefended config at: {new_path}")

    except Exception as e:
        print(f"❌ Error processing {original_path}: {e}")


def test_main():
    config_paths = [
        "configs/cifar/blacklight/surfree/untargeted/standard/config.json",
        "configs/cifar/piha/surfree/untargeted/standard/config.json",
        "configs/cifar/piha/square/untargeted/standard/config.json",
        "configs/cifar/blacklight/square/untargeted/standard/config.json",
        "configs/cifar/blacklight/nesscore/targeted/standard/config.json",
        "configs/cifar/piha/nesscore/targeted/standard/config.json"
    ]

    for path in config_paths:
        test_create_undefended_config(path)


if __name__ == "__main__":
    command_interface()
