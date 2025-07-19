import os
import platform
import sys
import argparse
import parse
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from icecream import ic
import warnings
import matplotlib
import matplotlib.pyplot as plt

# 忽略所有警告信息，防止在日志处理时被无关紧要的警告干扰
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default=None)  # 添加日志路径参数
    parser.add_argument('--defense_log_path', type=str, default=None)  # 添加具有防御的日志路径参数
    parser.add_argument('--defense', action='store_true') # 是否在进行防御分析

    args = parser.parse_args()

    # 从指定路径中获取所有包含"log"关键字的文件并排序
    logs = sorted([x for x in os.listdir(args.log_path) if 'log' in x])
    if not args.defense_log_path == None:
        # 如果指定了防御日志路径，则获取该路径下的所有日志文件
        defense_logs = sorted([x for x in os.listdir(args.defense_log_path) if 'log' in x])
        defense_aggregate_log = []

    aggregate_log = []
    for log in logs:
        path = os.path.join(args.log_path, log)  # 获取日志文件完整路径
        log_lines = open(path).readlines()       # 读取所有行
        log_lines = [x for x in log_lines if 'FINISHED' not in x]  # 去除包含 "FINISHED" 的行
        log_lines = [x for x in log_lines if 'INFO:root:True Label' in x]  # 保留包含标签信息的行
        aggregate_log.extend(log_lines)  # 将处理后的日志添加到聚合日志中

    if not args.defense_log_path == None:
        for log in defense_logs:
            # 如果指定了防御日志路径，则处理防御日志
            defense_path = os.path.join(args.defense_log_path, log)
            defense_log_lines = open(defense_path).readlines()
            defense_log_lines = [x for x in defense_log_lines if 'FINISHED' not in x]
            defense_log_lines = [x for x in defense_log_lines if 'INFO:root:True Label' in x]
            defense_aggregate_log.extend(defense_log_lines)


    print("Length of log file: {}".format(len(aggregate_log)))  # 打印处理后日志行数

    # 初始化各种统计数据的列表
    y_true, y_pred = [], []
    queries = []
    cache_hits = []

    # 逐行处理日志
    for idx, line in enumerate(aggregate_log):
        # 定义日志格式并解析每一行数据
        format_str = "INFO:root:True Label : {} | Predicted Label : {} | Cache Hits / Total Queries : {} / {}"
        parsed = parse.parse(format_str, line)

        # 添加到对应的列表中
        y_true.append(int(parsed[0]))
        y_pred.append(int(parsed[1]))
        cache_hits.append(int(parsed[2]))
        queries.append(int(parsed[3]))

    # 如果指定了防御日志路径，则处理防御日志
    if not args.defense_log_path == None:
        defense_y_true, defense_y_pred = [], []
        for idx, line in enumerate(defense_aggregate_log):
            # 定义防御日志格式并解析每一行数据
            format_str = "INFO:root:True Label : {} | Predicted Label : {} | Cache Hits / Total Queries : {} / {}"
            parsed = parse.parse(format_str, line)

            # 添加到对应的列表中
            defense_y_true.append(int(parsed[0]))
            defense_y_pred.append(int(parsed[1]))

    # 计算使用了非零查询的样本占总数的比例（非零查询即模型真正调用了目标系统）
    accuracy = len([i for i in range(len(y_true)) if queries[i] != 0]) * 100 / len(y_true)

    # 计算整体的准确率（预测正确的占比）
    adv_accuracy = accuracy_score(y_true, y_pred) * 100

    # 计算宏平均 F1 分数（对每一类的 F1 取平均）
    adv_macro_f1 = f1_score(y_true, y_pred, average='macro') * 100

    # 计算攻击成功率（即预测错误的样本在非零查询中的占比）
    attack_success = len([
        i for i in range(len(y_true))
        if y_true[i] != y_pred[i] and queries[i] > 0
    ]) * 100 / len([i for i in range(len(y_true)) if queries[i] > 0])

    # 统计攻击成功样本的平均查询次数和标准差
    avg_queries = np.mean([
        queries[i] for i in range(len(queries))
        if y_true[i] != y_pred[i] and queries[i] > 0
    ])
    std_queries = np.std([
        queries[i] for i in range(len(queries))
        if y_true[i] != y_pred[i] and queries[i] > 0
    ])

    # 统计攻击成功样本的平均缓存命中数及标准差
    avg_cache_hits = np.mean([
        cache_hits[i] for i in range(len(cache_hits))
        if y_true[i] != y_pred[i] and queries[i] > 0
    ])
    std_cache_hits = np.std([
        cache_hits[i] for i in range(len(cache_hits))
        if y_true[i] != y_pred[i] and queries[i] > 0
    ])

    # 如果考虑封号机制：统计攻击成功样本的总“查询数 + 缓存命中数”的平均值及标准差
    avg_queries_if_account_bans = np.mean([
        queries[i] + cache_hits[i] for i in range(len(cache_hits))
        if y_true[i] != y_pred[i] and queries[i] > 0
    ])
    std_queries_if_account_bans = np.std([
        queries[i] + cache_hits[i] for i in range(len(cache_hits))
        if y_true[i] != y_pred[i] and queries[i] > 0
    ])

    # 预留字典，用于根据封号预算评估攻击成功率（尚未使用）
    attack_success_per_ban_budget = {}

    # 使用 icecream 打印所有关键统计数据
    ic(
        accuracy, adv_accuracy, adv_macro_f1, attack_success,
        avg_queries, std_queries,
        avg_cache_hits, std_cache_hits,
        avg_queries_if_account_bans
    )

    # 如果在Linux环境
    if platform.system() == 'Linux' and os.environ.get('DISPLAY', '') == '':
        # 设置 matplotlib 使用 Agg 后端
        matplotlib.use('Agg')


    # 记录累计攻击成功率
    cumulative_success = []
    success_count = 0
    if args.defense:
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i] and queries[i] > 0:
                success_count += 1
            cumulative_success.append(success_count / (i + 1))
    else:
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i] and queries[i] > 0:
                success_count += 1
            cumulative_success.append(success_count / (i + 1))

    # 记录加入防御之后的累计攻击成功率
    if not args.defense_log_path == None:
        defense_cumulative_success = []
        defense_count = 0
        for i in range(len(defense_y_true)):
            if defense_y_true[i] != defense_y_pred[i]:
                defense_count += 1
            defense_cumulative_success.append(defense_count / (i + 1))

    # 绘制累计攻击成功率曲线
    plt.figure(figsize=(16, 4))
    plt.plot(range(1, len(cumulative_success) + 1), np.array(cumulative_success) * 100, label='Cumulative Attack Success Rate')
    if not args.defense_log_path == None:
        plt.plot(range(1, len(defense_cumulative_success) + 1), np.array(defense_cumulative_success) * 100, \
                 label='Cumulative Attack Success Rate With Defense', color='red', linewidth=2)
        plt.text(len(defense_cumulative_success) + 1, defense_cumulative_success[-1] * 100, \
                 f"{defense_cumulative_success[-1] * 100:.1f}%", weight="bold", color='red')

    plt.xlabel("Number of Attacked Images")
    plt.ylabel("Cumulative Success Rate (%)")
    plt.title(f"Cumulative Attack Success Rate (Final: {cumulative_success[-1] * 100:.1f}%)")
    plt.text(len(cumulative_success) + 1, cumulative_success[-1] * 100, f"{cumulative_success[-1] * 100:.1f}%", weight="bold")
    plt.grid(True)
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()

    if platform.system() == 'Linux' and os.environ.get('DISPLAY', '') == '':
        # 保存图像到文件
        path = os.path.join(args.log_path, "attack_success_rate.png")
        plt.savefig(path)
        print("Plot saved as attack_success_rate.png")
    else:
        plt.show()

    if args.defense:
        # 如果启用了防御分析，打印针对每一张图片相关的防御信息
        path = os.path.join(args.log_path, "results.txt")
        with open(path, 'w') as f:
            for i in range(len(cumulative_success)):
                f.write(f"Image {i + 1}:totalQueries={queries[i]}, rejectedQueries={cache_hits[i]}, cumulativeSuccessRate={cumulative_success[i] * 100:.2f}%\n")



