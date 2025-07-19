import tkinter as tk
import sys
import io
import time
import threading

# 假设这是你的深度学习训练函数
def train_model():
    for epoch in range(10):
        for step in range(100):
            time.sleep(0.01)  # 模拟训练过程
            print(f"Epoch {epoch+1}, Step {step+1}: Loss = {(epoch + step) / 100.0:.4f}")
    print("Training finished!")

def start_training(output_text):
    # 创建一个StringIO对象来捕获stdout
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    try:
        train_model()
    finally:
        # 恢复stdout
        sys.stdout = old_stdout

    # 将捕获到的所有输出一次性显示在文本框中
    output_text.insert(tk.END, redirected_output.getvalue())
    output_text.see(tk.END) # 自动滚动到底部

def update_output(output_text):
    # 创建一个StringIO对象来捕获stdout
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    try:
        # 这里执行你的深度学习训练代码
        # 注意：为了实时更新，你可能需要在你的训练循环中定期调用这个函数
        for epoch in range(5):
            for step in range(50):
                time.sleep(0.05)  # 模拟训练过程
                print(f"Epoch {epoch+1}, Step {step+1}: Loss = {(epoch + step) / 50.0:.4f}")
                # 每次打印后都更新文本框
                output_text.insert(tk.END, redirected_output.getvalue())
                output_text.see(tk.END) # 自动滚动到底部
                redirected_output.seek(0) # 重置StringIO的指针
                redirected_output.truncate(0) # 清空StringIO的内容
                root.update() # 强制更新GUI

        print("Training finished!")
        output_text.insert(tk.END, redirected_output.getvalue())
        output_text.see(tk.END)
    finally:
        # 恢复stdout
        sys.stdout = old_stdout

def start_training_threaded(output_text):
    # 使用线程来运行训练函数，避免GUI阻塞
    thread = threading.Thread(target=update_output, args=(output_text,))
    thread.start()

# 创建主窗口
root = tk.Tk()
root.title("深度学习训练信息")

# 创建文本框
output_text = tk.Text(root, height=20, width=80)
output_text.pack(padx=10, pady=10)

# 创建开始训练按钮
start_button = tk.Button(root, text="开始训练", command=lambda: start_training_threaded(output_text))
start_button.pack(pady=5)

# 启动GUI主循环
root.mainloop()