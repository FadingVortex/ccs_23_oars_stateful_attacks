from flask import Flask, render_template, Response, stream_with_context, jsonify
import time
import io
import sys
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

training_output = io.StringIO()
training_running = False
output_lock = threading.Lock()  # 新增全局锁

def train_model():
    global training_output, training_running
    training_running = True
    old_stdout = sys.stdout
    with output_lock:  # 加锁重定向 stdout
        sys.stdout = training_output
    try:
        for epoch in range(3):
            logger.info(f"Starting epoch {epoch+1}")
            for step in range(5):
                time.sleep(0.5)
                message = f"Epoch {epoch+1}, Step {step+1}: Loss = {(epoch + step) / 10.0:.4f}"
                with output_lock:  # 加锁执行 print
                    print(message)
                logger.info(f"Training output: {message}")
        with output_lock:  # 加锁执行 print
            print("Training finished!")
        logger.info("Training finished!")
    finally:
        with output_lock:  # 加锁恢复 stdout
            sys.stdout = old_stdout
        training_running = False

def generate():
    global training_output, training_running
    while training_running or training_output.getvalue():
        time.sleep(0.1)
        with output_lock:  # 加锁读取并清空输出
            output = training_output.getvalue()
            training_output.seek(0)
            training_output.truncate(0)
        if output:
            lines = output.splitlines()
            for line in lines:
                logger.info(f"Yielding: {line}")
                yield f"data: {line}\n\n"
        elif not training_running:
            logger.info("Yielding training end marker")
            yield "data: [TRAINING_END]\n\n"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_training')
def start_training():
    global training_running
    if not training_running:
        threading.Thread(target=train_model).start()
        return jsonify({'message': 'Training started'})
    else:
        return jsonify({'message': 'Training already in progress'})

@app.route('/training_stream')
def training_stream():
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)