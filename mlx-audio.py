import multiprocessing as mp
from audio_core import recorder_loop, transcriber_loop, preload_model

# 配置
DEVICE_NAME = 'BlackHole 2ch'
MODEL_PATH = './mlx_models/whisper-large-v3-turbo-q4'

def main():
    preload_model(MODEL_PATH)
    
    # 创建队列
    audio_queue = mp.Queue()
    
    # 创建进程
    recorder = mp.Process(target=recorder_loop, args=(audio_queue, DEVICE_NAME))
    transcriber = mp.Process(target=transcriber_loop, args=(audio_queue, MODEL_PATH))
    
    # 启动进程
    print("开始录音，按Ctrl+C停止...")
    transcriber.start()
    recorder.start()
    
    # 等待进程结束
    recorder.join()
    transcriber.terminate()
    transcriber.join()

if __name__ == "__main__":
    main()