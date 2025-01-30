# distutils: language = c++
# cython: language_level = 3
# cython: boundscheck = False
# cython: wraparound = False

import numpy as np
cimport numpy as np
import sounddevice as sd
import mlx_whisper
import mlx.core as mx
from cython.parallel cimport prange
from libcpp.vector cimport vector
from libc.string cimport memcpy
import traceback

np.import_array()

# 类型定义
ctypedef np.float32_t DTYPE_t

# 常量
cdef int SAMPLE_RATE = 16000
cdef int CHUNK_DURATION = 3
cdef int MAX_AUDIO_DURATION = 8

cdef class AudioProcessor:
    cdef public np.ndarray current_audio
    cdef public str model_path
    
    def __cinit__(self, str model_path):
        self.current_audio = np.array([], dtype=np.float32)
        self.model_path = model_path

    cpdef np.ndarray[DTYPE_t, ndim=1] record_chunk(self, str device_name):
        cdef np.ndarray[DTYPE_t, ndim=2] recording = sd.rec(
            int(CHUNK_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            device=device_name
        )
        sd.wait()
        return recording.flatten()

    cpdef dict process_audio_segments(self, dict result, np.ndarray[DTYPE_t, ndim=1] current_audio):
        cdef int max_samples = MAX_AUDIO_DURATION * SAMPLE_RATE
        cdef float total_duration, segment_end, last_segment_start
        cdef int samples_to_keep, buffer_samples, last_segment_index
        cdef dict last_segment
        cdef list segments

        try:
            # print(f"@当前音频长度: {current_audio.shape[0]/SAMPLE_RATE:.2f}秒")
            if current_audio.shape[0] > max_samples:
                current_audio = current_audio[-max_samples:]
                # print(f"@音频超过{MAX_AUDIO_DURATION}秒,截取最新部分")
            
            if not result["segments"]:
                # print("No segments found")
                return {"is_complete": True, "audio": current_audio}
            
            segments = result["segments"]
            # print(f"Segments: {segments}")
            if len(segments) == 1:
                # print(f"Single segment: {segments[0]}")
                if segments[0]["text"].startswith("ご視聴ありがとうございました"):
                    return {"is_complete": True, "audio": current_audio}
                    
                total_duration = current_audio.shape[0] / SAMPLE_RATE
                segment_end = segments[0]["end"]
                
                if total_duration - segment_end > 0.3:
                    return {"is_complete": False, "audio": np.array([], dtype=np.float32)}
                return {"is_complete": True, "audio": current_audio}
            
            try:
                last_segment_index = len(segments) - 1
                last_segment = segments[last_segment_index]
                last_segment_start = last_segment["start"]
                segment_end = last_segment["end"]
                total_duration = current_audio.shape[0] / SAMPLE_RATE
                # print(f"Last segment: {last_segment}")
            except:
                print(f"Error in process_audio_segments: {segments}")
                print(f"Last segment: {last_segment}")
                print("Detailed error:")
                print(traceback.format_exc())
                return {"is_complete": True, "audio": current_audio}
            
            if total_duration - segment_end > 0.3:
                return {"is_complete": False, "audio": np.array([], dtype=np.float32)}
            
            buffer_samples = int(0.2 * SAMPLE_RATE)
            samples_to_keep = max(0, int(last_segment_start * SAMPLE_RATE) - buffer_samples)
            return {"is_complete": False, "audio": current_audio[samples_to_keep:]}
        except Exception as e:
            print(f"@Error in process_audio_segments: {str(e)}")
            print("@Detailed error:")
            print(traceback.format_exc())
            return {"is_complete": True, "audio": current_audio}

    cpdef dict transcribe_audio(self, np.ndarray[DTYPE_t, ndim=1] audio):
        return mlx_whisper.transcribe(
            mx.array(audio),
            path_or_hf_repo=self.model_path,
            language="ja",
            task="transcribe"
        )

cpdef void recorder_loop(object queue, str device_name):
    cdef AudioProcessor processor = AudioProcessor("")
    cdef np.ndarray[DTYPE_t, ndim=1] audio_data
    while True:
        try:
            audio_data = processor.record_chunk(device_name)
            queue.put(audio_data)
            # print("录音完成")
        except KeyboardInterrupt:
            break

# 修改transcriber_loop函数
cpdef void transcriber_loop(object queue, str model_path):
    cdef AudioProcessor processor = AudioProcessor(model_path)
    cdef dict result, processed
    cdef np.ndarray[DTYPE_t, ndim=1] audio_data
    cdef str text
    cdef int max_samples = MAX_AUDIO_DURATION * SAMPLE_RATE
    
    while True:
        try:
            audio_data = queue.get()
            # print("开始转录...")
            
            # 检查拼接后的音频长度是否超过8秒
            if (processor.current_audio.shape[0] + audio_data.shape[0]) > max_samples:
                # print(f"音频超过{MAX_AUDIO_DURATION}秒,截取最新部分")
                combined_audio = np.concatenate([processor.current_audio, audio_data])
                processor.current_audio = combined_audio[-max_samples:]
            else:
                processor.current_audio = np.concatenate([processor.current_audio, audio_data])
            # print(f"当前音频长度: {processor.current_audio.shape[0]/SAMPLE_RATE:.2f}秒")
            
            result = processor.transcribe_audio(processor.current_audio)
            # print(f"转录结果: {str(result)}")
            processed = processor.process_audio_segments(result, processor.current_audio)
            # print(f"处理结果: {processed}")
            
            processor.current_audio = processed["audio"]
            
            for segment in result["segments"]:
                text = segment["text"].strip()
                if text and not text.startswith("ご視聴ありがとうございました") and not text == "ん":
                    print(f"转写文本: {text}")
            
        except Exception as e:
            print(f"转录错误: {str(e)}")
            print("详细错误信息:")
            print(traceback.format_exc())
            processor.current_audio = np.array([], dtype=np.float32)

cpdef void preload_model(str model_path):
    print("正在加载模型...")
    cdef np.ndarray[DTYPE_t, ndim=1] empty_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    
    _ = mlx_whisper.transcribe(
        mx.array(empty_audio),
        path_or_hf_repo=model_path,
        language="ja",
        task="transcribe"
    )
    print("模型加载完成")