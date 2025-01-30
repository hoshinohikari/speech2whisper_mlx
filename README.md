# speech2whisper_mlx

****

| Author | HoshinoKun |
| ------ | ----------- |
| E-mail | hoshinokun@346pro.club |

****

<!-- [中文简介](/readme_cn.md) -->
## What's this?
Real time capture of system sound and input of whisper-mlx transcription  
**MacOS only**

### How to use
1. Install Blackhole 2ch with brew  
    ```brew install blackhole-2ch```
2. Set the sound output of the system to blackhole, or create a multi output audio device group
3. Set the path of the model or the model name of huggingface in mlx-audito.py
4. Compile code and run it  
    ```python setup.py build_ext --inplace```
5. Run
    ```python mlx-audio.py```

## Known issues
1. Sometimes it gets stuck due to model issues
2. Sometimes there is a phenomenon of sentence repetition, which is due to repeated audio buffers