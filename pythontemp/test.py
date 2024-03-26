# pip install peft accelerate bitsandbytes
from transformers import pipeline
import torch
MODEL_ID = "vinai/PhoWhisper-large"

pipe = pipeline(task="automatic-speech-recognition", model=MODEL_ID)
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_ID,
    chunk_length_s=30, 
    stride_length_s=(5,5), 
    return_timestamps=True,
)


res = pipe('converted_file.wav', )
print(res)


