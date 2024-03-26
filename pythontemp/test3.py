import torch
from faster_whisper import WhisperModel
from datasets import load_dataset

# define our torch configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if torch.cuda.is_available() else "float32"

# load model on GPU if available, else cpu
model = WhisperModel("distil-whisper/distil-large-v3-ct2", device=device, compute_type=compute_type)

# load toy dataset for example
sample = "1.mp3"

segments, info = model.transcribe(sample, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500), word_timestamps=True, language="vi")

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
