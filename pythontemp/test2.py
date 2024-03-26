# pip install peft accelerate bitsandbytes
import torch
import torchaudio
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer

PEFT_MODEL_ID = "doof-ferb/whisper-large-peft-lora-vi"
BASE_MODEL_ID = PeftConfig.from_pretrained(PEFT_MODEL_ID).base_model_name_or_path

FEATURE_EXTRACTOR = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_ID)
TOKENIZER = WhisperTokenizer.from_pretrained(BASE_MODEL_ID)

MODEL = PeftModel.from_pretrained(
    WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16).to("cuda:0"),
    PEFT_MODEL_ID
).merge_and_unload(progressbar=True)

DECODER_ID = torch.tensor(
    TOKENIZER.convert_tokens_to_ids(["<|startoftranscript|>", "<|vi|>", "<|transcribe|>", "<|notimestamps|>"]),
    device=MODEL.device
).unsqueeze(dim=0)

waveform, sampling_rate = torchaudio.load("1.mp3")
if waveform.size(0) > 1:  # convert dual to mono channel
    waveform = waveform.mean(dim=0, keepdim=True)

inputs = FEATURE_EXTRACTOR(waveform, sampling_rate=sampling_rate, return_tensors="pt").to(MODEL.device)
with torch.inference_mode(), torch.autocast(device_type="cuda"):  # required by PEFT
    predicted_ids = MODEL.generate(input_features=inputs.input_features, decoder_input_ids=DECODER_ID)

TOKENIZER.batch_decode(predicted_ids, skip_special_tokens=True)[0]
