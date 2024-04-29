import gc

import os

import gradio as gr
import torch
from transformers import  AutoProcessor, pipeline , SeamlessM4Tv2Model
from pyannote.audio import Pipeline

from util import (
    ASR_TARGET_LANGUAGE_NAMES,
    LANGUAGE_NAME_TO_CODE,
    S2ST_TARGET_LANGUAGE_NAMES,
    S2TT_TARGET_LANGUAGE_NAMES,
    T2ST_TARGET_LANGUAGE_NAMES,
    T2TT_TARGET_LANGUAGE_NAMES,
    TEXT_SOURCE_LANGUAGE_NAMES,
    crop_audio,
    decode_audio,
)

AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 6000000
download_root = "/home/gitlab/asr/models/"
DEFAULT_TARGET_LANGUAGE = "English"
if torch.cuda.is_available():
    cuda = torch.device("cuda:0")
    compute_type=torch.float16

    devicewhisper = "cuda"
else:
    cuda = torch.device("cpu")
    compute_type=torch.float32
    devicewhisper = "cpu"
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token="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" , cache_dir = download_root)
pipeline.to(cuda)

model_id = "facebook/seamless-m4t-v2-large"
model = SeamlessM4Tv2Model.from_pretrained(model_id, torch_dtype=compute_type, low_cpu_mem_usage=True, use_safetensors=True , cache_dir=download_root)
model.to(cuda)
processor = AutoProcessor.from_pretrained(model_id , cache_dir=download_root)

    
def run_asr(input_audio: str, target_language: str) -> str:

    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
    new_arr = decode_audio(input_audio)
    # input_data = processor(audios = new_arr, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt").to(cuda)
    # output_tokens = model.generate(**input_data, tgt_lang=target_language_code, generate_speech=False)
    # finalText = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    diarization = pipeline(input_audio , num_speakers=2 )
    dictlist = []
    tmp_speaker = None
    tmp_start = None
    tmp_end = None
    index = 1
    if new_arr is None:
        gr.Warning("No audio detected! Please upload an audio file with speech before submitting your request.")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if ( index == 1) :
            tmp_speaker = speaker
        tmp_start = turn.start
        tmp_end = turn.end
        if turn.start is not None and turn.end is not None:
            if tmp_speaker == speaker and index != 1:
                tmp_start += turn.start
                tmp_end += turn.end
                continue
            else:
                index = 0
                audio_tensor = crop_audio(new_arr, AUDIO_SAMPLE_RATE, tmp_start, tmp_end)
                if audio_tensor is None:
                    continue
                else :
                    input_data = processor(audios = audio_tensor, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt").to(cuda)
                    output_tokens = model.generate(**input_data, tgt_lang=target_language_code, generate_speech=False)
                    translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
                
                    dict = {
                        "start":turn.start,
                        "end":turn.end,
                        "speaker":tmp_speaker,
                        "text":translated_text_from_audio
                        }
                
                    dictlist.append(dict)
            
            
    
    gc.collect()
    torch.cuda.empty_cache()
    Text = ''
    
    for sement in dictlist :
        
        Text += f"start={sement['start']:.1f}s stop={sement['end']:.1f}s {sement['speaker']} {sement['text']}"  + "\n" 

    return Text 
    

    



with gr.Blocks() as demo_asr:
    with gr.Row():
        with gr.Column():
            with gr.Group():
                input_audio = gr.Audio(label="Input speech", type="filepath")
                target_language = gr.Dropdown(
                    label="Target language",
                    choices=ASR_TARGET_LANGUAGE_NAMES,
                    value=DEFAULT_TARGET_LANGUAGE,
                )
            btn = gr.Button("Translate")
        with gr.Column():
            output_text = gr.Textbox(label="Translated text")

    gr.Examples(
        examples=[],
        inputs=[input_audio, target_language],
        outputs=output_text,
        fn=run_asr,
        # cache_examples=CACHE_EXAMPLES,
        api_name=False,
    )

    btn.click(
        fn=run_asr,
        inputs=[input_audio, target_language],
        outputs=output_text,
        api_name="asr",
    )
with gr.Blocks(css="style.css") as demo:
    # gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )

    with gr.Tabs():
        with gr.Tab(label="ASR"):
            demo_asr.render()

if __name__ == "__main__":
    demo.queue(max_size=50).launch()
