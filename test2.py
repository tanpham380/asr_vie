import gc

import os
import gc
import gradio as gr
import torch
# from diarize import DiarizationPipeline, assign_word_speakers
from util import (
    ASR_TARGET_LANGUAGE_NAMES,


    format_timestamp,
)
import whisperx
AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 6000000
batch_size = 16
download_root_cache = "/home/gitlab/asr/models/"
DEFAULT_TARGET_LANGUAGE = "English"

if torch.cuda.is_available():
    cuda = torch.device("cuda:0")
    compute_type = torch.float16

    devicewhisper = "cuda"
else:
    cuda = torch.device("cpu")
    compute_type = torch.float32
    devicewhisper = "cpu"

computer_type = "float32" if compute_type == torch.float32 else "float32"  # float32"
custom_asr_options = custom_asr_options = {
    "max_new_tokens": 120,
    # (Add other relevant options for your torch ASR library)
}
model = whisperx.load_model("large-v2", devicewhisper, compute_type=computer_type, download_root=download_root_cache, threads=100,
                            asr_options=custom_asr_options

                            )
diarize_model = whisperx.DiarizationPipeline(
    use_auth_token="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", device=devicewhisper)


def run_asr(input_audio: str, target_language: str) -> str:
    gc.collect()
    torch.cuda.empty_cache()
    result = model.transcribe(
        input_audio, batch_size=batch_size, print_progress=False, chunk_size=10)

    diarize_segments = diarize_model(input_audio, max_speakers=2)
    result = whisperx.assign_word_speakers(
        diarize_segments, result)
    print(result)
    gc.collect()
    torch.cuda.empty_cache()
    Text = ""
    for i in result["segments"]:
        if i["end"] - i["start"] < 0.7:
            continue
        if "speaker" not in i:
            continue
        if i["text"] in [' Hẹn gặp lại các bạn trong những video tiếp theo nhé!', ' Cảm ơn các bạn đã theo dõi.']:

            continue

        Text += format_timestamp(i["start"]) + " " + format_timestamp(i["end"]) + " " + \
            i["speaker"] + " " + i["text"] + "\n"
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
