# app.py
import time
from main import AudioOfficial

import logging
import gradio as gr

from util import ASR_TARGET_LANGUAGE_NAMES, update_status


def update_value(value):
    return value


def transcribe(inputs, chunk=16, language="Vietnamese"):
    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please upload or record an audio file before submitting your request.")
    start_time = time.time()

    Result1, Result2 = Detect.ExtractText(
        inputs, chunk=chunk, language=language)
    end_time = time.time()

    if isinstance(Result1, str):
        result1_str = Result1
    else:
        result1_str = str(Result1)  # Convert to string

    if isinstance(Result2, str):
        result2_str = Result2
    else:
        result2_str = str(Result2)  # Convert to string

    total_time = f"Transcription took {end_time - start_time:.2f} seconds"

    return result1_str + "\n" + result2_str + "\n" + total_time


demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Slider(4, 32, step=1, label="Chunk Value"),
        gr.Dropdown(
            label="Target language",
            choices=ASR_TARGET_LANGUAGE_NAMES,
            value="Vietnamese",
        )
        # gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),

    ],
    outputs="text",
    # theme="huggingface",
    title="Speak To Text",
    description=(

    ),
    allow_flagging="never",
)


file_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[

        gr.Audio(sources=["upload"], type="filepath", label="Audio file"  # , max_length = 360
                 ),
        gr.Slider(4, 32, value=12, step=1, label="Chunk Value"),
        gr.Dropdown(
            label="Target language",
            choices=ASR_TARGET_LANGUAGE_NAMES,
            value="Vietnamese",
        )



    ],
    outputs="text",
    title="Audio to Text",
    description=(
        "Upload an audio file or record a new one to transcribe, WAV and MP3 files are supported. Just < 100 MB."
    ),
    allow_flagging="never",
)


# setup model
with demo:
    gr.TabbedInterface([file_transcribe, mf_transcribe], [
                       "Audio file", "Microphone"], title="Audio")


# with demo:
#     gr.TabbedInterface([mf_transcribe, ], ["Audio",])

    with gr.Row():
        refresh_button = gr.Button("Refresh Status")  # Create a refresh button

    sys_status_output = gr.Textbox(label="System Status", interactive=False)


#     # Link the refresh button to the refresh_status function
    refresh_button.click(update_status, None, [sys_status_output])

#     # Load the initial status using update_status function
    demo.load(update_status, inputs=None, outputs=[
              sys_status_output], every=2, queue=False)

#     graudio.stop_recording(handle_upload_audio,inputs=[graudio,grmodel_textbox,groutputs[0]],outputs=groutputs)
#     graudio.upload(handle_upload_audio,inputs=[graudio,grmodel_textbox,groutputs[0]],outputs=groutputs)
if __name__ == '__main__':
    Detect = AudioOfficial(
        down_nmodel_path="./models/",
        # vadfilter=None,
    )
    logging.basicConfig(
        filename='waitress.log',
        format='%(asctime)s [%(levelname)s] %(message)s',
        level=logging.INFO
    )
    demo.launch(server_port=7860,
                server_name="127.0.0.1",
                show_error=True,
                ssl_verify=False,  # Keep this for development only
                # ssl_certfile="cer/172.18.249.222.crt",
                # ssl_keyfile="cer/172.18.249.222.key",
                )
    # serve(app, host='0.0.0.0', port=5000 , )
