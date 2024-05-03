# app.py
import time
from main import AudioOfficial

import logging
import gradio as gr

from util import ASR_TARGET_LANGUAGE_NAMES
# swagger = Swagger(app)

# app = Flask(__name__)
# UPLOAD_FOLDER = './run/uploads/'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ALLOWED_EXTENSIONS = set(['mp3', 'wav'])

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route('/api', methods=['POST'])
# def extractvoicetext():
#     clearFolderContent(app.config['UPLOAD_FOLDER'])
#     if 'file' not in request.files:
#         return jsonify({'error': 'input image not provided'}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No input image selected'}), 401
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         # RUN DETECT HERE
#         json = Detect.ExtractText(app.config['UPLOAD_FOLDER'] + filename)
#         return jsonify(json)
#     else:
#         return jsonify({'error': 'Invalid file type'}), 402


def transcribe(inputs, language="English"):
    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please upload or record an audio file before submitting your request.")
    start_time = time.time()
    text1 = ""
    text2 = ""
    text3 = ''
    Result1, Result2, a = Detect.ExtractText(inputs)
    if Result1 is not None:
        for text in Result1:
            text1 += f"{text}. \n"
    # if Result2 is not None:
    #     for text in Result2 :
    #         text2 += f"{text}. \n"
    if Result2 is not None:
        text2 = Result2
    for key, value in a.items():
        text3 += f"{key}: {value}"
    end_time = time.time()
    total_time = f"Transcription took {end_time - start_time:.2f} seconds"
    return text1 + "\n" + text2 + "\n" + total_time + "\n" + text3


demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
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
        gr.Dropdown(
            label="Target language",
            choices=ASR_TARGET_LANGUAGE_NAMES,
            value="English",
        )


        # gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),
    ],
    outputs="text",
    # theme="huggingface",
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

#     with gr.Row():
#         refresh_button = gr.Button("Refresh Status")  # Create a refresh button

#     sys_status_output = gr.Textbox(label="System Status", interactive=False)


#     # Link the refresh button to the refresh_status function
#     refresh_button.click(refresh_status, None, [sys_status_output])

#     # Load the initial status using update_status function
#     demo.load(update_status, inputs=None, outputs=[sys_status_output], every=2, queue=False)

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
