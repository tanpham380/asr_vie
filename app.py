# app.py
from flask import render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from main import AudioOfficial
from util import clearFolderContent
from flask_cors import cross_origin
from flask import Flask, request, jsonify
from waitress import serve
import logging
from flasgger import Swagger
import gradio as gr
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
    







def transcribe(inputs , task):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
    text = ""
    # max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
    # if new_arr.shape[1] > max_length:
    #     new_arr = new_arr[:, :max_length]
    #     gr.Warning(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.")
    # text = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]
    data , Resuft = Detect.ExtractText(inputs)
    for segment in data:
        text += f"Start: {segment['start']:.2f} End: {segment['end']:.2f} Text: '{segment['text']}'. \n"

    return  text + "FullText: "  + Resuft





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
        gr.Audio(sources=["upload"], type="filepath", label="Audio file" , max_length = 360),


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
    gr.TabbedInterface([file_transcribe, mf_transcribe], ["Audio file","Microphone"], title = "Audio")


if __name__ == '__main__':
    Detect = AudioOfficial(
        down_nmodel_path = "./models/" ,
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
