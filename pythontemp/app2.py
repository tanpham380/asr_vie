from flask import render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from main import AudioOfficial
from util import clearFolderContent
from flask_cors import cross_origin
from flask import Flask, request
from flasgger import Swagger
from waitress import serve
import logging
from flask import redirect
from flask import url_for
from flask import send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = './run/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['mp3', 'wav'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

swagger = Swagger(app)
@app.route("/video/<filename>")
def get_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/")
def home():
    return redirect(url_for('flasgger.apidocs'))

@app.route('/api', methods=['POST'])
def extractvoicetext():
    """
    Extracts text from audio file.

    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: no
        description: The audio file to process (MP3 or WAV).
    responses:
      200:
        description: Text extracted successfully.
        schema:
          type: object
          properties:
            text:
              type: string
              description: Extracted text from the audio file.
            video_url:
              type: string
              description: URL to play the video.
    """
    clearFolderContent(app.config['UPLOAD_FOLDER'])
    if 'file' not in request.files:
        return jsonify({'error': 'input image not provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No input image selected'}), 401
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # RUN DETECT HERE
        json = Detect.ExtractText(app.config['UPLOAD_FOLDER'] + filename)
        return jsonify(json)
    else:
        return jsonify({'error': 'Invalid file type'}), 402
    
@app.before_request
def log_request_info():
    app.logger.info('Request Headers: %s', request.headers)
    app.logger.info('Request Method: %s', request.method)
    app.logger.info('Request URL: %s', request.url)
    app.logger.info('Request Data: %s', request.data)
    
if __name__ == '__main__':
    Detect = AudioOfficial(
        # audio_path="./models/seamless-m4t-v2-large/",
        down_nmodel_path = "./models/" ,
        model = "large-v3",
    )
    logging.basicConfig(
        filename='waitress.log',
        format='%(asctime)s [%(levelname)s] %(message)s',
        level=logging.INFO
    )
    serve(app, host='0.0.0.0', port=5000)
