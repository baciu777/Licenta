import secrets
from datetime import time

from flask import Flask, flash, request, redirect, url_for, render_template, jsonify,g
import os
import magic
import tornado.wsgi
import tornado.httpserver
from werkzeug.utils import secure_filename
from bleach import clean
from src.model import ModelIAM
import tensorflow as tf
import threading
from src.segmentation import Segmentation


import cv2
from keras.saving.legacy.model_config import model_from_json
from spellchecker import SpellChecker
from src.predict import Predict




#13:52-cred

app = Flask(__name__)
import quart


UPLOAD_FOLDER = 'D:/school-projects/year3sem1/licenta/summer/static/data'


app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.static_folder = 'D:\school-projects\year3sem1\licenta\summer\static'


data_directory = 'D:\school-projects\year3sem1\licenta\summer\static\data'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])






def allowed_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False
    # Validate file type based on content---code injection
    mime_type = magic.from_buffer(request.files['image'].read(1024), mime=True)
    if not mime_type.startswith('image'):
        return False
    return True

import asyncio

@app.route('/')
def home():
    return "Welcome to Digital Hand! Click on the Capture Image button to capture or choose an image from gallery to perform text recognition."

#global ocr
#ocr= Segmentation()




@app.route('/image', methods=['POST'])
def upload_image():
    print('-------------------------------------')
    ocr = Segmentation()

    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if not allowed_file(file.filename):
        return jsonify({'error': 'Allowed image types are - png, jpg, jpeg, gif, bmp'})
    # Limit file size
    if len(file.read()) > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({'error': 'File too large'})
    file.seek(0)
    # Sanitize user input
    filename = clean(secure_filename(file.filename[:-4]+threading.current_thread().name+'.png'))
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    destination = "/".join([data_directory, filename])
    print(destination)
    prediction = ocr.predict_photo_text(destination)
    prediction_str = str(prediction)  # to be serializable to json
    print(prediction_str)
    return jsonify({'prediction': prediction_str})



@app.route('/display/<filename>')
def display_image(filename):
    return jsonify({'url': url_for('static', filename='data/' + filename)})





def start_tornado(app, port=8080):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()



if __name__ == "__main__":
    app.run(host='192.168.0.125',port=8080,threaded=True)
    #start_tornado(app)


