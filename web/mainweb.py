from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import os

import tornado.wsgi
import tornado.httpserver
from werkzeug.utils import secure_filename
from src.model import ModelIAM
import tensorflow as tf
from src.prediction import Prediction


import cv2
from keras.saving.legacy.model_config import model_from_json
from spellchecker import SpellChecker
from src.predict import Predict






app = Flask(__name__)

UPLOAD_FOLDER = 'D:/school-projects/year3sem1/licenta/summer/static/data'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.static_folder = 'D:\school-projects\year3sem1\licenta\summer\static'


data_directory = 'D:\school-projects\year3sem1\licenta\summer\static\data'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])






def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    #flash("")
    return "Welcome to Handwritten Text Recognition. How can we help you today?"

global ocr
ocr= Prediction()


@app.route('/image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        destination = "/".join([data_directory, filename])
        print(destination)
        prediction= ocr.predict_photo_text(destination)
        prediction_str=str(prediction)#to be serializable to json
        print(prediction_str)
        return jsonify({'prediction': prediction_str})
    else:
        return jsonify({'error': 'Allowed image types are - png, jpg, jpeg, gif, bmp'})


@app.route('/display/<filename>')
def display_image(filename):
    return jsonify({'url': url_for('static', filename='data/' + filename)})
"""
@app.route('/display/<filename>')
def display_image(filename):
    return jsonify({'url': url_for('static', filename='data/' + filename)})
"""





def start_tornado(app, port=8080):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()



if __name__ == "__main__":
    #app.run(debug=True)
    start_tornado(app)


