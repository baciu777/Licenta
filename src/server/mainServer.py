import secrets
from flask import Flask, flash, request, redirect,  jsonify
import os
import magic
from werkzeug.utils import secure_filename
from bleach import clean
from src.model import ModelIAM
import tensorflow as tf
import threading
from src.segmentation import Segmentation

from src.prediction import Prediction


app = Flask(__name__)


UPLOAD_FOLDER = 'D:/school-projects/year3sem1/licenta/summer/static/data'


app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.static_folder = 'D:\school-projects\year3sem1\licenta\summer\static'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


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

@app.route('/')
def home():
    return "Welcome to Digital Hand! Click on the capture image button to capture or choose an image from gallery to perform text recognition."



session = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(session)


global graph
graph = tf.Graph()

# you create a "cache" attribute for the app.
app.cache = {}
app.cache['foo'] = Segmentation()



@app.route('/image', methods=['POST'])
def upload_image():
    ocr=app.cache['foo']
    with session.as_default():
        with graph.as_default():
            if 'image' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['image']

            if file.filename == '':
                return redirect(request.url)
            if not allowed_file(file.filename):
                return jsonify({'error': 'Allowed image types are - png, jpg, jpeg'})
            # Limit file size
            if len(file.read()) > app.config['MAX_CONTENT_LENGTH']:
                return jsonify({'error': 'File too large'})
            file.seek(0)

            # Sanitize user input
            filename = clean(secure_filename(file.filename[:-4]+threading.current_thread().name+'.png'))
            destination = os.path.join(app.static_folder,'data', filename)
            file.save(destination)
            #print(destination)
            prediction = ocr.predict_photo_text(destination)
            prediction_str = str(prediction)  # to be serializable to json
            #print(prediction_str)
            os.remove(destination)
            return jsonify({'prediction': prediction_str})





if __name__ == "__main__":
    app.run(host='192.168.0.125',port=8080,debug=False,threaded=True)


