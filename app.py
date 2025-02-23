from flask import Flask, render_template, request
from flask import jsonify
from PIL import Image, ImageOps
from classifier import Classifier
import numpy as np
import uuid
import tempfile
import shutil
from flask_cors import CORS, cross_origin

classifier = Classifier()
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

size = 64,64
dirpath = tempfile.mkdtemp()

def load_image(infilename) :
  img = Image.open(infilename)
  img.load()
  data = np.asarray(ImageOps.fit(img, size, Image.ANTIALIAS), dtype="int32" )
  return data

@app.route('/')
@cross_origin()
def index(): 
  return render_template('index.html')

classes = ['Dias 0 a 1', 'Dias 1 a 2', 'Dias 4 a 5', 'Dias 7 a 10']

@app.route('/predict', methods=['POST'])
def upload():
  if request.method == 'POST':
    
    uploadedFile = request.files['uploadedFile']
    
    filename = str(uuid.uuid4())

    uploadedFile.save(f'{dirpath}/{filename}')

    im = load_image(f'{dirpath}/{filename}')

    prediction = classifier.model.predict_classes(np.array([im]))

    class_id = int(prediction[0])
    
    classification_result = {
      'class_id': class_id,
      'prediction': classes[class_id]
    }

    return jsonify(classification_result)
