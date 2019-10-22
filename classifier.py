import os
from keras.models import load_model

class Classifier(): 
  def __init__(self):
    path = './model.h5'
    if os.path.isfile(path):
      self.model = load_model(path)
      self.model._make_predict_function()
    else:
      print('No existe el archivo ', path)