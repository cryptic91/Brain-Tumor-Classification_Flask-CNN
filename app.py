import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# app = Flask(__name__)

# model = load_model('BrainTumor100Epochs.h5') 

# print('Model loaded. Check http://127.0.0.1:5000/') 

# def get_className(classNo):
#     if classNo == 0:
#         return "No Brain Tumor"
#     elif classNo == 1:
#         return "Glioma Brain Tumor"
#     elif classNo == 2:
#         return "Meningioma Brain Tumor"
#     elif classNo == 3:
#         return "Pituitary Brain Tumor"


# def getResult(img):
#     image=cv2.imread(img)
#     image = Image.fromarray(image, 'RGB')
#     image = image.resize((64, 64))
#     image=np.array(image)
#     input_img = np.expand_dims(image, axis=0)
#     result=model.predict_classes(input_img)
#     return result


# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')


# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         f = request.files['file']

#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)
#         value=getResult(file_path)
#         result=get_className(value) 
#         return result
#     return None


# if __name__ == '__main__':
#     app.run(debug=True)



import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('BrainTumor100Epochs.h5')

print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Glioma Brain Tumor"
    elif classNo == 2:
        return "Meningioma Brain Tumor"
    elif classNo == 3:
        return "Pituitary Brain Tumor"

def getResult(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB
    image = Image.fromarray(image)
    image = image.resize((64, 64))
    image = np.array(image)

    # Normalize the image
    image = image / 255.0

    # Expand dimensions to match the input shape of the model
    input_img = np.expand_dims(image, axis=0)

    # Predict the class
    result = model.predict(input_img)
    predicted_class = np.argmax(result, axis=1)[0]
    return predicted_class

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)
