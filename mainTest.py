# import cv2
# from keras.models import load_model
# from PIL import Image
# import numpy as np

# model=load_model('BrainTumor100Epochs.h5')

# image=cv2.imread('D:\\Code\\Flask\\BrainTumorClassification\\prediction\\Tr-gl_0014.jpg')

# img=Image.fromarray(image)

# img=img.resize((64,64))

# img=np.array(img)

# input_img=np.expand_dims(img, axis=0)

# result=model.predict_classes(input_img)
# print(result)


#######################################################


import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model('BrainTumor100Epochs.h5')

# Load and preprocess the image
image = cv2.imread('D:\\Code\\Flask\\BrainTumorClassification\\prediction\\Tr-pi_0014.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image from BGR to RGB
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)

# Normalize the image
img = img / 255.0

# Expand dimensions to match the input shape of the model
input_img = np.expand_dims(img, axis=0)

# Predict the class
result = model.predict(input_img)
predicted_class = np.argmax(result, axis=1)
print(predicted_class)




