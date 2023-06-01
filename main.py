import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import uuid

app = Flask(__name__)

# Load the trained model
model_path = 'model.json'
model = tf.keras.models.model_from_json(open(model_path).read())
model.load_weights('model_weights.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the HTML form
    file = request.files['image']
    # Generate a unique filename for each uploaded image
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    file_path = os.path.join('static', filename)
    if not os.path.exists('static'):
        os.makedirs('static')
    file.save(file_path)

    # Load the image and process it with the model
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(img, 15, 75, 75)
    cv2.imwrite(file_path, bilateral)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, (150,150))
    img_array = np.array(img)
    img_array = img_array.reshape(1,150,150,3)
    a = model.predict(img_array)
    result = a.argmax()
    arr=['Glioma_tumor','Meningioma_tumor','No_tumor','Pituitary_tumor']
    prediction = arr[result]

    # Return the prediction to the HTML page
    return render_template('index.html', prediction=prediction, image=filename)

if __name__ == '__main__':
    app.run(debug=True)
