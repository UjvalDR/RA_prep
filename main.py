from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model('SavedModel_V1.h5')
class_names = ['Negative', 'Positive']

app = Flask(__name__)

def predict(image):
    img = Image.open(image)
    img = img.resize((256,256))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    prediction = class_names [np.argmax(model.predict(img))]
    a = model.predict(img)
    confidence = max(a)[np.argmax(a)]*100
    return prediction, confidence.round(2)

# index route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# predict route
@app.route('/prediction', methods=['POST'])
def prediction():

    img = request.files['img']
    img.save('img.jpg')
    prediction, confidence = predict("img.jpg") 
    return render_template('prediction.html', data=[prediction, confidence])



if __name__ == '__main__':
    app.run(debug=True)