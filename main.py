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
    return prediction

# index route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# predict route
@app.route('/prediction', methods=['POST'])
def prediction():

    img = request.files['img']
    img.save('img.jpg')
    prediction = predict("img.jpg") 
    return render_template('prediction.html', data=prediction)



if __name__ == '__main__':
    app.run(debug=True)