"""
reference
https://medium.com/@draj0718/deploying-deep-learning-model-using-flask-api-810047f090ac
"""
from flask import Flask,request,render_template
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img

app = Flask(__name__)
model = load_model('covid_model_git.h5')
target_img = os.path.join(os.getcwd(),'static/test_images')

@app.route('/',methods=['GET'])
def index_view():
    return render_template('index.html')

# Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXT

# Function to load and prepare the image in right shape
def read_image(filename):
    imageW = 64
    imageH = 64
    img = image.load_img(filename,target_size=(imageW, imageH))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/test_images', filename)
            file.save(file_path)
            img = read_image(file_path)
            class_prediction = model.predict(img)
            classes_x = np.argmax(class_prediction, axis=1)
            if classes_x == 0:
                CT = "Covid"
            elif classes_x == 1:
                CT = "Normal"
            else:
                CT = "Viral Pneumonia"
            return render_template('predict.html', CT=CT, prob=class_prediction, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"

if __name__ == '__main__':
   app.run(debug=True, port=8000)

