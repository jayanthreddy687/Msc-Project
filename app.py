# Importing required libs
from flask import Flask, render_template, request
from model import load_image, predict
import tensorflow as tf
# Instantiating flask app
app = Flask(__name__,template_folder='C:/Users/jayan/OneDrive/Desktop/Msc Project/Webapp/mywebapp/static/templates')

impression_encoder = tf.saved_model.load('models/impression_encoder')
impression_decoder = tf.saved_model.load('models/impression_decoder')
findings_encoder = tf.saved_model.load('models/findings_encoder')
findings_decoder = tf.saved_model.load('models/findings_decoder')
indication_encoder = tf.saved_model.load('models/indication_encoder')
indication_decoder = tf.saved_model.load('models/indication_decoder')
image_features_extract_model = tf.saved_model.load('models/image_features_extract_model')

# Home route
@app.route("/home")
def main():
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    try:
        if request.method == 'POST':
            img = load_image(request.files['file'].stream)
            pred = predict(img)
            return render_template("result.html", predictions=str(pred))

    except Exception as e:
        error = "File cannot be processed."
        return render_template("result.html", err=error)

# Driver code
if __name__ == "__main__":
    app.run(port=8005, debug=True)
