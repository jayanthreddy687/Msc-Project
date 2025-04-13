# Importing required libs
from flask import Flask, render_template, request
from model import load_image, predict
import tensorflow as tf
import asyncio
import re
# Instantiating flask app
app = Flask(__name__, template_folder='C:/Users/jayan/OneDrive/Desktop/Msc Project/Webapp/mywebapp/static/templates')

async def model() :
    model_paths = {
                    "impression_encoder": 'models/impression_encoder',
                    "impression_decoder": 'models/impression_decoder',
                    "findings_encoder": 'models/findings_encoder',
                    "findings_decoder": 'models/findings_decoder',
                    "indication_encoder": 'models/indication_encoder',
                    "indication_decoder": 'models/indication_decoder',
                    "image_features_extract_model": 'models/image_features_extract_model'
                }
    
    model_tasks = {name: load_model_async(path)
                               for name, path in model_paths.items()}
    return await asyncio.gather(*model_tasks.values()) 

async def load_model_async(path):
        return await asyncio.to_thread(tf.saved_model.load, path)

def get_vocab(path):
    with open(path, 'r') as f:
        vocabulary = [line.strip() for line in f]
    return vocabulary


async def run_prediction_async(img, encoder, decoder, image_features_extract_model, vocab):
    pred = await asyncio.to_thread(predict, img, encoder, decoder, image_features_extract_model, vocab)
    return pred


def get_vocab(path):
    with open(path, 'r') as f:
        vocabulary = [line.strip() for line in f]
    return vocabulary


async def load_vocab_async(path):
    return await asyncio.to_thread(get_vocab, path)

@app.route("/")
async def main():
    return render_template("index.html")

def textprocessing(txt):
    txt= str(txt)
    txt = txt.lower()
    txt = re.sub(r"endseq","",txt)
    txt = re.sub(r"startseq","",txt)
    txt.strip()
    return txt
# Prediction route
@app.route('/prediction', methods=['POST'])
async def predict_image_file():
    try:
        if request.method == 'POST':
            img = load_image(request.files['file'].stream)
            loaded_models = await model()

            # Assign the loaded models to their respective variables
            (impression_encoder, impression_decoder, findings_encoder,
             findings_decoder, indication_encoder, indication_decoder,
             image_features_extract_model) = loaded_models

            # Define paths to the vocabulary files
            vocab_path = {
                "impression_vocabulary": 'impression_vocabulary.txt',
                "findings_vocabulary": 'findings_vocabulary.txt',
                "indication_vocabulary": 'indication_vocabulary.txt'
            }
            # Load vocabularies asynchronously
            vocab_tasks = {name: load_vocab_async(
                path) for name, path in vocab_path.items()}
            loaded_vocabularies = await asyncio.gather(*vocab_tasks.values())

            # Assign the loaded vocabularies to their respective variables
            impression_vocabulary, findings_vocabulary, indication_vocabulary = loaded_vocabularies

            # Run predictions for each encoder-decoder pair in parallel
            prediction_tasks = [
                run_prediction_async(img, impression_encoder,
                                     impression_decoder, image_features_extract_model, impression_vocabulary),
                run_prediction_async(img, findings_encoder,
                                     findings_decoder, image_features_extract_model, findings_vocabulary),
                run_prediction_async(img, indication_encoder,
                                     indication_decoder, image_features_extract_model, indication_vocabulary),
            ]

            # Gather results from all tasks
            results = await asyncio.gather(*prediction_tasks)

            impression_prediction, findings_prediction, indication_prediction = results

            caption = ' '.join(indication_prediction) +'.\n' + ' '.join(impression_prediction) +'.\n' +' '.join(findings_prediction) + '.'
            caption = textprocessing(caption)
            # You can now use `impression_prediction`, `findings_prediction`, and `indication_prediction` as needed.
            print("caption:", caption)

            return render_template("result.html", predictions=str(caption))

    except Exception as e:
        error = "File cannot be processed."
        return render_template("result.html", err=e)


# Driver code
if __name__ == "__main__":
    app.run(port=8005, debug=True)
