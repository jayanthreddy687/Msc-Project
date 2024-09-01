# Importing required libs
from keras.utils import img_to_array
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.applications.densenet import preprocess_input


def load_image(img_path):
    op_img = Image.open(img_path)
    if op_img.mode != 'RGB':
        op_img = op_img.convert('RGB')
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize)
    img2arr = preprocess_input(img2arr)
    return img2arr


def predict(image, encoder, decoder, image_features_extract_model, vocabulary):
    word_to_index = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=vocabulary)

    index_to_word = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=vocabulary,
        invert=True)

    max_length = 25
    hidden = tf.zeros((1, 1024 * 2))
    temp_input = tf.expand_dims(image, 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(
        img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([word_to_index('startseq')], 0)
    result = []
    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(
            dec_input, features, hidden)
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())
        result.append(predicted_word)
        if predicted_word == 'endseq':
            return result
        dec_input = tf.expand_dims([predicted_id], 0)
        dec_input = tf.cast(dec_input, tf.int64)
    return result
