# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import logging
import os

from flask import Flask, request, jsonify
import tensorflow as tf
from google.cloud import storage


MODEL_BUCKET = None
MODEL_FILENAME = None
MODEL = None

app = Flask(__name__)

@app.before_first_request
def _load_model():
    global MODEL
    # client = storage.Client()
    # bucket = client.get_bucket(MODEL_BUCKET)
    # blob = bucket.get_blob(MODEL_FILENAME)
    #
    # blob.download_to_filename('model.h5')
    #
    # MODEL = tf.keras.models.load_model('model.h5', compile = False)

    MODEL = tf.keras.applications.EfficientNetB4(input_shape=(256,256,3),
                                                 weights=None,
                                                 classes=101)


@app.route('/')
def root():
    hello = tf.constant('This web address should only be accessed via the app and not directly')
    return hello.numpy()

@app.route('/check_model', methods=['GET'])
def index():
    global MODEL
    return str(MODEL.name), 200

@app.route('/predict', methods=['POST'])
def predict():
    global MODEL
    b = request.get_json()['image_bytes']
    b = eval(b)
    image = tf.io.decode_jpeg(tf.cast(b, tf.string))
    image = tf.cast(image, 'float32')/255.
    image = tf.image.resize_with_crop_or_pad(image, target_height=256, target_width=256)
    image = tf.reshape(image, (1, 256, 256, 3))
    label = MODEL.predict(image).tolist()
    return jsonify({'label': label}), 200

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python3_render_template]
# [END gae_python38_render_template]
