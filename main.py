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


import sys
from loguru import logger
from flask import Flask, request, jsonify
import tensorflow as tf
import google.cloud.logging
from google.auth.exceptions import DefaultCredentialsError
import base64
from src import get_model, predict_image

try:
    client = google.cloud.logging.Client()
    client.get_default_handler()
    client.setup_logging()
    logger.add(sys.stderr,
               format="{level} {message}",
               level="INFO",
               backtrace=True,
               diagnose=True,)
except DefaultCredentialsError:
    logger.add("server.log",
               rotation="50 MB",
               backtrace=True,
               diagnose=True,
               format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
    logger.add(sys.stderr,
               format="{time} {level} {message}",
               level="INFO",
               backtrace=True,
               diagnose=True,)


MODEL_BUCKET = 'kaggledata2'

# MODEL_FILENAME = 'foodclass/model.h5'
MODEL_FILENAME = 'foodclass/fp16.tflite'
MODEL = None

app = Flask(__name__)

@app.before_first_request
def _load_model():
    global MODEL
    MODEL = get_model(MODEL_BUCKET, MODEL_FILENAME)
    logger.debug('successfully loaded model')

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
    logger.debug('receiving prediction request')
    jpegbytes = request.get_json()['image_bytes']

    jpegbytes = base64.b64decode(jpegbytes)
    print(jpegbytes[:5])
    label = predict_image(MODEL, jpegbytes)
    return jsonify({'label': label}), 200
    # return label, 200

@app.errorhandler(500)
def server_eerror(e):
    logger.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver Yprocess such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python3_render_template]
# [END gae_python38_render_template]
