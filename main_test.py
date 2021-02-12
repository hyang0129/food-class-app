# Copyright 2015 Google Inc. All Rights Reserved.
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

import requests
import tensorflow as tf
import base64

# image_bytes = tf.io.encode_jpeg(tf.zeros((256, 256, 3), dtype=tf.uint8))

image_bytes = tf.io.read_file('dumplings.jpg')


# r = requests.get('http://127.0.0.1:8080/check_model')
url = 'http://127.0.0.1:8080/predict'
# url = 'https://tpu-44747.uc.r.appspot.com/predict'
r = requests.post(url, json=({'image_bytes': base64.b64encode(image_bytes.numpy()).decode('ascii')}))
#
print(r.status_code)
print(r.json())
