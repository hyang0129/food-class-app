from loguru import logger
import tensorflow as tf

from google.cloud import storage

import efficientnet.tfkeras as efn
from src.class_names import class_names


def get_model(MODEL_BUCKET, MODEL_FILENAME):

    logger.debug('loading model')

    try:

        if MODEL_BUCKET is None:

            MODEL = efn.EfficientNetB3(input_shape=(256, 256, 3),
                                                         weights=None,
                                                         classes=101)

            logger.debug('loaded dummy model')
        else:


            logger.debug('downloading model file')
            client = storage.Client.create_anonymous_client()
            bucket = client.get_bucket(MODEL_BUCKET)
            blob = bucket.get_blob(MODEL_FILENAME)


            if '.h5' in MODEL_FILENAME:
                try:
                    path = '/tmp/model.h5'
                    blob.download_to_filename(path)
                except FileNotFoundError:
                    path = 'model.h5'
                    blob.download_to_filename(path)

                logger.debug('loading model')
                MODEL = tf.keras.models.load_model(path, compile=False)

            elif '.tflite' in MODEL_FILENAME:
                try:
                    path = '/tmp/model.tflite'
                    blob.download_to_filename(path)
                except FileNotFoundError:
                    path = 'model.tflite'
                    blob.download_to_filename(path)

                logger.debug('loading tflite model')

                interpreter = tf.lite.Interpreter(model_path=path)
                interpreter.allocate_tensors()

                MODEL = interpreter

    except:

        raise

    return MODEL

def predict_image(MODEL, jpegbytes):

    try:
        logger.debug('preparing data for prediction')
        image = tf.io.decode_jpeg(tf.cast(jpegbytes, tf.string))

        image = tf.cast(image, tf.float32)

        image = image / 255.

        image = tf.image.resize_with_pad(image, target_height=256, target_width=256)
        image = tf.reshape(image, (1, 256, 256, 3))

        logger.debug('predicting on prepared data')

        if MODEL.__class__ == tf.lite.Interpreter:
            input_index = MODEL.get_input_details()[0]["index"]
            output_index = MODEL.get_output_details()[0]["index"]

            MODEL.set_tensor(input_index, image)
            MODEL.invoke()
            pred = MODEL.get_tensor(output_index)

        else:
            pred = MODEL.predict(image)
            try:
                pred = pred['label']
            except:
                pass

        label_id = tf.argmax(pred, axis=-1).numpy()[0]
        label = class_names[label_id]

        logger.debug('returning prediction')
    except:
        logger.debug('error encountered during prediction process')
        raise

    return label
