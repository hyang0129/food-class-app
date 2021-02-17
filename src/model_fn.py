from loguru import logger
import tensorflow as tf

from google.cloud import storage

import efficientnet.tfkeras as efn
from src.class_names import CLASS_NAMES


def get_model(model_bucket, model_filename):

    logger.debug('loading model')

    try:

        if model_bucket is None:

            model = efn.EfficientNetB3(input_shape=(256, 256, 3),
                                       weights=None,
                                       classes=101)

            logger.debug('loaded dummy model')
        else:


            logger.debug('downloading model file')
            client = storage.Client.create_anonymous_client()
            bucket = client.get_bucket(model_bucket)
            blob = bucket.get_blob(model_filename)


            if '.h5' in model_filename:
                try:
                    path = '/tmp/model.h5'
                    blob.download_to_filename(path)
                except FileNotFoundError:
                    path = 'model.h5'
                    blob.download_to_filename(path)

                logger.debug('loading model')
                model = tf.keras.models.load_model(path, compile=False)

            elif '.tflite' in model_filename:
                try:
                    path = '/tmp/model.tflite'
                    blob.download_to_filename(path)
                except FileNotFoundError:
                    path = 'model.tflite'
                    blob.download_to_filename(path)

                logger.debug('loading tflite model')

                interpreter = tf.lite.Interpreter(model_path=path)
                interpreter.allocate_tensors()

                model = interpreter

    except:

        raise

    return model

def predict_image(model, jpegbytes):

    try:
        logger.debug('preparing data for prediction')
        image = tf.io.decode_jpeg(tf.cast(jpegbytes, tf.string))

        image = tf.cast(image, tf.float32)

        image = image / 255.

        image = tf.image.resize_with_pad(image, target_height=256, target_width=256)
        image = tf.reshape(image, (1, 256, 256, 3))

        logger.debug('predicting on prepared data')

        if model.__class__ == tf.lite.Interpreter:
            input_index = model.get_input_details()[0]["index"]
            output_index = model.get_output_details()[0]["index"]

            model.set_tensor(input_index, image)
            model.invoke()
            pred = model.get_tensor(output_index)

        else:
            pred = model.predict(image)
            try:
                pred = pred['label']
            except:
                pass

        label_id = tf.argmax(pred, axis=-1).numpy()[0]
        label = CLASS_NAMES[label_id]

        logger.debug('returning prediction')
    except:
        logger.debug('error encountered during prediction process')
        raise

    return label
