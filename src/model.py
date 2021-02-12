from loguru import logger
import tensorflow as tf

from google.cloud import storage

import efficientnet.tfkeras as efn
from src.class_names import class_names

mean = [103.939, 116.779, 123.68]
std = [58.393, 57.12, 57.375]




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
            client = storage.Client()
            bucket = client.get_bucket(MODEL_BUCKET)
            blob = bucket.get_blob(MODEL_FILENAME)
            blob.download_to_filename('/tmp/model.h5')

            logger.debug('loading model')
            MODEL = tf.keras.models.load_model('/tmp/model.h5', compile=False)

    except:

        pass

    return MODEL

def predict_image(MODEL, jpegbytes):

    try:
        logger.debug('preparing data for prediction')
        image = tf.io.decode_jpeg(tf.cast(jpegbytes, tf.string))

        image = tf.cast(image, tf.float32)

        image = image - mean
        image = image - std

        image = image / 255.

        image = tf.image.resize_with_crop_or_pad(image, target_height=256, target_width=256)
        image = tf.reshape(image, (1, 256, 256, 3))

        logger.debug('predicting on prepared data')
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
