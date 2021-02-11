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
            client = storage.Client()
            bucket = client.get_bucket(MODEL_BUCKET)
            blob = bucket.get_blob(MODEL_FILENAME)
            blob.download_to_filename('model.h5')

            logger.debug('loading model')
            MODEL = tf.keras.models.load_model('model.h5', compile=False)

    except:

        try:
            logger.debug('loading model from gcs directly')
            MODEL = tf.keras.models.load_model('gs://kaggledata2/foodclass/model.h5', compile=False)

        except:
            logger.exception('failed to load model')
            raise


    return MODEL

def predict_image(MODEL, jpegbytes):

    try:
        logger.debug('preparing data for prediction')
        image = tf.io.decode_jpeg(tf.cast(jpegbytes, tf.string))
        image = tf.cast(image, 'float32')/255.
        image = tf.image.resize_with_crop_or_pad(image, target_height=256, target_width=256)
        image = tf.reshape(image, (1, 256, 256, 3))

        logger.debug('predicting on prepared data')
        label_id = tf.argmax(MODEL.predict(image), axis=-1).numpy()[0]
        label = class_names[label_id]

        logger.debug('returning prediction')
    except:
        logger.debug('error encountered during prediction process')
        raise

    return label
