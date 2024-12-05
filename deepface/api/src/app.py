# GPU configuration must be set before importing TensorFlow
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Enable memory growth for all GPUs
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            
        # Allow TensorFlow to use all GPU memory
        for gpu in physical_devices:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0)]
            )
        print(f"Found {len(physical_devices)} Physical GPUs")
    except RuntimeError as e:
        print(e)

# 3rd parth dependencies
from flask import Flask
from flask_cors import CORS

# project dependencies
from deepface import DeepFace
from deepface.api.src.modules.core.routes import blueprint
from deepface.commons.logger import Logger

logger = Logger()


def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(blueprint)
    logger.info(f"Welcome to DeepFace API v{DeepFace.__version__}!")
    return app
