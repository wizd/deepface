# built-in dependencies
from typing import Union
import os
import logging

import tensorflow as tf
import keras

# 在导入TensorFlow之前设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TF日志输出

# 3rd party dependencies
from flask import Blueprint, request
import numpy as np
from keras import backend as K
from deepface.commons.logger import Logger

logger = Logger()

# 配置GPU内存
def configure_gpu():
    logger.info("开始检测GPU...")
    
    # 检查CUDA环境变量
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    # 检查TensorFlow是否能看到CUDA
    logger.info(f"TensorFlow是否支持CUDA: {tf.test.is_built_with_cuda()}")
    logger.info(f"TensorFlow是否支持GPU: {tf.test.is_built_with_gpu_support()}")
    
    try:
        # 设置TensorFlow日志级别
        tf.get_logger().setLevel('INFO')
        
        # 设置内存增长前先清理GPU内存
        keras.backend.clear_session()
        
        # 配置GPU内存使用
        gpus = tf.config.experimental.list_physical_devices('GPU')
        logger.info(f"检测到的GPU设备: {gpus}")
        
        if not gpus:
            logger.info("未检测到GPU设备，将使用CPU模式")
            return False
            
        # 尝试多种GPU配置方式
        try:
            # 方式1：限制GPU内存增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"已为设备 {gpu} 启用内存增长")
        except Exception as e:
            logger.info(f"设置内存增长失败，尝试其他配置方式: {e}")
            try:
                # 方式2：限制GPU内存使用量
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]
                    )
                logger.info("已限制GPU内存使用量为8GB")
            except Exception as e2:
                logger.error(f"所有GPU配置方式均失败: {e2}")
                return False
        
        # 验证GPU是否可用
        try:
            with tf.device('/GPU:0'):
                # 创建一个小的测试张量
                test = tf.random.normal([8, 8])
                logger.info(f"GPU测试成功，张量设备: {test.device}")
                logger.info("GPU配置成功完成")
                return True
        except Exception as e:
            logger.error(f"GPU验证失败: {e}")
            return False
            
    except Exception as e:
        logger.error(f"GPU配置过程中发生错误: {e}")
        return False

# 全局GPU配置
GPU_AVAILABLE = configure_gpu()
logger.info(f"GPU最终状态: {'可用' if GPU_AVAILABLE else '不可用'}")

if not GPU_AVAILABLE:
    logger.info("GPU不可用，将使用CPU模式运行。请确保：")
    logger.info("1. CUDA和cuDNN已正确安装")
    logger.info("2. TensorFlow-GPU版本与CUDA版本匹配")
    logger.info("3. LD_LIBRARY_PATH包含CUDA库路径")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用GPU

# project dependencies
from deepface import DeepFace
from deepface.api.src.modules.core import service
from deepface.commons import image_utils
from functools import wraps
import traceback
import base64
import cv2
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import threading

blueprint = Blueprint("routes", __name__)

# pylint: disable=no-else-return, broad-except

def error_handler(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}, 500
    return decorated_function

@blueprint.route("/")
def home():
    return f"<h1>Welcome to DeepFace API v{DeepFace.__version__}!</h1>"


def extract_image_from_request(img_key: str) -> Union[str, np.ndarray]:
    """
    Extracts an image from the request either from json or a multipart/form-data file.

    Args:
        img_key (str): The key used to retrieve the image data
            from the request (e.g., 'img1').

    Returns:
        img (str or np.ndarray): Given image detail (base64 encoded string, image path or url)
            or the decoded image as a numpy array.
    """

    # Check if the request is multipart/form-data (file input)
    if request.files:
        # request.files is instance of werkzeug.datastructures.ImmutableMultiDict
        # file is instance of werkzeug.datastructures.FileStorage
        file = request.files.get(img_key)

        if file is None:
            raise ValueError(f"Request form data doesn't have {img_key}")

        if file.filename == "":
            raise ValueError(f"No file uploaded for '{img_key}'")

        img = image_utils.load_image_from_file_storage(file)

        return img
    # Check if the request is coming as base64, file path or url from json or form data
    elif request.is_json or request.form:
        input_args = request.get_json() or request.form.to_dict()

        if input_args is None:
            raise ValueError("empty input set passed")

        # this can be base64 encoded image, and image path or url
        img = input_args.get(img_key)

        if not img:
            raise ValueError(f"'{img_key}' not found in either json or form data request")

        return img

    # If neither JSON nor file input is present
    raise ValueError(f"'{img_key}' not found in request in either json or form data")


@blueprint.route("/represent", methods=["POST"])
@error_handler
def represent():
    input_args = request.get_json() or request.form.to_dict()

    try:
        img = extract_image_from_request("img")
    except Exception as err:
        return {"exception": str(err)}, 400

    obj = service.represent(
        img_path=img,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=bool(input_args.get("enforce_detection", True)),
        align=bool(input_args.get("align", True)),
        anti_spoofing=bool(input_args.get("anti_spoofing", False)),
        max_faces=int(input_args.get("max_faces", 1)),
    )

    logger.debug(obj)

    return obj


@blueprint.route("/verify", methods=["POST"])
@error_handler
def verify():
    input_args = request.get_json() or request.form.to_dict()

    try:
        img1 = extract_image_from_request("img1")
    except Exception as err:
        return {"exception": str(err)}, 400

    try:
        img2 = extract_image_from_request("img2")
    except Exception as err:
        return {"exception": str(err)}, 400

    verification = service.verify(
        img1_path=img1,
        img2_path=img2,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        distance_metric=input_args.get("distance_metric", "cosine"),
        align=bool(input_args.get("align", True)),
        enforce_detection=bool(input_args.get("enforce_detection", True)),
        anti_spoofing=bool(input_args.get("anti_spoofing", False)),
    )

    logger.debug(verification)

    return verification


@blueprint.route("/analyze", methods=["POST"])
@error_handler
def analyze():
    input_args = request.get_json() or request.form.to_dict()

    try:
        img = extract_image_from_request("img")
    except Exception as err:
        return {"exception": str(err)}, 400

    actions = input_args.get("actions", ["age", "gender", "emotion", "race"])
    # actions is the only argument instance of list or tuple
    # if request is form data, input args can either be text or file
    if isinstance(actions, str):
        actions = (
            actions.replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .replace('"', "")
            .replace("'", "")
            .replace(" ", "")
            .split(",")
        )

    demographies = service.analyze(
        img_path=img,
        actions=actions,
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=bool(input_args.get("enforce_detection", True)),
        align=bool(input_args.get("align", True)),
        anti_spoofing=bool(input_args.get("anti_spoofing", False)),
    )

    logger.debug(demographies)

    return demographies


@blueprint.route("/extract_faces", methods=["POST"])
@error_handler
def extract_faces():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "空输入集合"}, 400

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "必须传入img_path参数"}, 400

    try:
        # 记录GPU使用情况
        if GPU_AVAILABLE:
            logger.info("Processing with GPU")
            # 确保TensorFlow使用GPU
            with tf.device('/GPU:0'):
                faces = DeepFace.extract_faces(
                    img_path=img_path,
                    detector_backend=input_args.get("detector_backend", "retinaface"),
                    enforce_detection=input_args.get("enforce_detection", True),
                    align=input_args.get("align", True),
                    expand_percentage=input_args.get("expand_percentage", 0),
                    anti_spoofing=input_args.get("anti_spoofing", False),
                )
        else:
            logger.info("Processing with CPU as GPU is not available")
            faces = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend=input_args.get("detector_backend", "retinaface"),
                enforce_detection=input_args.get("enforce_detection", True),
                align=input_args.get("align", True),
                expand_percentage=input_args.get("expand_percentage", 0),
                anti_spoofing=input_args.get("anti_spoofing", False),
            )

        # 优化图像处理和转换
        for face in faces:
            if "face" in face and isinstance(face["face"], np.ndarray):
                _, buffer = cv2.imencode('.jpg', face["face"], [cv2.IMWRITE_JPEG_QUALITY, 85])
                face["face"] = f"data:image/jpeg;base64,{base64.b64encode(buffer.tobytes()).decode('utf-8')}"

        return {"results": faces}
    except Exception as err:
        logger.error(f"Error processing image: {str(err)}")
        logger.error(traceback.format_exc())
        return {"error": f"提取人脸时发生异常: {str(err)}"}, 400
