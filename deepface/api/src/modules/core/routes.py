# built-in dependencies
from typing import Union

# 3rd party dependencies
from flask import Blueprint, request
import numpy as np
import tensorflow as tf
from deepface.commons.logger import Logger

logger = Logger()

# GPU配置
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 使用固定内存限制而不是动态增长
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]  # 限制使用8GB显存
        )
        logger.info(f"GPU available and configured with 8GB memory limit: {gpus}")
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")
        # 如果配置失败，尝试禁用GPU
        try:
            tf.config.set_visible_devices([], 'GPU')
            logger.error("GPU has been disabled due to configuration error")
        except:
            pass

# project dependencies
from deepface import DeepFace
from deepface.api.src.modules.core import service
from deepface.commons import image_utils
import numpy as np
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


# 创建线程池
thread_pool = ThreadPoolExecutor(max_workers=4)
thread_local = threading.local()

def get_deepface():
    if not hasattr(thread_local, "deepface"):
        thread_local.deepface = DeepFace
    return thread_local.deepface

@blueprint.route("/extract_faces", methods=["POST"])
@error_handler
def extract_faces():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "空输入集合"}, 400

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "必须传入img_path参数"}, 400

    def process_image():
        try:
            deepface = get_deepface()
            faces = deepface.extract_faces(
                img_path=img_path,
                detector_backend=input_args.get("detector_backend", "retinaface"),  # 默认使用更快的检测器
                enforce_detection=input_args.get("enforce_detection", True),
                align=input_args.get("align", True),
                expand_percentage=input_args.get("expand_percentage", 0),
                anti_spoofing=input_args.get("anti_spoofing", False),
            )

            # 优化图像处理和转换
            for face in faces:
                if "face" in face and isinstance(face["face"], np.ndarray):
                    # 用更高效的图像编码方式
                    _, buffer = cv2.imencode('.jpg', face["face"], [cv2.IMWRITE_JPEG_QUALITY, 85])
                    face["face"] = f"data:image/jpeg;base64,{base64.b64encode(buffer.tobytes()).decode('utf-8')}"

            return {"results": faces}
        except Exception as err:
            logger.error(f"Error processing image: {str(err)}")
            logger.error(traceback.format_exc())
            return {"error": f"提取人脸时发生异常: {str(err)}"}, 400

    # 异步提交任务到线程池
    future = thread_pool.submit(process_image)
    return future.result()
