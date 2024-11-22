# built-in dependencies
from typing import Union

# 3rd party dependencies
from flask import Blueprint, request
import numpy as np

# project dependencies
from deepface import DeepFace
from deepface.api.src.modules.core import service
from deepface.commons import image_utils
from deepface.commons.logger import Logger
import numpy as np
from functools import wraps
import traceback
import base64
import cv2
from io import BytesIO
from PIL import Image

logger = Logger()

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

    faces = service.extract_faces(
        img_path=img_path,
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=input_args.get("enforce_detection", True),
        align=input_args.get("align", True),
        expand_percentage=input_args.get("expand_percentage", 0),
        anti_spoofing=input_args.get("anti_spoofing", False),
    )

    # 检查返回值是否为元组（表示错误）
    if isinstance(faces, tuple):
        return faces  # 这里直接返回错误信息和状态码

    # 如果不是元组，那么就是正常的结果
    # 将 NumPy 数组转换为 PNG 格式的 base64 编码
    for face in faces.get("results", []):
        if "face" in face:
            if isinstance(face["face"], np.ndarray):
                # 将 NumPy 数组转换为 PIL Image
                img = Image.fromarray(face["face"].astype('uint8'))
                
                # 创建一个字节流
                buffered = BytesIO()
                
                # 将图像保存为 PNG 格式到字节流
                img.save(buffered, format="PNG")
                
                # 获取字节流的内容并进行 base64 编码
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # 添加 data URI 前缀
                face["face"] = f"data:image/png;base64,{img_str}"

    logger.debug(faces)

    return faces
