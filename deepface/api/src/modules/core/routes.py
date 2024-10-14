from flask import Blueprint, request
from deepface import DeepFace
from deepface.api.src.modules.core import service
from deepface.commons.logger import Logger
import numpy as np
from functools import wraps
import traceback

logger = Logger()

blueprint = Blueprint("routes", __name__)

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


@blueprint.route("/represent", methods=["POST"])
@error_handler
def represent():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}, 400

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}, 400

    obj = service.represent(
        img_path=img_path,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=input_args.get("enforce_detection", True),
        align=input_args.get("align", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
        max_faces=input_args.get("max_faces"),
    )

    logger.debug(obj)

    return obj


@blueprint.route("/verify", methods=["POST"])
@error_handler
def verify():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}, 400

    img1_path = input_args.get("img1") or input_args.get("img1_path")
    img2_path = input_args.get("img2") or input_args.get("img2_path")

    if img1_path is None:
        return {"message": "you must pass img1_path input"}, 400

    if img2_path is None:
        return {"message": "you must pass img2_path input"}, 400

    verification = service.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        distance_metric=input_args.get("distance_metric", "cosine"),
        align=input_args.get("align", True),
        enforce_detection=input_args.get("enforce_detection", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
    )

    logger.debug(verification)

    return verification


@blueprint.route("/analyze", methods=["POST"])
@error_handler
def analyze():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}, 400

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}, 400

    demographies = service.analyze(
        img_path=img_path,
        actions=input_args.get("actions", ["age", "gender", "emotion", "race"]),
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=input_args.get("enforce_detection", True),
        align=input_args.get("align", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
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
    # 将 NumPy 数组转换为可 JSON 序列化的格式
    for face in faces.get("results", []):
        if "face" in face:
            if isinstance(face["face"], np.ndarray):
                face["face"] = face["face"].tolist()
            elif not isinstance(face["face"], list):
                face["face"] = list(face["face"])

    logger.debug(faces)

    return faces
