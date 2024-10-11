from flask import Blueprint, request
from deepface import DeepFace
from deepface.api.src.modules.core import service
from deepface.commons.logger import Logger
import numpy as np

logger = Logger()

blueprint = Blueprint("routes", __name__)


@blueprint.route("/")
def home():
    return f"<h1>Welcome to DeepFace API v{DeepFace.__version__}!</h1>"


@blueprint.route("/represent", methods=["POST"])
def represent():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

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
def verify():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img1_path = input_args.get("img1") or input_args.get("img1_path")
    img2_path = input_args.get("img2") or input_args.get("img2_path")

    if img1_path is None:
        return {"message": "you must pass img1_path input"}

    if img2_path is None:
        return {"message": "you must pass img2_path input"}

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
def analyze():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

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
def extract_faces():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "空输入集合"}

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "必须传入img_path参数"}

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
