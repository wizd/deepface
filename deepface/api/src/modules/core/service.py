# built-in dependencies
import traceback
from typing import Optional, Union

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()


# pylint: disable=broad-except


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
    max_faces: Optional[int] = None,
):
    try:
        result = {}
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
        result["results"] = embedding_objs
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while representing: {str(err)} - {tb_str}"}, 400


def verify(
    img1_path: Union[str, np.ndarray],
    img2_path: Union[str, np.ndarray],
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        obj = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
        )
        return obj
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while verifying: {str(err)} - {tb_str}"}, 400


def analyze(
    img_path: Union[str, np.ndarray],
    actions: list,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
    try:
        result = {}
        demographies = DeepFace.analyze(
            img_path=img_path,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            silent=True,
            anti_spoofing=anti_spoofing,
        )
        result["results"] = demographies
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400


def extract_faces(
    img_path: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    expand_percentage: int,
    anti_spoofing: bool,
):
    try:
        faces = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
            anti_spoofing=anti_spoofing,
        )
        # 将 NumPy 数组转换为可 JSON 序列化的格式
        for face in faces:
            if "face" in face:
                face["face"] = face["face"].tolist()
        return {"results": faces}
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"提取人脸时发生异常: {str(err)} - {tb_str}"}, 400


def landmark(
    img_path: Union[str, np.ndarray],
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import cv2
        from deepface.commons import image_utils
        import numpy as np
        import os
        
        # 获取模型文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
        model_path = os.path.join(project_root, "deepface", "weights", "face_landmarker.task")
        
        # 检查模型文件是否存在
        if not os.path.isfile(model_path):
            raise ValueError(f"找不到面部关键点检测模型文件。请确保文件 {model_path} 存在")
        
        # 创建 FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        
        # 读取和预处理图像
        if isinstance(img_path, str):
            if img_path.startswith("data:image"):  # base64 image
                img = image_utils.load_image(img_path)
                if isinstance(img, tuple):
                    img = img[0]  # 获取图像数组部分
            else:  # image path
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"无法读取图像文件: {img_path}")
        elif isinstance(img_path, np.ndarray):
            img = img_path.copy()
        else:
            raise ValueError("不支持的图像格式")
            
        # 转换为RGB格式并创建MediaPipe Image对象
        if not isinstance(img, np.ndarray):
            raise ValueError("图像必须是numpy数组格式")
            
        if len(img.shape) != 3:
            raise ValueError("图像必须是3通道的")
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # 检测面部关键点
        detection_result = detector.detect(mp_image)
        
        if not detection_result.face_landmarks and enforce_detection:
            raise ValueError("未检测到人脸")
            
        result = {"results": []}
        
        if detection_result.face_landmarks:
            height, width = img.shape[:2]
            
            for face_landmarks in detection_result.face_landmarks:
                # 转换关键点坐标为像素坐标
                landmarks = []
                for idx, landmark in enumerate(face_landmarks):
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    z = landmark.z  # 保留z坐标的相对深度信息
                    landmarks.append({
                        "x": x,
                        "y": y,
                        "z": z,
                        "point_number": idx + 1
                    })
                
                # 计算人脸区域
                x_coordinates = [l.x * width for l in face_landmarks]
                y_coordinates = [l.y * height for l in face_landmarks]
                
                facial_area = {
                    "x": int(min(x_coordinates)),
                    "y": int(min(y_coordinates)),
                    "w": int(max(x_coordinates) - min(x_coordinates)),
                    "h": int(max(y_coordinates) - min(y_coordinates))
                }
                
                # 定义面部特征区域
                # MediaPipe提供了478个关键点，我们需要根据这些点的索引来定义不同的面部特征
                facial_features = {
                    "jaw_line": landmarks[0:17],  # 下巴轮廓
                    "left_eyebrow": landmarks[17:22],  # 左眉毛
                    "right_eyebrow": landmarks[22:27],  # 右眉毛
                    "nose_bridge": landmarks[27:31],  # 鼻梁
                    "left_eye": landmarks[36:42],  # 左眼
                    "right_eye": landmarks[42:48],  # 右眼
                    "outer_lip": landmarks[48:60]  # 外唇
                }
                
                # 添加面部表情系数(如果有)
                face_blendshapes = None
                if detection_result.face_blendshapes:
                    face_blendshapes = [
                        {
                            "category_name": category.category_name,
                            "score": float(category.score)
                        }
                        for category in detection_result.face_blendshapes[0]
                    ]
                
                result["results"].append({
                    "facial_area": facial_area,
                    "landmarks": landmarks,
                    "facial_features": facial_features,
                    "face_blendshapes": face_blendshapes
                })
        
        return result
        
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"面部关键点检测时发生异常: {str(err)} - {tb_str}"}, 400
