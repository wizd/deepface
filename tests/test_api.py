# built-in dependencies
import os
import base64
import unittest
from unittest.mock import patch, MagicMock
from packaging import version
from typing import Optional, Dict, List, Any, Union

# 3rd party dependencies
import pytest
import gdown
import numpy as np
import flask
from flask import Flask
import werkzeug
from flask.testing import FlaskClient
from werkzeug.test import TestResponse

# project dependencies
from deepface.api.src.app import create_app
from deepface.api.src.modules.core import routes
from deepface.commons.logger import Logger

logger = Logger()

IMG1_SOURCE = (
    "https://raw.githubusercontent.com/serengil/deepface/refs/heads/master/tests/dataset/img1.jpg"
)
IMG2_SOURCE = (
    "https://raw.githubusercontent.com/serengil/deepface/refs/heads/master/tests/dataset/img2.jpg"
)
DUMMY_APP = Flask(__name__)


class TestVerifyEndpoint(unittest.TestCase):
    def setUp(self):
        download_test_images(IMG1_SOURCE)
        download_test_images(IMG2_SOURCE)
        app = create_app()
        app.config["DEBUG"] = True
        app.config["TESTING"] = True
        self.app = app.test_client()

    def test_tp_verify(self):
        data = {
            "img1": "dataset/img1.jpg",
            "img2": "dataset/img2.jpg",
        }
        response = self.app.post("/verify", json=data)
        assert response.status_code == 200
        result = response.json
        logger.debug(result)

        assert result.get("verified") is not None
        assert result.get("model") is not None
        assert result.get("similarity_metric") is not None
        assert result.get("detector_backend") is not None
        assert result.get("distance") is not None
        assert result.get("threshold") is not None
        assert result.get("facial_areas") is not None

        assert result.get("verified") is True

        logger.info("✅ true-positive verification api test is done")

    def test_tn_verify(self):
        data = {
            "img1": "dataset/img1.jpg",
            "img2": "dataset/img2.jpg",
        }
        response = self.app.post("/verify", json=data)
        assert response.status_code == 200
        result = response.json
        logger.debug(result)

        assert result.get("verified") is not None
        assert result.get("model") is not None
        assert result.get("similarity_metric") is not None
        assert result.get("detector_backend") is not None
        assert result.get("distance") is not None
        assert result.get("threshold") is not None
        assert result.get("facial_areas") is not None

        assert result.get("verified") is True

        logger.info("✅ true-negative verification api test is done")

    def test_represent(self):
        data = {
            "img": "dataset/img1.jpg",
        }
        response = self.app.post("/represent", json=data)
        assert response.status_code == 200
        result = response.json
        logger.debug(result)
        assert result.get("results") is not None
        assert isinstance(result["results"], list) is True
        assert len(result["results"]) > 0
        for i in result["results"]:
            assert i.get("embedding") is not None
            assert isinstance(i.get("embedding"), list) is True
            assert len(i.get("embedding")) == 4096
            assert i.get("face_confidence") is not None
            assert i.get("facial_area") is not None

        logger.info("✅ representation api test is done (for image path)")

    def test_represent_encoded(self):
        image_path = "dataset/img1.jpg"
        with open(image_path, "rb") as image_file:
            encoded_string = "data:image/jpeg;base64," + base64.b64encode(image_file.read()).decode(
                "utf8"
            )

        data = {"model_name": "Facenet", "detector_backend": "mtcnn", "img": encoded_string}

        response = self.app.post("/represent", json=data)
        assert response.status_code == 200
        result = response.json
        logger.debug(result)
        assert result.get("results") is not None
        assert isinstance(result["results"], list) is True
        assert len(result["results"]) > 0
        for i in result["results"]:
            assert i.get("embedding") is not None
            assert isinstance(i.get("embedding"), list) is True
            assert len(i.get("embedding")) == 128
            assert i.get("face_confidence") is not None
            assert i.get("facial_area") is not None

        logger.info("✅ representation api test is done (for encoded image)")

    def test_represent_url(self):
        data = {
            "model_name": "Facenet",
            "detector_backend": "mtcnn",
            "img": "https://github.com/serengil/deepface/blob/master/tests/dataset/couple.jpg?raw=true",
        }

        response = self.app.post("/represent", json=data)
        assert response.status_code == 200
        result = response.json
        logger.debug(result)
        assert result.get("results") is not None
        assert isinstance(result["results"], list) is True
        assert len(result["results"]) == 2  # 2 faces are in the image link
        for i in result["results"]:
            assert i.get("embedding") is not None
            assert isinstance(i.get("embedding"), list) is True
            assert len(i.get("embedding")) == 128
            assert i.get("face_confidence") is not None
            assert i.get("facial_area") is not None

        logger.info("✅ representation api test is done (for image url)")

    def test_analyze(self):
        data = {
            "img": "dataset/img1.jpg",
        }
        response = self.app.post("/analyze", json=data)
        assert response.status_code == 200
        result = response.json
        logger.debug(result)
        assert result.get("results") is not None
        assert isinstance(result["results"], list) is True
        assert len(result["results"]) > 0
        for i in result["results"]:
            assert i.get("age") is not None
            assert isinstance(i.get("age"), (int, float))
            assert i.get("dominant_gender") is not None
            assert i.get("dominant_gender") in ["Man", "Woman"]
            assert i.get("dominant_emotion") is not None
            assert i.get("dominant_race") is not None

        logger.info("�� analyze api test is done")

    def test_analyze_inputformats(self):
        image_path = "dataset/couple.jpg"
        with open(image_path, "rb") as image_file:
            encoded_image = "data:image/jpeg;base64," + base64.b64encode(image_file.read()).decode(
                "utf8"
            )

        image_sources = [
            # image path
            image_path,
            # image url
            f"https://github.com/serengil/deepface/blob/master/tests/{image_path}?raw=true",
            # encoded image
            encoded_image,
        ]

        results = []
        for img in image_sources:
            data = {
                "img": img,
            }
            response = self.app.post("/analyze", json=data)

            assert response.status_code == 200
            result = response.json
            results.append(result)

            assert result.get("results") is not None
            assert isinstance(result["results"], list) is True
            assert len(result["results"]) > 0
            for i in result["results"]:
                assert i.get("age") is not None
                assert isinstance(i.get("age"), (int, float))
                assert i.get("dominant_gender") is not None
                assert i.get("dominant_gender") in ["Man", "Woman"]
                assert i.get("dominant_emotion") is not None
                assert i.get("dominant_race") is not None

        assert len(results[0]["results"]) == len(results[1]["results"]) and len(
            results[0]["results"]
        ) == len(results[2]["results"])

        for i in range(len(results[0]["results"])):
            assert (
                results[0]["results"][i]["dominant_emotion"]
                == results[1]["results"][i]["dominant_emotion"]
                and results[0]["results"][i]["dominant_emotion"]
                == results[2]["results"][i]["dominant_emotion"]
            )

            assert (
                results[0]["results"][i]["dominant_gender"]
                == results[1]["results"][i]["dominant_gender"]
                and results[0]["results"][i]["dominant_gender"]
                == results[2]["results"][i]["dominant_gender"]
            )

            assert (
                results[0]["results"][i]["dominant_race"]
                == results[1]["results"][i]["dominant_race"]
                and results[0]["results"][i]["dominant_race"]
                == results[2]["results"][i]["dominant_race"]
            )

        logger.info("✅ different inputs test is done")

    def test_invalid_verify(self):
        data = {
            "img1": "dataset/invalid_1.jpg",
            "img2": "dataset/invalid_2.jpg",
        }
        response = self.app.post("/verify", json=data)
        assert response.status_code == 400
        logger.info("✅ invalid verification request api test is done")

    def test_invalid_represent(self):
        data = {
            "img": "dataset/invalid_1.jpg",
        }
        response = self.app.post("/represent", json=data)
        assert response.status_code == 400
        logger.info("✅ invalid represent request api test is done")

    def test_invalid_analyze(self):
        data = {
            "img": "dataset/invalid.jpg",
        }
        response = self.app.post("/analyze", json=data)
        assert response.status_code == 400

    def test_analyze_for_multipart_form_data(self):
        if is_form_data_file_testable() is False:
            return

        with open("/tmp/img1.jpg", "rb") as img_file:
            response = self.app.post(
                "/analyze",
                content_type="multipart/form-data",
                data={
                    "img": (img_file, "test_image.jpg"),
                    "actions": '["age", "gender"]',
                    "detector_backend": "mtcnn",
                },
            )
            assert response.status_code == 200
            result = response.json
            assert isinstance(result, dict)
            assert result.get("age") is not True
            assert result.get("dominant_gender") is not True
            logger.info("✅ analyze api for multipart form data test is done")

    def test_verify_for_multipart_form_data(self):
        if is_form_data_file_testable() is False:
            return

        with open("/tmp/img1.jpg", "rb") as img1_file:
            with open("/tmp/img2.jpg", "rb") as img2_file:
                response = self.app.post(
                    "/verify",
                    content_type="multipart/form-data",
                    data={
                        "img1": (img1_file, "first_image.jpg"),
                        "img2": (img2_file, "second_image.jpg"),
                        "model_name": "Facenet",
                        "detector_backend": "mtcnn",
                        "distance_metric": "euclidean",
                    },
                )
                assert response.status_code == 200
                result = response.json
                assert isinstance(result, dict)
                assert result.get("verified") is not None
                assert result.get("model") == "Facenet"
                assert result.get("similarity_metric") is not None
                assert result.get("detector_backend") == "mtcnn"
                assert result.get("threshold") is not None
                assert result.get("facial_areas") is not None

                logger.info("✅ verify api for multipart form data test is done")

    def test_represent_for_multipart_form_data(self):
        if is_form_data_file_testable() is False:
            return

        with open("/tmp/img1.jpg", "rb") as img_file:
            response = self.app.post(
                "/represent",
                content_type="multipart/form-data",
                data={
                    "img": (img_file, "first_image.jpg"),
                    "model_name": "Facenet",
                    "detector_backend": "mtcnn",
                },
            )
            assert response.status_code == 200
            result = response.json
            assert isinstance(result, dict)
            logger.info("✅ represent api for multipart form data test is done")

    def test_represent_for_multipart_form_data_and_filepath(self):
        if is_form_data_file_testable() is False:
            return

        response = self.app.post(
            "/represent",
            content_type="multipart/form-data",
            data={
                "img": "/tmp/img1.jpg",
                "model_name": "Facenet",
                "detector_backend": "mtcnn",
            },
        )
        assert response.status_code == 200
        result = response.json
        assert isinstance(result, dict)
        logger.info("✅ represent api for multipart form data and file path test is done")

    def test_extract_image_from_form_data(self):
        if is_form_data_file_testable() is False:
            return

        img_key = "img1"
        img_itself = np.zeros((100, 100, 3), dtype=np.uint8)
        # Establish a temporary request context using the Flask app
        with DUMMY_APP.test_request_context("/dummy_endpoint"):
            # Mock the file part
            with patch("deepface.api.src.modules.core.routes.request") as mock_request:
                mock_file = MagicMock()
                mock_file.filename = "image.jpg"
                mock_request.files = {img_key: mock_file}

                # Mock the image loading function
                with patch(
                    "deepface.commons.image_utils.load_image_from_file_storage",
                    return_value=img_itself,
                ):
                    result = routes.extract_image_from_request(img_key)

                    assert isinstance(result, np.ndarray)
                    assert np.array_equal(result, img_itself)

        logger.info("✅ test extract_image_from_request for real image from form data done")

    def test_extract_image_string_from_json_data(self):
        if is_form_data_file_testable() is False:
            return

        img_key = "img1"
        img_data = "image_url_or_path_or_base64"

        with DUMMY_APP.test_request_context("/dummy_endpoint"):
            with patch("deepface.api.src.modules.core.routes.request") as mock_request:
                # Mock JSON data
                mock_request.files = None
                mock_request.is_json = True
                mock_request.get_json = MagicMock(return_value={img_key: img_data})

                result = routes.extract_image_from_request(img_key)

                assert isinstance(result, str)
                assert result == img_data

        logger.info("✅ test extract_image_from_request for image string from json done")

    def test_extract_image_string_from_form_data(self):
        if is_form_data_file_testable() is False:
            return

        img_key = "img1"
        img_data = "image_url_or_path_or_base64"

        with DUMMY_APP.test_request_context("/dummy_endpoint"):
            with patch("deepface.api.src.modules.core.routes.request") as mock_request:
                # Mock form data
                mock_request.files = None

                mock_request.is_json = False
                mock_request.get_json = MagicMock(return_value=None)

                mock_request.form = MagicMock()
                mock_request.form.to_dict.return_value = {img_key: img_data}

                result = routes.extract_image_from_request(img_key)

                assert isinstance(result, str)
                assert result == img_data

        logger.info("✅ test extract_image_from_request for image string from form done")

    def test_landmark(self):
        """测试landmark API的基本功能"""
        # 确保测试图片存在
        download_test_images(IMG1_SOURCE)
        
        data = {
            "img": "tests/dataset/img1.jpg",
        }
        response: TestResponse = self.app.post("/landmark", json=data)
        assert response.status_code == 200
        
        result: Optional[Dict[str, Any]] = response.get_json()
        if result is None:
            pytest.fail("Response did not return valid JSON")
            
        logger.debug(result)
        
        # 类型安全的检查
        if not isinstance(result, dict):
            pytest.fail("Response is not a dictionary")
            
        results: Optional[List[Dict[str, Any]]] = result.get("results")
        if not results:
            pytest.fail("No results found in response")
            
        if not isinstance(results, list):
            pytest.fail("Results is not a list")
            
        if not results:
            pytest.fail("Results list is empty")
            
        for face_result in results:
            if not isinstance(face_result, dict):
                pytest.fail("Face result is not a dictionary")
                
            # 检查面部区域
            if "facial_area" not in face_result:
                pytest.fail("facial_area not found in face result")
                
            facial_area = face_result["facial_area"]
            if not all(key in facial_area for key in ["x", "y", "w", "h"]):
                pytest.fail("Missing required keys in facial_area")
            
            # 检查关键点
            if "landmarks" not in face_result:
                pytest.fail("landmarks not found in face result")
                
            landmarks = face_result["landmarks"]
            if not isinstance(landmarks, list) or not landmarks:
                pytest.fail("landmarks is not a valid list")
                
            for landmark in landmarks:
                if not all(key in landmark for key in ["x", "y", "z", "point_number"]):
                    pytest.fail("Missing required keys in landmark")
            
            # 检查面部特征区域
            if "facial_features" not in face_result:
                pytest.fail("facial_features not found in face result")
                
            facial_features = face_result["facial_features"]
            required_features = [
                "jaw_line", "left_eyebrow", "right_eyebrow",
                "nose_bridge", "left_eye", "right_eye", "outer_lip"
            ]
            if not all(key in facial_features for key in required_features):
                pytest.fail("Missing required facial features")
            
            # 检查���部表情系数(可选)
            if "face_blendshapes" in face_result:
                blendshapes = face_result["face_blendshapes"]
                if not isinstance(blendshapes, list):
                    pytest.fail("face_blendshapes is not a list")
                    
                for shape in blendshapes:
                    if not isinstance(shape, dict):
                        pytest.fail("blendshape is not a dictionary")
                    if not all(key in shape for key in ["category_name", "score"]):
                        pytest.fail("Missing required keys in blendshape")

        logger.info("✅ landmark api test is done")

    def test_landmark_with_base64(self):
        """测试使用base64编码图片的landmark API"""
        try:
            # 确保测试图片存在
            download_test_images(IMG1_SOURCE)
            image_path = "/tmp/img1.jpg"  # 使用绝对路径
            
            with open(image_path, "rb") as image_file:
                encoded_string = "data:image/jpeg;base64," + base64.b64encode(image_file.read()).decode("utf8")

            data = {
                "img": encoded_string,
                "detector_backend": "mediapipe"
            }

            response: TestResponse = self.app.post("/landmark", json=data)
            assert response.status_code == 200
            
            result: Optional[Dict[str, Any]] = response.get_json()
            if result is None:
                pytest.fail("Response did not return valid JSON")
                
            results = result.get("results")
            if not results:
                pytest.fail("No results found in response")
                
            if not isinstance(results, list):
                pytest.fail("Results is not a list")
                
            if not results:
                pytest.fail("Results list is empty")

            logger.info("✅ landmark api test with base64 image is done")
        except Exception as e:
            pytest.fail(f"Test failed with error: {str(e)}")

    def test_invalid_landmark(self):
        """测试无效输入的landmark API"""
        data = {
            "img": "/tmp/invalid.jpg",  # 使用不存在的文件路径
        }
        response = self.app.post("/landmark", json=data)
        assert response.status_code == 400
        logger.info("✅ invalid landmark request api test is done")


def download_test_images(url: str):
    file_name = url.split("/")[-1]
    target_file = f"/tmp/{file_name}"
    if os.path.exists(target_file) is True:
        return

    gdown.download(url, target_file, quiet=False)


def is_form_data_file_testable() -> bool:
    """
    Sending a file from form data fails in unit test with
        415 unsupported media type error for flask 3.X
        but it is working for flask 2.0.2
    Returns:
        is_form_data_file_testable (bool)
    """
    try:
        flask_version = version.parse(flask.__version__)
        werkzeug_version = version.parse(getattr(werkzeug, '__version__', '0.0.0'))
        threshold_version = version.parse("2.0.2")
        is_testable = flask_version <= threshold_version and werkzeug_version <= threshold_version
        if not is_testable:
            logger.warn(
                "sending file in form data is not testable because of flask, werkzeug versions. "
                f"Expected <= {threshold_version}, but flask={flask_version} and werkzeug={werkzeug_version}."
            )
        return is_testable
    except Exception as e:
        logger.error(f"Error checking form data testability: {str(e)}")
        return False
