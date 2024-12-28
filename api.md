# DeepFace API 文档

## 基础信息

- 基础URL: `/`
- API版本: 与DeepFace库版本一致

## API 端点

### 1. 首页

- **URL:** `/`
- **方法:** `GET`
- **描述:** 返回欢迎信息和API版本
- **响应示例:**
  ```
  Welcome to DeepFace API v{version}!
  ```
- **响应类型:** `text/html`

### 2. 人脸特征提取

- **URL:** `/represent`
- **方法:** `POST`
- **描述:** 提取图像中的人脸特征向量
- **请求格式:** `multipart/form-data` 或 `application/json`
- **参数:**
  - `img`: 必需，图像数据（可以是base64编码的图像、图像URL、图像文件）
  - `model_name`: 可选，默认值 "VGG-Face"
  - `detector_backend`: 可选，默认值 "opencv"
  - `enforce_detection`: 可选，布尔值，默认值 true
  - `align`: 可选，布尔值，默认值 true
  - `anti_spoofing`: 可选，布尔值，默认值 false
  - `max_faces`: 可选，整数，默认值 1
- **响应:** JSON对象，包含提取的特征向量
  ```json
  {
    "embedding": [
      // 包含128或2622个浮点数的数组（具体长度取决于使用的模型）
      -0.0234,
      0.0825,
      // ...更多特征值
    ],
    "facial_area": {
      "x": 104,      // 人脸区域左上角x坐标
      "y": 85,       // 人脸区域左上角y坐标
      "w": 251,      // 人脸区域宽度
      "h": 251       // 人脸区域高度
    },
    "confidence": 0.99  // 检测置信度，范围0-1
  }
  ```
- **错误响应:** 
  ```json
  {
    "exception": "错误信息"
  }
  ```

### 3. 人脸验证

- **URL:** `/verify`
- **方法:** `POST`
- **描述:** 比较两张图片中的人脸是否属于同一个人
- **请求格式:** `multipart/form-data` 或 `application/json`
- **参数:**
  - `img1`: 必需，第一张图片（可以是base64编码的图像、图像URL、图像文件）
  - `img2`: 必需，第二张图片（可以是base64编码的图像、图像URL、图像文件）
  - `model_name`: 可选，默认值 "VGG-Face"
  - `detector_backend`: 可选，默认值 "opencv"
  - `distance_metric`: 可选，默认值 "cosine"
  - `align`: 可选，布尔值，默认值 true
  - `enforce_detection`: 可选，布尔值，默认值 true
  - `anti_spoofing`: 可选，布尔值，默认值 false
- **响应:** 
  ```json
  {
    "verified": true,          // 布尔值，表示是否为同一个人
    "distance": 0.2,          // 浮点数，表示两张人脸的距离（相似度）
    "threshold": 0.4,         // 浮点数，判定阈值
    "model": "VGG-Face",      // 字符串，使用的模型名称
    "detector_backend": "opencv",  // 字符串，使用的检测器
    "similarity_metric": "cosine", // 字符串，使用的距离度量方法
    "facial_areas": {         // 两张图片中检测到的人脸区域
      "img1": {
        "x": 104,
        "y": 85,
        "w": 251,
        "h": 251
      },
      "img2": {
        "x": 100,
        "y": 87,
        "w": 248,
        "h": 248
      }
    },
    "time": 2.34  // 浮点数，处理时间（秒）
  }
  ```
- **错误响应:**
  ```json
  {
    "exception": "错误信息"
  }
  ```

### 4. 人脸分析

- **URL:** `/analyze`
- **方法:** `POST`
- **描述:** 分析图像中的人脸属性（年龄、性别、情绪、种族等）
- **请求格式:** `multipart/form-data` 或 `application/json`
- **参数:**
  - `img`: 必需，图像数据（可以是base64编码的图像、图像URL、图像文件）
  - `actions`: 可选，要分析的属性列表，默认值 ["age", "gender", "emotion", "race"]
  - `detector_backend`: 可选，默认值 "opencv"
  - `enforce_detection`: 可选，布尔值，默认值 true
  - `align`: 可选，布尔值，默认值 true
  - `anti_spoofing`: 可选，布尔值，默认值 false
- **响应:** 
  ```json
  {
    "results": [
      {
        "age": 28,           // 整数，预测的年龄
        "gender": {
          "Woman": 0.99,     // 性别预测概率
          "Man": 0.01
        },
        "emotion": {         // 情绪预测概率
          "angry": 0.01,
          "disgust": 0.02,
          "fear": 0.01,
          "happy": 0.85,
          "sad": 0.03,
          "surprise": 0.03,
          "neutral": 0.05
        },
        "race": {           // 种族预测概率
          "asian": 0.92,
          "indian": 0.03,
          "black": 0.01,
          "white": 0.02,
          "middle eastern": 0.01,
          "latino hispanic": 0.01
        },
        "facial_area": {    // 人脸区域坐标
          "x": 104,
          "y": 85,
          "w": 251,
          "h": 251
        },
        "dominant_gender": "Woman",      // 字符串，主要性别
        "dominant_emotion": "happy",     // 字符串，主要情绪
        "dominant_race": "asian"         // 字符串，主要种族
      }
    ]
  }
  ```
- **错误响应:**
  ```json
  {
    "exception": "错误信息"
  }
  ```

### 5. 人脸提取

- **URL:** `/extract_faces`
- **方法:** `POST`
- **描述:** 从图像中提取人脸区域
- **请求格式:** `application/json`
- **参数:**
  - `img` 或 `img_path`: 必需，图像数据
  - `detector_backend`: 可选，默认值 "retinaface"
  - `enforce_detection`: 可选，布尔值，默认值 true
  - `align`: 可选，布尔值，默认值 true
  - `expand_percentage`: 可选，整数，默认值 0
  - `anti_spoofing`: 可选，布尔值，默认值 false
- **响应:** 
  ```json
  {
    "results": [
      {
        "face": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",  // base64编码的裁剪后人脸图像
        "facial_area": {
          "x": 104,    // 整数，人脸区域左上角x坐标
          "y": 85,     // 整数，人脸区域左上角y坐标
          "w": 251,    // 整数，人脸区域宽度
          "h": 251     // 整数，人脸区域高度
        },
        "confidence": 0.99,    // 浮点数，检测置信度（0-1）
        "landmarks": {         // 可选，人脸关键点坐标
          "right_eye": [180, 200],
          "left_eye": [250, 200],
          "nose": [215, 230],
          "mouth_right": [180, 270],
          "mouth_left": [250, 270]
        }
      }
      // 可能包含多个人脸结果
    ]
  }
  ```
- **错误响应:**
  ```json
  {
    "error": "错误信息"
  }
  ```

## 通用说明

1. 所有API都支持错误处理，当发生错误时会返回相应的错误信息和HTTP状态码（通常是400或500）
2. 图像输入支持多种格式：
   - Base64编码的图像字符串
   - 图像URL
   - 图像文件（multipart/form-data）
3. GPU加速：
   - 服务会自动检测并使用可用的GPU
   - 如果GPU不可用，会自动切换到CPU模式
4. 所有布尔类型参数都可以使用字符串 "true"/"false" 或布尔值 true/false
5. 所有API的响应都包含 Content-Type: application/json，除了首页端点返回 text/html