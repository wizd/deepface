import os
import requests

def download_landmarks_model():
    """下载dlib的面部关键点检测模型文件"""
    
    # 创建权重目录
    weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deepface", "weights")
    os.makedirs(weights_path, exist_ok=True)
    
    # 模型文件路径
    model_path = os.path.join(weights_path, "shape_predictor_68_face_landmarks.dat")
    
    # 如果文件已存在,跳过下载
    if os.path.isfile(model_path):
        print("模型文件已存在:", model_path)
        return
        
    print("开始下载面部关键点检测模型...")
    
    # 从GitHub下载模型文件
    url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    
    try:
        # 下载压缩文件
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        compressed_path = model_path + ".bz2"
        with open(compressed_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        # 解压文件
        import bz2
        with bz2.BZ2File(compressed_path) as fr, open(model_path, "wb") as fw:
            fw.write(fr.read())
            
        # 删除压缩文件
        os.remove(compressed_path)
        
        print("模型文件下载成功:", model_path)
    except Exception as e:
        print("下载失败:", str(e))
        if os.path.exists(model_path):
            os.remove(model_path)
        raise

if __name__ == "__main__":
    download_landmarks_model() 