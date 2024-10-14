# 基础镜像
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
LABEL org.opencontainers.image.source https://github.com/serengil/deepface

# 创建所需文件夹
RUN mkdir /app
RUN mkdir /app/deepface

# 切换到应用目录
WORKDIR /app

# 更新镜像操作系统
RUN apt-get update && apt-get install -y curl software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.10=3.10.15* python3.10-distutils python3.10-dev

# 安装OpenCV所需的依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# 安装pip for Python 3.10
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.10 get-pip.py
RUN rm get-pip.py

# 设置Python 3.10为默认版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --set python3 /usr/bin/python3.10

# 复制所需文件
COPY ./deepface /app/deepface
COPY ./requirements.txt /app/requirements.txt
COPY ./requirements_local /app/requirements_local.txt
COPY ./package_info.json /app/
COPY ./setup.py /app/
COPY ./README.md /app/

# 安装CUDA和cuDNN相关依赖
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org nvidia-cublas-cu11 nvidia-cudnn-cu11

# 安装TensorFlow GPU版本
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org tensorflow

# 安装PyTorch
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org torch==2.1.2

# 安装依赖
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -r /app/requirements_local.txt

# 从源代码安装deepface
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -e .

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib

# 运行应用
WORKDIR /app/deepface/api/src
EXPOSE 5000
CMD ["gunicorn", "--workers=1", "--timeout=3600", "--bind=0.0.0.0:5000", "app:create_app()"]
