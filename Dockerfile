# 基础镜像
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
LABEL org.opencontainers.image.source https://github.com/serengil/deepface

# -----------------------------------
# create required folder
RUN mkdir -p /app && chown -R 1001:0 /app
RUN mkdir /app/deepface



# 切换到应用目录
WORKDIR /app

# -----------------------------------
# update image os
# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------
# Copy required files from repo into image
COPY ./deepface /app/deepface
COPY ./requirements.txt /app/requirements.txt
COPY ./requirements_local /app/requirements_local.txt
COPY ./package_info.json /app/
COPY ./setup.py /app/
COPY ./README.md /app/
COPY ./entrypoint.sh /app/deepface/api/src/entrypoint.sh

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
# CMD ["gunicorn", "--workers=1", "--timeout=3600", "--bind=0.0.0.0:5000", "app:create_app()"]
ENTRYPOINT [ "sh", "entrypoint.sh" ]
