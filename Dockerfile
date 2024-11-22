FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

# 预先设置时区，避免交互
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装Python 3.8.12
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置Python 3.8为默认Python版本
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

LABEL org.opencontainers.image.source https://github.com/serengil/deepface

# -----------------------------------
# create required folder
RUN mkdir -p /app && chown -R 1001:0 /app
RUN mkdir /app/deepface



# -----------------------------------
# switch to application directory
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
# even though we will use local requirements, this one is required to perform install deepface from source code
COPY ./requirements.txt /app/requirements.txt
COPY ./requirements_local /app/requirements_local.txt
COPY ./package_info.json /app/
COPY ./setup.py /app/
COPY ./README.md /app/
COPY ./entrypoint.sh /app/deepface/api/src/entrypoint.sh

# 添加 CUDA 相关环境变量
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda

# 安装 CUDA 工具
RUN apt-get update && apt-get install -y \
    cuda-command-line-tools-12-4 \
    && rm -rf /var/lib/apt/lists/*

# 替换原有的 tensorflow 安装命令
RUN pip install tensorflow

# 替换 TensorRT 相关安装命令
RUN pip install nvidia-pyindex && \
    pip install nvidia-tensorrt==8.6.1 && \
    pip install tensorflow[and-cuda]

# 如果上述方案不工作，可以尝试这个替代方案：
# RUN python3 -m pip install --upgrade pip && \
#     pip install nvidia-tensorrt && \
#     pip install tensorflow-gpu

# -----------------------------------
# install deepface from pypi release (might be out-of-date)
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org deepface
# -----------------------------------
# install dependencies - deepface with these dependency versions is working
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -r /app/requirements_local.txt
# install deepface from source code (always up-to-date)
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -e .

# -----------------------------------
# some packages are optional in deepface. activate if your task depends on one.
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org cmake==3.24.1.1
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org dlib==19.20.0
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org lightgbm==2.3.1

# -----------------------------------
# environment variables
ENV PYTHONUNBUFFERED=1

# -----------------------------------
# run the app (re-configure port if necessary)
WORKDIR /app/deepface/api/src
EXPOSE 5000
# CMD ["gunicorn", "--workers=1", "--timeout=3600", "--bind=0.0.0.0:5000", "app:create_app()"]
ENTRYPOINT [ "sh", "entrypoint.sh" ]