# base image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
LABEL org.opencontainers.image.source https://github.com/serengil/deepface

# -----------------------------------
# create required folder
RUN mkdir /app
RUN mkdir /app/deepface

# -----------------------------------
# switch to application directory
WORKDIR /app

# 设置时区环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# -----------------------------------
# update image os
RUN apt-get update && apt-get install -y \
    curl \
    software-properties-common \
    cmake \
    build-essential \
    tzdata

# 安装OpenCV所需的依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# 安装Python 3.10
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.10 python3.10-distutils

# 安装pip for Python 3.10
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.10 get-pip.py
RUN rm get-pip.py

# 设置Python 3.10为默认版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --set python3 /usr/bin/python3.10

# 安装CUDA工具包
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-toolkit-12-4

# 设置CUDA环境变量
ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# -----------------------------------
# Copy required files from repo into image
COPY ./deepface /app/deepface
# even though we will use local requirements, this one is required to perform install deepface from source code
COPY ./requirements.txt /app/requirements.txt
COPY ./requirements_local /app/requirements_local.txt
COPY ./requirements_additional.txt /app/requirements_additional.txt
COPY ./package_info.json /app/
COPY ./setup.py /app/
COPY ./README.md /app/

# -----------------------------------
# if you plan to use a GPU, you should install the 'tensorflow-gpu' package
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org tensorflow-gpu==2.9.0

# if you plan to use face anti-spoofing, then activate this line
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org torch==2.1.2
# -----------------------------------
# install deepface from pypi release (might be out-of-date)
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org deepface
# -----------------------------------
# install dependencies - deepface with these dependency versions is working
RUN pip install --no-deps --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -r /app/requirements_local.txt
# install deepface from source code (always up-to-date)
RUN pip install --ignore-installed --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -e .

# 安装 dlib 所需的依赖
RUN apt-get update && apt-get install -y \
    libx11-dev \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    python3-dev \
    cmake

# 安装 pkg-config
RUN apt-get install -y pkg-config

# 安装 dlib
RUN pip install dlib tf-keras

RUN pip install --no-deps --ignore-installed --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -r /app/requirements_additional.txt

# -----------------------------------
# some packages are optional in deepface. activate if your task depends on one.
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org cmake==3.24.1.1
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org dlib==19.20.0
# RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org lightgbm==2.3.1

# -----------------------------------
# environment variables
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# -----------------------------------
# run the app (re-configure port if necessary)
WORKDIR /app/deepface/api/src
EXPOSE 5000
CMD ["gunicorn", "--workers=1", "--timeout=3600", "--bind=0.0.0.0:5000", "app:create_app()"]
