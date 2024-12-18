FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch>=2.1 \
    torchvision>=0.16 \
    numpy>=1.24 \
    scipy>=1.11 \
    open3d>=0.17 \
    nuscenes-devkit>=1.1 \
    einops>=0.7 \
    wandb>=0.16 \
    tqdm>=4.66 \
    PyYAML>=6.0 \
    matplotlib>=3.8

COPY . .
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/workspace:$PYTHONPATH

CMD ["bash"]
