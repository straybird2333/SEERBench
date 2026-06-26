# 本地开源模型评测 - Docker 镜像（带 CUDA 的 PyTorch 基础镜像）
# 构建与运行见 README「Docker 运行」一节

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN pip install --no-cache-dir \
    "transformers>=4.45.0" \
    "accelerate>=0.20.0" \
    Pillow PyYAML

WORKDIR /workspace

CMD ["/bin/bash"]
