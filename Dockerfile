FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

LABEL description="WAN T2V RP Handler - RunPod Serverless"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV CUDA_VISIBLE_DEVICES=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    unzip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /workspace


RUN git clone https://github.com/Wan-Video/Wan2.2.git


COPY rp_handler.py /workspace/Wan2.2/rp_handler.py

WORKDIR /workspace/Wan2.2
RUN pip install --upgrade pip
RUN pip install flash-attn --no-build-isolation
RUN pip install runpod librosa decord hf_transfer boto3
RUN pip install transformers peft
RUN pip install -r requirements.txt
RUN pip install --upgrade transformers

RUN mkdir -p results

CMD ["python", "-u", "rp_handler.py"]
