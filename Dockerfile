FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_VISIBLE_DEVICES=0
RUN apt-get update
RUN apt-get install 
RUN pip install matplotlib==3.0.0 numpy==1.16.1 Pillow==6.2.0 protobuf==3.8.0 scipy==1.1.0 tqdm==4.25.0 easydict==1.9 onnx==1.5.0 torchvision==0.10.0 thop==0.0.31.post2001170342
COPY . STDC-Seg
RUN rm /workspace/STDC-Seg/data/leftImg8bit /workspace/STDC-Seg/data/gtFine; ln -s /workspace/data/leftImg8bit /workspace/STDC-Seg/data/leftImg8bit ; ln -s /workspace/data/gtFine /workspace/STDC-Seg/data/gtFine 
CMD /bin/bash

# model_maxmIOU101.pth -> without_aug.pth
# model_maxmIOU111.pth -> with_aug.pth