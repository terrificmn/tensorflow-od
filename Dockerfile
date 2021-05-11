FROM ubuntu:18.04
FROM python:3.8.8

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    python3-opencv libgl1-mesa-glx 
# python3-opencv libgl1-mesa-glx 는 에러방지

# 여기는 테스트: h264 되게 하는건데 소용없음
#RUN apt-get install -y x264 libx264-dev && \
#    apt-get install -y ffmpeg 
# 여기는 테스트

#### 잘됨

# 컨테이너에 만들어준다 (디렉토리) 
RUN mkdir -p /tensorflow/models 
COPY /src/models/ /tensorflow/models

# Compile protobuf configs
RUN (cd /tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
# 컨테이너 안의 디렉토리
WORKDIR /tensorflow/models/research/

RUN export PYTHONPATH=$PYTHONPATH:/tensorflow/models/research:/tensorflow/models/research/slim

# workdir research 디렉토리로 setup.py 파일 복사 (현재 workdir경로)
RUN cp object_detection/packages/tf2/setup.py .
RUN python -m pip install .

### 잘됨


WORKDIR /src
COPY requirements.txt ./requirements.txt 
RUN pip3 install -r requirements.txt


# # 파일들을 복사 (requirements에 있는 파일들)
COPY . .  

EXPOSE 8501
CMD streamlit run app.py --server.maxUploadSize=10
