# tensorflow를 이용한 Object Detection 
파이썬 streamlit 프레임워크를 이용해서 Object Detection을 수행하는 웹

1. 먼저 디렉토리를 만들어준다. 이름은 원하는 걸로... 그리고 이동
```
mkdir tensorflow_app
cd tensorflow_app
```

2. 그리고 현재 깃을 클론해준다   
```
git clone https://github.com/terrificmn/tensorflow-od.git
```

3. 그 이후에 깃 클론된 디렉토리명이 tensorflow_od 인데 src로 바꿔준다  
```
mv tensorflow_od src
```

4. 그리고 Dockerfile, docker-compose.yaml 파일 등을 사용하기 파일들을 복사해준다    
상위 경로인 처음 만든 tensorflow_app 디렉토리로 복사. 현재 변경된 src로 이동 후에 복사해준다  
```
cd src
cp Dockerfile docker-compose.yml requirements.txt ../
```

5. 다시 상위 디렉토리(프로젝트 ROOT)로 이동 후에 build 및 up 진행
```
cd ~/tensorflow_app
docker-compose build
... 완료 후
docker-compose up
```

> 처음에 docker관련 파일을 안 올리고(따로 관리하려다..) 시작했다가 결국은 이런 형태가;;   
고쳐야하는데 그냥 쓰려고 함, 쓸사람도 없고ㅋㅋ, 하는 법만 기억날 수 있게 정리함   

6. 이제 중요한 .gitignore에서 빠져있는 디렉토리를 만들고 파일들을 따로 복사해주면 된다  

```
├── data
│   ├── images
│   └── videos
├── enet_data
│   └── enet-cityscapes
├── models
│   ├── community
│   ├── official
│   ├── orbit
│   └── research
└── yolo_model
```

위 처럼 디렉토리 구조가 되어 있어야 한다    
- data 디렉토리 안에는 coco_classes.txt 파일과, yolo.h5 있어야 함   
- yolo_model 디렉토리 안에는 yolo_model.py, darknet53.py 가 있어야 함  











