# docker적용할 때 깃허브로 tensorflow/model를 받고 
# 소스파일을 컨테이너로 복사시켜서 해야지 실행이 됨
# 로컬에서 protoc 컴파일 후 api설치까지 해도.. 컨테이너에서 인식을 못하는거 같음
# 도커 컨테이너 안에 적용을 시킨 후 (protoc등..api설치까지)

import streamlit as st
import pathlib
import cv2
import os
from PIL import Image
from datetime import datetime
import time

from object_detection.utils import label_map_util

from tensorflow_od import load_model, show_inference
from image_func import load_image, save_uploaded_file 
from video_func import save_uploaded_video
from yolo import get_classes, detect_image
from saveCap import reCaptureVideo

from semanticSeg import makeSegmentation
from scaledown import imageResize
from fake import fakeShow
from ec2_warning import warningPrint

st.set_page_config(page_title='ml', page_icon=None, layout='centered', initial_sidebar_state='auto')


def main() :


    selectboxList = ['메뉴를 선택하세요', 'Tensorflow-object-detection', 'Video Object Detection',
                        'YOLO', 'SSD', 'Semantic Segmentation', 'aboutMe']
    selectbox = st.sidebar.selectbox("선택하세요", selectboxList)
    st.title('안녕하세요')
    st.title('메뉴를 선택하세요')    
    
    if selectbox == 'test' :
        # 테스트 입니다.
        pass    

    elif selectbox == 'Tensorflow-object-detection' :
        st.write('## Tensorflow Models Object Detection')
        imageDir = 'data/images/show'
        st.image(imageDir + '/' + 'models-hero.svg')

        st.write('')
        st.write('Object detection은 물체 자체 무엇인지 그리고 물체의 위치도 같이 찾을 수 있는 것을 말하는데 \
                    Tensorflow Model Zoo에서는 미리 학습이 된 모델을 제공을 하는데 \
                    대표적으로 CenterNet, EfficientDet, MobileNet, R-CNN, ExtremeNet 등이 있다')

        st.write('이렇게 딥러닝으로 이미 학습이 되어 있는 텐서 플로우 모델을 사용해서') 
        st.write('사진 속의 물체를 식별할 수가 있습니다.')
        st.write('물체가 어떤 것인지 분류를 하면서 ')
        st.image(imageDir + '/' + 'dog-640.jpg')
        st.write('예를 들어 사진 속 물체가 개 인지 아니면 사람인지 분류를 하게 됩니다.')
        st.write('Image Classification 이라고 합니다.')

        st.write('')

        st.image(imageDir + '/' + 'children-640.jpg')
        st.write('동시에 어느 위치에 있는지까지 (박스를 만듬) 탑색 합니다.')
        st.write('Object Localization 입니다.')
        st.write('')
        st.write('이제 이 2개의 기능을 동시에 할 수 있습니다.')

        st.image(imageDir + '/' + 'boxed.jpg')
        st.text('<실제 분석 완료된 사진>')
        st.write('강아지와 사람을 정확히 분별했네요. 심지어 의자까지도 탐지합니다.')
        st.write('')
        
        st.write('자~ 이제 사진 파일을 업로드 하세요! AI가 분석해드립니다^^')
        upload_img_list = st.file_uploader('이미지 파일 업로드', type=['png', 'jpeg', 'jpg'], accept_multiple_files=False)
    
        ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
        ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
        ############ 여기서부터 주석 시작 #######
        if upload_img_list is not None:
            img = load_image(upload_img_list)
            
            #한장만 저장
            if st.button('저장 및 분석하기') :
                directory = 'data/images/user-upload'
                filename = save_uploaded_file(directory, img)
                
                PATH_TO_TEST_IMAGES_DIR = pathlib.Path('data/images/user-upload')
                TEST_IMAGE_PATHS = pathlib.Path(PATH_TO_TEST_IMAGES_DIR, filename)
                print(TEST_IMAGE_PATHS)
                #TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))# 여러장일 때사용

                # # # 모델 불러오기 , 함수호출
                model_name = 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8'
                model_date = '20200711'
                print('model: {}'.format(model_name))

                print('start to load model...')
                detection_model = load_model(model_name, model_date)
                # #print(detection_model.signatures['serving_default'].output_dtypes)
                # #print(detection_model.signatures['serving_default'].output_shapes)

                show_inference(detection_model, TEST_IMAGE_PATHS)
                # for image_path in TEST_IMAGE_PATHS:  #여러장 지원할 때, 현재 1장만 지원함 
                #     show_inference(detection_model, image_path)
            ############여기까지 실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
            ############여기까지 실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
            ############ 여기서까지 주석 #######
            
        
        ##### cpu문제로 이미지 보여주기용 입니다. #########
        ##### cpu문제로 이미지 보여주기용 입니다. #########
        #warningPrint()
        
        video_file = open('data/videos/show/complete_ftod_image.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

        # 미리 작업된 사진 고르기 
        st.write(' ')
        st.info('이미지를 선택해서 탐색 결과를 볼 수도 있습니다. (사전에 object dectection이 완료된 사진')
        
        st.write('사진을 선택해주세요')

        radioSelection = st.radio('샘플사진을 선택하세요', ['pic1', 'pic2', 'pic3' ])
        
        if radioSelection == 'pic1':
            imgName = 'park-people-1280.jpg'
            resizedImg = imageResize(radioSelection, imgName, 0.2, 0.2)
            st.image(resizedImg)

        elif radioSelection == 'pic2':
            imgName = 'students-640.jpg'
            resizedImg = imageResize(radioSelection, imgName)
            st.image(resizedImg)

        elif radioSelection == 'pic3':
            imgName = 'elder-1920.jpg'
            resizedImg = imageResize(radioSelection, imgName, 0.15, 0.15)
            st.image(resizedImg)

        if st.button('선택한 이미지 디텍션 하기') :
            fakeShow(imgName)
            st.text('<CPU의 한계로 인해.. 미리 작업이 완료된 사진 입니다.>')
        ##### cpu문제로 이미지 보여주기용 입니다. #########
        ##### cpu문제로 이미지 보여주기용 입니다. #########


    elif selectbox == 'Video Object Detection' :
        
        st.title('Tensorflow Model Object Detection')
        
        st.write('텐서 플로우의 모델을 사용해서 물체를 탐지하는데 \
                    이를 동영상에 적용할 수가 있습니다. \
                    실제로는 동영상도 사진들이 연속적으로 보여지는 것이므로 \
                    Tensorflow Model Object Detection을 \
                    동영상에도 적용해 볼 수가 있습니다.')

        st.write('')
        
        warningPrint()
        
        video_file = open('data/videos/show/complete_ftod_video_output.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        

        ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
        ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
        # if st.button('물체 탐색 시작하기') :
        #     #20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz
        #     #model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
            
        #     model_name = 'centernet_resnet50_v2_512x512_kpts_coco17_tpu-8'
        #     model_date = '20200711'
        #     #http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz
        #     #model_name = 'faster_rcnn_resnet152_v1_640x640_coco17_tpu-8'
        #     #model_date = '20200711'
        #     #model_name = 'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8'
        #     detection_model = load_model(model_name, model_date)
        #     print(detection_model.signatures['serving_default'].output_dtypes)
        #     print(detection_model.signatures['serving_default'].output_shapes)


        #     # 카메라의 영상 실행
        #     #cap = cv2.VideoCapture(0) # 캠 영상 실행
        #     cap = cv2.VideoCapture('data/videos/road_dog_bike_for_tfod_video.mp4')

        #     if cap.isOpened() == False:
        #         print("error occured to start to play a video")

        #     else:
                
        #         #######
        #         frame_width = int(cap.get(3))
        #         frame_height = int(cap.get(4))

        #         # #이미지 사이즈 줄이기
        #         # if int(frame_width / 2) % 2 ==0 : #짝수 
        #         #     frame_width = int(frame_width / 2)
        #         # else:
        #         #     frame_width = int(frame_width / 2) + 1 #홀수가 안되게 만들어 줌

        #         # if int(frame_height / 2) % 2 == 0 :
        #         #     frame_height = int(frame_height / 2)
        #         # else:
        #         #     frame_height = int(frame_height / 2 ) + 1

        #         #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # h264는 에러발생
        #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4v 성공
        #         #fourcc = cv2.VideoWriter_fourcc(*'h264')  # x264 , h264 실패
        #         out = cv2.VideoWriter('data/videos/output.mp4', 
        #                                 fourcc,
        #                                 10, 
        #                                 ( frame_width, frame_height) )
        #         ## 저장하는 코드 write()메소드 부분을 주석 해제할 것.. 아래코드
        #         #######
        #         totalTime = 0
        #         while cap.isOpened():
        #             ret, frame = cap.read() #동영상의 사진을 하나씩 frame에 넣어준다
        #             if ret == True:
        #                 #cv2.imshow('Frame', frame)
        #                 startTime = time.time()

        #                 isImage = False # 비디오로 만들 것이기 때문에 False 를 준다. (기본으로 True)
        #                 img = show_inference(detection_model, frame, isImage)  # 이미지 경로는 필요없고 이미 np array로 받아왔기때문에 frame넘겨주면 됨
        #                 #show_inference(detection_model, frame)
        #                 endTime = time.time()
        #                 # 처리 시간 출력
        #                 precessTime = endTime-startTime
        #                 print(precessTime)
        #                 totalTime += precessTime
        #                 #save resized image 
        #                 out.write(img)

        #                 if cv2.waitKey(25) & 0xFF == 27:
        #                     break
        #             else:
        #                 break

                
        #         cap.release()
        #         print('complete')
        #         print('total time: {}'.format(totalTime))
        #         #video_file = open('data/videos/test.mp4', 'rb')  # mp4v로 인코딩 했다면 브라우저에서 실행이 안됨
        #         #video_bytes = video_file.read() 
        #         #st.video(video_bytes) 
        ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
        ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 

    # SSD 메뉴 선택 
    elif selectbox == 'SSD' :
        
        st.title('SSD: Single Shot Detector')
        st.write('한번에 물체를 탐지한다!')
        st.write('Tensorflow model에서 훈련 학습이 된 모델중에 SSD는 이미지를 \
                    CNN 처리를 할 때: Convolutional Neural Network, 하나의 feature로 처리')
        st.write('SSD는 2개로 구성이 되어 있는데, backbone model과 SSD head인데 \
                    백본 모델이란 미리 학습 훈련 된 모델이다.')
        st.write('CNN이 복잡하게 여러개로 구성되어 있는데, 컨볼루션 -> 맥스풀링을 거치면서 \
                    복잡하게 되어 있는데 아래 그림처럼..\
                    SSD head 부분에는 한 개의 convolutioinal layer를 Backbone model에 추가해주면서\
                        Object Detection인 박스를 치면서 위치정보와, 물체의 분류까지 해내게된다' )
        st.image('data/images/show/ssd-head.png')
        st.text('<출처: https://developers.arcgis.com>')
        st.write('SSD는 Grid 셀을 이용해서 이미지를 나누고 클래스와 위치를 찾게 된다')        

        st.write('위에서 말한대로 여러개 CNN 구성으로 다양한 크기의 Convolutional Layer가 생기는데\
                    그래서 SSD가 한번에 다양한 물체를 여러 사이즈의 이미지에서 찾아낼 수 있게 된다')
        st.write('SSD의 장점은 빠르고 정확하다는 것이고 큰 사이즈 이미지에도 잘 작동하지만\
                    그에 반해 작은 이미지에서는 조금 떨어진다고 한다')        
        
        st.write('')
        st.write('이제 아래 버튼을 누르면 SSD 모델을 이용해서 \
                    Object Dectection 할 수 있습니다.')
        st.text('아래의 사진을 확인해 보세요. (원본사진을 확인하세요)')
        st.image('data/images/test/traffic-640.jpg')

        text = '이미지'
        warningPrint(text)

        st.write('미리 로컬에서 오브젝트 디텍션을 마친 이미지 입니다.')
        st.image('data/images/show/ssd_output.jpg')

        ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
        ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
        #한장만 저장
        # if st.button('ssd 전송') :
            
        #     # directory = 'data/images/user-upload'
        #     #filename = save_uploaded_file(directory, img)

        #     testFile = 'traffic-640.jpg'
        #     PATH_TO_TEST_IMAGES_DIR = pathlib.Path('data/images/test')
        #     TEST_IMAGE_PATHS = pathlib.Path(PATH_TO_TEST_IMAGES_DIR, testFile)
        #     print(TEST_IMAGE_PATHS)
        #     #TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))# 여러장일 때사용

        #     # # # # 모델 불러오기 , 함수호출
        #     model_name = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
        #     model_date = '20200711'
        #     detection_model = load_model(model_name, model_date)
        #     # #print(detection_model.signatures['serving_default'].output_dtypes)
        #     # #print(detection_model.signatures['serving_default'].output_shapes)

        #     show_inference(detection_model, TEST_IMAGE_PATHS)
        #     #for image_path in TEST_IMAGE_PATHS:
        #     # show_inference(detection_model, image_path)

        # ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
        # ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 


    # YOLO 메뉴 선택 시 
    elif selectbox == 'YOLO' :

        st.title('Real-Time Object Detection')
        st.write('# You Only Look Once ')
        st.write('이름 한번 꽤 쿨하게 만든 Real-Time Object Detection은 YOLO라고 한다.')
        st.image('data/images/show/sayitYOLO.jpg')
        st.text('<출처: yolo 사이트>, You Only Live Once 아니였어?? ')
        st.write('')
        st.write('하나의 인공 신경망을 이미지에 적용하는 방법을 사용하는데 이미지를 그리드로 나누어서 \
                    예측을 하게 된다.')
        st.write('다른 CNN 방식은 인공신경망이 몇 백개 이상으로 컨볼루션을 해야하는데 \
                    대표적으로 R-CNN 은 하나의 이미지에 수천개의 네트워크가 필요하지만 \
                    하나의 신경망으로 아주 빠르게 예측할 수 있는 장점을 가지고 있다. ')

        st.video('https://www.youtube.com/watch?v=MPU2HistivI')
        st.text('<yolo 공식 영상>')

        st.write('YOLO는 분류와 물체 위치를 벡터로 표현해서 적용하는데.\
                CNN: Convolutional Neural Network 로 만약 개 와 사람을 분류를 하라고 하면 \
                    마지막 output layer가 sigmoid 함수로 하나만 나오게 되는데 \
                    실제로는 개와 사람이 겹쳐있거나 여러 물체가 있다면 하나의 결과만 나오게 되는 문제가 발생하게된다') 
        st.image('data/images/show/woman-640_yolox.jpg')
        st.text('<출처: pixabay.com>')
        st.write('YOLO는 그리드 셀이라는 것을 이용해서 사진을 그리드로 나누게 되고 \
                    쪼개진 그리드 안에서 더 많은 물체를 탐지할 수 있게 되는데\
                    여기에서 한 셀이 물체가 겹친다면 Pc값을 이용해서 높은값을 사용하게 된다  \
                    ')
        
        st.image('data/images/show/yolo_multi_exmp.png')
        st.text('<출처: https://debuggercafe.com>')

        st.write('IOU: Intersection Over Union 라는 것을 이용해서 \
                    Area of Overlap(교집합) / Area of Union (합집합) \
                    이 값은 0~1 사이로 나오는데 이 값이 클 수록 같은 물체가 된다')
        
        st.write('같은 물체의 중심이 겹치는 문제를 해결하기 위해서 \
                    한 그리드 셀에 2개의 벡터를 하나로 합쳐 14개의 벡터로 학습시키는 \
                    Anchor Boxes를 이용하게 된다')

        st.write('')
        
        st.write('이제 사진을 업로드해서') 
        st.write('# Object Dectection 경험해 보세요!')
        # 이미지 / 동영상 선택
        radioSelection = st.radio('사진 또는 동영상을 선택하세요', ['사진', '동영상'])
        
        if radioSelection == '사진' :
            upload_img_list = st.file_uploader('이미지 파일 업로드', type=['png', 'jpeg', 'jpg'], accept_multiple_files=True)
            upload_video = None #변수 선언되기 전에 사용되는거 방지 (아래코드에서 if로 쓰기때문)
        else :
            upload_video = st.file_uploader('동영상 파일 업로드', type=['mp4', 'avi'], accept_multiple_files=False)
            upload_img_list = None

        warningPrint()

        
        st.text('사진 또는 동영상 영상을 확인하려면 위의 라디오 버튼을 선택해주세요')
        
        if radioSelection == '사진' :
            st.write('<로컬에서 확인한 YOLO 이미지 Dectection 영상>')
            video_file = open('data/videos/show/complete_ftod_video_output.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)   
        else :
            st.write('<로컬에서 확인한 YOLO 동영상 Dectection 영상>')
            video_file = open('data/videos/show/complete_ftod_video_output.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)   

        ###########실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
        ###########실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 

        # 이미지 업로드일 경우 실행
        # if upload_img_list is not None:
        #     # 이미지 로드
        #     if st.button('저장 및 분석하기') :
                    
        #         directory = 'data/images/user-upload'
        #         filenameList = []
        #         for upload_img in upload_img_list:
        #             img = load_image(upload_img)
        #             filename = save_uploaded_file(directory, img)    # 이미지 저장
        #             filenameList.append(filename)
        #             print ('{} 저장하였습니다.'.format(filename))

        #         #욜로 모델 만들기
        #         from yolo_model.yolo_model import YOLO
                
        #         st.spinner()
        #         with st.spinner(text='파일이 저장 되었습니다. 분석을 시작합니다.'):
        #             #yolo 객체 만들기
        #             yolo = YOLO(0.6, 0.5)
        #             #클래스 파일 들어 있는 곳 지정
        #             all_classes = get_classes('data/coco_classes.txt')

        #             #image = cv2.imread('data/images/test/test1.jpg')
        #             for i, image in enumerate(filenameList) :
        #                 image = cv2.imread(directory + '/' + image)
                        
        #                 result_image = detect_image(image, yolo, all_classes)

        #                 #이미지가 커서 축소해서 보여주기 
        #                 scaleX = 1.0 
        #                 scaleY = 1.0
        #                 scaleDn = cv2.resize(result_image, None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_LINEAR)

        #                 convertedImg = cv2.cvtColor(scaleDn, cv2.COLOR_BGR2RGB)

        #                 st.image(convertedImg)
        #                 st.success('{}번째 이미지의 YOLO 탐색에 성공하였습니다.'.format(i+1))

        # # video 업로드를 선택했을 때
        # elif upload_video is not None:
        #     #print(upload_video.size)
            
        #     if st.button('동영상 저장 및 분석하기') :
                
        #         directory = 'data/videos/user-upload'
        #         #img = loadCheck(upload_video)
        #         filename = save_uploaded_video(upload_video, directory)    # 이미지 저장
                
        #         if filename is not None:
                    
        #             #videoPath_file = 'data/videos/library1.mp4'
        #             videoPath_file = directory + '/' + filename
                    
        #             type= 'yolo'
        #             # 영상 합성 성공시 True 리턴
        #             if reCaptureVideo(type, videoPath_file) :
        #             #     #video_file = open('data/videos/test_output1.mp4', 'rb') #mp4v 형식이라 브라우저에서 재생안됨, 추후 h264 대체 openh264 알아볼 것
        #             #     #video_bytes = video_file.read()
        #             #     #st.video(video_bytes)
        #                 st.success('영상 합성이 성공하였습니다.')
        #                 st.balloons()
        #             else :
        #                 pass

        #         else :
        #             st.error('파일 저장 중에 에러가 발생하였습니다.')
    
        ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
        ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 


    #시멘틱 세그멘테이션 
    elif selectbox == 'Semantic Segmentation' :

        st.title('Semantic Segmentation')
        st.write('모든 픽셀의 레이블을 예측하는 분할: segmentation 입니다. \
                    ')
        
        st.image('data/images/show/segmentation_ex1.png')

        st.write('Opencv를 이용해서 semantic segmentation을 구현했습니다.')
        st.write('opencv는 BGR로 처리하는 방식이기 때문에 Blob처리를 할 때 색처리를 바꿔줍니다 \
                    사이즈를 Down Scale을 해줍니다.')

        st.write(' Deep Convolution Neural Network 로 학습된 Enet모델을 이용해서 20개의 클래스를 분류 예측을 해서\
                    이때 softmax 액티베이션 함수로 값이 나오는데 이 중에서 가장 큰 값을 argmax()로 구하게 됩니다.\
                    ')
        st.write('다시 원래 이미지를 보여주기 위해서 Up Scale을 해주고\
                    원본 이미지와 마스크 이미지를 합쳐서 보여주게 됩니다.')

        st.image('data/images/show/segmentation_ex2.png')
        st.text('<참고 사이트: https://www.jeremyjordan.me/semantic-segmentation>')

        
        st.info('다행히 Segmentation은 EC2 서버 환경에서 가능합니다.')

        st.write('사진을 고르면 세그멘테이션화 합니다.')
        radioSelection = st.radio('원하는 샘플사진을 선택하세요', ['pic1', 'pic2', 'pic3', 'pic4' ])
        

        if radioSelection == 'pic1':
            imgName = 'bike-640.jpg'
            resizedImg = imageResize(radioSelection, imgName) #이미지 줄여주는 함수
            st.image(resizedImg)

        elif radioSelection == 'pic2':
            imgName = 'hanoi-640.jpg'
            resizedImg = imageResize(radioSelection, imgName)
            st.image(resizedImg)

        elif radioSelection == 'pic3':
            imgName = 'sport-640.jpg'
            resizedImg = imageResize(radioSelection, imgName)
            st.image(resizedImg)

        elif radioSelection == 'pic4':
            imgName = 'traffic-640.jpg'
            resizedImg = imageResize(radioSelection, imgName)
            st.image(resizedImg)
        
        st.text('<위의 이미지는 opencv를 이용하여 리사이즈된 썸네일 입니다.>')
        if st.button('선택! 세그멘테이션 하기') :
            originalImg, cv_enet_model_output, my_legend = makeSegmentation(resizedImg, imgName)
            st.write('선택한 원본 이미지 입니다.')
            st.image(originalImg)

            st.write('시멘틱 세그맨테이션이 적용되었습니다.')
            st.image(cv_enet_model_output)

            st.write('색 분류 표를 참고하세요')
            st.image(my_legend)


    #aboutMe 페이지
    elif selectbox == 'aboutMe' :
        st.write('프로 삽질러가 되어 여기저기 파고 또 파는 주니어(젊은?) dev 지망생? 입니다! ')
        st.write('어쩌다가..')

        st.write('http://54.180.113.157', '개인 블로그: 소소하게 개발하고 블로그 포스트 합니다. 부족한게 많습니다.')

        st.write('')        
        st.write('이번 프로젝트에 사용한 고마운 프로그램들')
        
        st.write('파이썬 언어로 개발했습니다.')
        st.image('data/images/logo/python_logo.png')

        st.write('tensorflow Model 똑똑한 능력자들이 만든~! 감사합니다!')
        st.image('data/images/logo/tensorflow_logo.png')

        st.write('streamlit 프레임워크~ 메인 application 입니다.')
        st.image('data/images/logo/streamlit_logo.png')

        st.write('AWS EC2 서버 입니다. free tier 이지만 \
                    서버로 활용하기에는 정말 훌륭하고 늘 배우고 있습니다.\
                    tensorflow model을 적용하기에는 어렵다는것도 배웁니다.')
        st.image('data/images/logo/awsec2_logo.png')

        st.write('리눅스: 우분투18.04 ubuntu bionic beaver\
                    AWS 서버의 우분투 배포판 운영체제 입니다.')
        st.image('data/images/logo/ubuntu_logo.png')

        st.write('리눅스: CentOS 8 ~ 로컬의 개발환경 OS, 개발할 때 사용했습니다~ \
                    로컬에서 개발하고 GitHub로 commit & push 합니다.')
        st.image('data/images/logo/centos_logo.png')

        st.write('git을 CLI 에서 실행 합니다.\
                    로컬에서 commit 후 AWS 서버에서 pull 받습니다.')
        st.image('data/images/logo/git_logo.png')


        st.write('GitHub repository를 이용해서 서버에 배포 했습니다\
                    업데이트 수정사항이 생길 때마다 사용합니다.')
        st.image('data/images/logo/github_logo.png')

        st.write('도커! 현재 이 application은 도커환경에서 실행되고 있습니다!\
                    어디에서나 동일한 환경으로 맞춰줍니다.')
        st.image('data/images/logo/docker_logo.png')

        st.write('아나콘다! 파이썬 로고가 뱀이라서? 아나콘다도 뱀이였구나..?\
                    초기 가상환경을 셋팅하는데 기본 구성 테스트에 사용하였습니다.')
        st.image('data/images/logo/anaconda_logo.png')

        # 파이썬, tensorflow, 스트림릿, 리눅스, 도커, 아나콘다, 


if __name__ == '__main__' :
    main()