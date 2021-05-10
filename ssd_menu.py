import streamlit as st

from process.tensorflow_od import load_model, show_inference
from process.saveCapTfod import reCaptureVideoTfod
from utils.ec2_warning import warningPrint
from utils.local import localImageShow
from utils.fake import fakeShow
from utils.video_func import save_uploaded_video
from utils.image_func import load_image, save_uploaded_file

def ssdDections ():
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

    st.write('자~ 이제 사진 파일을 업로드 하세요! AI가 분석해드립니다^^')

    radioSelection = st.radio('사진 또는 동영상을 선택하세요', ['사진', '동영상'])
    
    if radioSelection == '사진' :
        upload_img_list = st.file_uploader('이미지 파일 업로드', type=['png', 'jpeg', 'jpg'], accept_multiple_files=True)
        # 이미지 선택만 눌러도 버튼이 활성화 되는 것 방지
        if len(upload_img_list) == 0: # 라디오 선택 시 길이는 0 (업로드 안했을 때)
            upload_img_list = None

        upload_video = None #변수 선언되기 전에 사용되는거 방지 (아래코드에서 if로 쓰기때문)
    else :
        upload_video = st.file_uploader('동영상 파일 업로드', type=['mp4', 'avi'], accept_multiple_files=False)
        upload_img_list = None

    #### 이미지/동영상 업로드 정상 확인- 이미지 디텍션/동영상 디텍션 동작 확인 - may11 2011
    ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
    ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
    # if upload_img_list is not None:
    #     if st.button('SSD object detection') :
    #         #filename = save_uploaded_file(directory, img)

    #         # # # # 모델 불러오기 , 함수호출
    #         model_name = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
    #         model_date = '20200711'
    #         detection_model = load_model(model_name, model_date)
    #         #print(detection_model.signatures['serving_default'].output_dtypes)
    #         #print(detection_model.signatures['serving_default'].output_shapes)

    #         # 파일저장 및 파일이름 리스트
    #         directory = 'data/images/user-upload'
    #         filenameList = []
    #         for upload_img in upload_img_list:
    #             img = load_image(upload_img)
    #             filename = save_uploaded_file(directory, img)    # 이미지 저장
    #             filenameList.append(filename)
    #             print ('{} 저장하였습니다.'.format(filename))

    #         #여러장 처리
    #         for image_path in filenameList:
    #             print(directory + "/" + image_path)
    #             show_inference(detection_model, directory + "/" + image_path)
    #     ###########실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
    #     ###########실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 

    # ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
    # ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
    
    # ### video 업로드를 선택했을 때
    # elif upload_video is not None:
    #     #print(upload_video.size)
    #     if st.button('SSD object detection') :
            
    #         ### 모델 불러오기 , 함수호출
    #         model_name = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
    #         model_date = '20200711'
    #         detection_model = load_model(model_name, model_date)
    #         # #print(detection_model.signatures['serving_default'].output_dtypes)
    #         # #print(detection_model.signatures['serving_default'].output_shapes)

    #         ## ssd 는 고정 이미지로 한장만 불러와서 예측하고 있음
    #         ## 추후 파일 입력 받고, 동영상 부분도 함수로 만들어서 tfod와 같이 사용할 수 있게 하기
    #         # reCaptureVideo() 함수를 사용하면 되나, tfod의 모델을 넘겨줘야는데 그러려면 파라미터를 추가해줘야해서 함수사용안함
    #         # 따로 만들던가, ssd동영상 부분이 안만들어져있으니 그 부분을 생각하고 만들기~ 
    #         # 그리고 reCaptureVideo()를 아예 yolo 전용으로 만드는 것을 생각해보기
    #         ## 일단 비디오 부분은 yolo랑 같이 할지 생각해볼 것. yolo랑 같이 하면 파라미터를 추가해야함 2021 5 11

    #         #영상 처리 부분
    #         directory = 'data/videos/user-upload'
    #         #img = loadCheck(upload_video)
    #         filename = save_uploaded_video(upload_video, directory)    # 이미지 저장
            
    #         if filename is not None:
    #             videoPath_file = directory + '/' + filename
                
    #             # 영상 합성 성공시 True 리턴
    #             # yolo랑은 다르게 tfod/ssd는 show_inference()함수에 모델을 넘겨줘야해서 파라미터가 다름
    #             type='ssd'
    #             if reCaptureVideoTfod(type, videoPath_file, detection_model) :
    #             ## 파일 열어볼 때 현재는 주석 처리(사용안함 - may 2021) / mp4v형식으로 저장되는 형식이라서 브라우저에서 재생아노딤
    #             ##     #video_file = open('data/videos/test_output1.mp4', 'rb') // 추후 h264 대체 openh264 알아볼 것
    #             ##     #video_bytes = video_file.read()
    #             ##     #st.video(video_bytes)
    #                 st.success('영상 합성이 성공하였습니다.')
    #                 st.balloons()
    #             else :
    #                 pass

    #         else :
    #             st.error('파일 저장 중에 에러가 발생하였습니다.')
    # ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
    # ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 


    ####### warning 및 동영상으로 대체 부분 ######
    ###### warning 및 동영상으로 대체 부분 ######
    
    text = '이미지' # warningPrint() 파라미터 넘겨주기 string
    warningPrint(text)
    # 원래 로컬 파일 작업 영상 보여주는 코드
    st.write(' ')
    st.write('다음은 SSD 로 로컬에서 실행되는 영상 입니다.')
    video_file = open('data/videos/show/ssd_video_processing.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    st.write('')
    st.write('Object Dectection이 완료된 사진 또는 동영상 영상을 선택하세요')

    st.info('이미지를 선택해서 탐색 결과를 볼 수도 있습니다. -사전에 object dectection이 완료된 사진')
    ### 미리 작업된 사진 / 동영상 선택 부분
    fakeSelection = st.radio('사진 또는 동영상을 선택하세요', ['image', 'video'])        
    if fakeSelection == 'image' :
        fileNameList = ['bike-640.jpg', 'bike_waiting.png', 'hanoi-640.jpg', 'traffic-640.jpg', \
                            'couplebike-640.jpg', 'ford-640.jpg']
        imgName = localImageShow(fileNameList)

        if st.button('선택한 이미지 디텍션 하기') :
            fakeShow(imgName, addDir='ssd')
            st.text('<CPU의 한계로 인해.. 미리 작업이 완료된 사진 입니다.>')
    else : 
        st.write('you need to create codes ... here for video')

    ###### warning 및 동영상으로 대체 부분 ######
    ###### warning 및 동영상으로 대체 부분 ######