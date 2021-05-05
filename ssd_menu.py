import streamlit as st


from utils.ec2_warning import warningPrint
from utils.local import localImageShow
from utils.fake import fakeShow


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


    ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
    ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
    if upload_img_list is not None:

        if st.button('SSD object detection') :
        
            
            #filename = save_uploaded_file(directory, img)

            # # # # 모델 불러오기 , 함수호출
            model_name = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
            model_date = '20200711'
            detection_model = load_model(model_name, model_date)
            #print(detection_model.signatures['serving_default'].output_dtypes)
            #print(detection_model.signatures['serving_default'].output_shapes)

            # 파일저장 및 파일이름 리스트
            directory = 'data/images/user-upload'
            filenameList = []
            for upload_img in upload_img_list:
                img = load_image(upload_img)
                filename = save_uploaded_file(directory, img)    # 이미지 저장
                filenameList.append(filename)
                print ('{} 저장하였습니다.'.format(filename))

            #여러장 처리
            for image_path in filenameList:
                print(directory + "/" + image_path)
                show_inference(detection_model, directory + "/" + image_path)
        ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
        ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 

    ####### warning 및 동영상으로 대체 부분 ######
    ###### warning 및 동영상으로 대체 부분 ######
    
    text = '이미지' # warningPrint() 파라미터 넘겨주기 string
    warningPrint(text)

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

    # 원래 로컬 파일 보여주는 코드
    # st.write('대신 미리 로컬에서 오브젝트 디텍션을 마친 이미지를 확인해 보세요.')
    # st.image('data/images/show/ssd_output.jpg')

    # st.write(' ')
    # st.write('다음은 SSD 로 로컬에서 실행되는 영상 입니다.')
    # video_file = open('data/videos/show/ssd_video_processing.mp4', 'rb')
    # video_bytes = video_file.read()
    # st.video(video_bytes)
    
    ###### warning 및 동영상으로 대체 부분 ######
    ###### warning 및 동영상으로 대체 부분 ######

    ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
    ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
    #한장만 저장
    # if st.button('SSD object detection') :
        
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
        #for image_path in TEST_IMAGE_PATHS:
        # show_inference(detection_model, image_path)

    # ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
    # ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 