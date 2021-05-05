import streamlit as st

#사용자 함수
from process.tensorflow_od import load_model, show_inference
from utils.image_func import load_image, save_uploaded_file 
from utils.video_func import save_uploaded_video

from utils.ec2_warning import warningPrint
from utils.local import localImageShow
from utils.fake import fakeShow

def tfodDections(type='image') :
    if type=='image' :
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
        # if upload_img_list is not None:
        #     img = load_image(upload_img_list)
            
        #     #한장만 저장
        #     if st.button('저장 및 분석하기') :
        #         directory = 'data/images/user-upload'
        #         filename = save_uploaded_file(directory, img)
                
        #         PATH_TO_TEST_IMAGES_DIR = pathlib.Path('data/images/user-upload')
        #         TEST_IMAGE_PATHS = pathlib.Path(PATH_TO_TEST_IMAGES_DIR, filename)
        #         print(TEST_IMAGE_PATHS)
        #         #TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))# 여러장일 때사용

        #         # # # 모델 불러오기 , 함수호출
        #         model_name = 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8'
        #         model_date = '20200711'
        #         print('model: {}'.format(model_name))

        #         print('start to load model...')
        #         detection_model = load_model(model_name, model_date)
        #         # #print(detection_model.signatures['serving_default'].output_dtypes)
        #         # #print(detection_model.signatures['serving_default'].output_shapes)

        #         show_inference(detection_model, TEST_IMAGE_PATHS)
        #         for image_path in TEST_IMAGE_PATHS:  #여러장 지원할 때, 현재 1장만 지원함 
        #             show_inference(detection_model, image_path)
            ###########여기까지 실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
            ###########여기까지 실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
            ########### 여기서까지 주석 #######
            
        
        ##### cpu문제로 이미지 보여주기용 입니다. #########
        ##### cpu문제로 이미지 보여주기용 입니다. #########
        warningPrint()
        
        st.write('Object Detection을 local에서 하는 캡쳐영상 입니다.')
        video_file = open('data/videos/show/tfod_image_processing.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

        # 미리 작업된 사진 고르기 
        st.write(' ')

        fileNameList = ['pedestrain-car.png', 'students-640.jpg', 'elder-1920.jpg', 'girl-640.jpg', \
                                'crosswalk.jpg']
        # localImageShow() 라디오버튼 및 사진 미리보기 보여주는 함수
        imgName = localImageShow(fileNameList)

        if st.button('선택한 이미지 디텍션 하기') :
            fakeShow(imgName, addDir='tfod')  #파라미터로 디렉토리명 넘겨주기
            st.text('<CPU의 한계로 인해.. 미리 작업이 완료된 사진 입니다.>')

        ##### cpu문제로 이미지 보여주기용 입니다. #########
        ##### cpu문제로 이미지 보여주기용 입니다. #########

    # 비디오 디텍션
    elif type=='video' :
        st.title('Tensorflow Model Object Detection')
    
        st.write('텐서 플로우의 모델을 사용해서 물체를 탐지하는데 \
                    이를 동영상에 적용할 수가 있습니다. \
                    실제로는 동영상도 사진들이 연속적으로 보여지는 것이므로 \
                    Tensorflow Model Object Detection을 \
                    동영상에도 적용해 볼 수가 있습니다.')

        st.write('')
        ####### 워닝 및 비디오만 보여주기
        warningPrint()
        
        st.write('Object Dectection하는 과정 입니다.')
        video_file = open('data/videos/show/tfod_video_processing.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

        st.write(' ')
        st.write('다음은 Object Dectection이 완료된 영상 입니다.')
        st.write('같은 영상으로 YOLO 모델로  Object Dectection 한 영상도 있으니 비교해 보세요 ^^')
        video_file = open('data/videos/show/complete_tfod_video_output.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

        ####### 워닝 및 비디오만 보여주기
        ####### 워닝 및 비디오만 보여주기

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

        #                 # if cv2.waitKey(25) & 0xFF == 27:  #브라우저에서는 안되는 듯
        #                 #     break
        #             else:
        #                 break

                
        #         cap.release()
        #         print('complete')
        #         print('total time: {}'.format(totalTime))
                #video_file = open('data/videos/test.mp4', 'rb')  # mp4v로 인코딩 했다면 브라우저에서 실행이 안됨
                #video_bytes = video_file.read() 
                #st.video(video_bytes) 
        ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
        ############실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
