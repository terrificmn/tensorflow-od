import streamlit as st

#사용자 함수
from process.yolo import get_classes, detect_image
from process.saveCap import reCaptureVideo


from utils.ec2_warning import warningPrint
from utils.local import localImageShow
from utils.fake import fakeShow


def yoloDections ():
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
        # 이미지 선택만 눌러도 버튼이 활성화 되는 것 방지
        if len(upload_img_list) == 0: # 라디오 선택 시 길이는 0 (업로드 안했을 때)
            upload_img_list = None

        upload_video = None #변수 선언되기 전에 사용되는거 방지 (아래코드에서 if로 쓰기때문)
    else :
        upload_video = st.file_uploader('동영상 파일 업로드', type=['mp4', 'avi'], accept_multiple_files=False)
        upload_img_list = None

    ###### warning 및 동영상으로 대체 부분 ######
    ###### warning 및 동영상으로 대체 부분 ######
    warningPrint()
    
    st.write('Object Dectection이 완료된 사진 또는 동영상 영상을 선택하세요')
    fakeSelection = st.radio('사진 또는 동영상을 선택하세요', ['image', 'video'])

    # 이미지 선택일 경우
    if fakeSelection == 'image' :
        st.write('<로컬에서 확인한 YOLO 이미지 Dectection 영상>')
        video_file = open('data/videos/show/yolo_image_prosessing.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)   

        # 미리 작업된 사진 고르기 
        st.write(' ')
        fileNameList = ['ford-640.jpg', 'students-640.jpg', 'elder-1920.jpg', 'girl-640.jpg', \
                            'bike-couple-640.jpg']
        # localImageShow() 라디오버튼 및 사진 미리보기 보여주는 함수
        imgName = localImageShow(fileNameList)

        if st.button('선택한 이미지 디텍션 하기') :
            fakeShow(imgName, addDir='yolo')  #파라미터로 디렉토리명 넘겨주기
            st.text('<CPU의 한계로 인해.. 미리 작업이 완료된 사진 입니다.>')

    else :
        st.write('<로컬에서 확인한 YOLO 동영상 Dectection 영상>')
        video_file = open('data/videos/show/yolo_video_processing.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)   

        st.write(' ')
        st.write('< YOLO Object Dectection 완료 영상>')
        video_file = open('data/videos/show/yolo_video_output.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)   
    
    ###### warning 및 동영상으로 대체 부분 ######
    ###### warning 및 동영상으로 대체 부분 ######

    ###########실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 
    ###########실제 작동 확인 완료 ############### 실제 실행 시 주석을 해제 (cpu한계로 주석처리) 

    #이미지 업로드일 경우 실행
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
    #             print('load model complete')
                
    #             #image = cv2.imread('data/images/test/test1.jpg')
    #             for i, image in enumerate(filenameList) :
    #                 image = cv2.imread(directory + '/' + image)
                    
    #                 result_image = detect_image(image, yolo, all_classes)

    #                 #이미지가 커서 축소해서 보여주기  # 현재는 축소 안함
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