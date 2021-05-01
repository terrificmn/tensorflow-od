import numpy as np
#import argparse
import imutils
import time
import cv2
import streamlit as st


def makeSegmentation(image_cv, imgName) :
    st.spinner()
    with st.spinner(text='분석을 시작합니다...'):
        TEST_IMAGE_DIR = 'data/images/test'
        ## input layer로 입력할 때 600으로 고정
        SET_WIDTH = int(600)
        # feature scaling 
        normalize_image = 1 / 255.0
        resize_image_shape = (1024, 512)

        #originalImg = cv2.imread('data4/images/example_02.jpg')
        originalImg = cv2.imread( TEST_IMAGE_DIR + '/' + imgName)

        # image resize 하는 im(image)util를 사용
        # cv2 에서도 resize할 수 있음
        originalImg = imutils.resize(originalImg, width=SET_WIDTH)

        # opencv 의 pre-trained model을 통해서, 예측하기 위해서는
        ## 입력이미지를 blob으로 바꿔줘야 한다

        # swapRB=True, crop=False 디폴트 파라미터 생략가능
        ## swapRB 파라미터는 보통 RGB로 되어 있는데 cv는 BGR로 처리하기 때문에 R과 B를 바꿔주는 것
        blob_img = cv2.dnn.blobFromImage(originalImg, normalize_image, resize_image_shape, 0, swapRB=True, crop=False)

        # Enet 모델 가져오기 # 예측
        cv_enet_model = cv2.dnn.readNet('enet_data/enet-cityscapes/enet-model.net')
        #print(cv_enet_model)

        # blob 이미지 셋팅 blob한 이미지를 넣어준다
        cv_enet_model.setInput(blob_img)

        # forward() 결과를 저장
        cv_enet_model_output = cv_enet_model.forward()

        # enet은 20개의 세그멘티이션 해줌 Enet은 20개 클래스를 학습시키는 것이므로 20개가 나온다 
        ## 이미지는 1개, 20개의 분류(클래스의 갯수) 512 행의 갯수 1024 렬의 갯수
        ##이제 20개 행렬의 각각 첫 번째 픽셀값에는 각각 label 첫번째 unabled 값일 확률이 쭉 나오게 되고 
        ##softmax로 숫자가 되어 있게 되고, 이 중에서 가장 큰 값 argmax()를 하면 나오는 값이 가장 높은 확률이 되게 된다 
        #print(cv_enet_model_output.shape)
        #(1, 20, 512, 1024) 결과는 이렇게 나옴


        ## 레이블 이름 로딩
        ## 텍스트 파일이 엔터로 구분되어 있음

        label_values = open('enet_data/enet-cityscapes/enet-classes.txt').read().split('\n')
        ## 마지막 ''제거
        label_values = label_values[: -1]
        #print(label_values)


        IMG_OUTPUT_SHAPE_START = 1
        IMG_OUTPUT_SHAPE_END = 4
        # 튜플의 하나씩 배열에 저장
        classes_num, h, w = cv_enet_model_output.shape[IMG_OUTPUT_SHAPE_START : IMG_OUTPUT_SHAPE_END]


        # **** 중요 ******
        # 원래의 모양인 (1, 20, 512, 1024)에 있는 값을, 변수로 저장
        # 20은 클래스의 갯수, 512는 높이, 1024는 너비로 저장
        # 가장 큰 값을 저장하게 해준다 (가장 큰 값들만 가지고 있음)
        ## 20개의 행렬 중에 softmax 값으로 가지고 있는 것을 큰 값을 저장하게 된다 
        ## 20개의 행렬 [0]배열에 있음
        # 모델의 아웃풋 20개 행렬을, 하나의 행렬로 만든다. 
        class_map = np.argmax(cv_enet_model_output[0], axis = 0)


        # 색 정보 가져오기
        # 색으로 표시하기 위해서 color.txt 파일에 색으로 지정해놓았다
        # 그래서 레이블을 색으로 표시할 수 있음

        CV_ENET_SHAPE_IMG_COLORS = open('enet_data/enet-cityscapes/enet-colors.txt').read().split('\n')
        # 맨 마지막 따옴표 없애기 slicing
        CV_ENET_SHAPE_IMG_COLORS = CV_ENET_SHAPE_IMG_COLORS[: -2+1]

        # 리스트 컴프리핸션 한줄로 표현하기
        temp_list = np.array( [ np.array(color.split(',')).astype('int') for color in CV_ENET_SHAPE_IMG_COLORS ] )
        # 위의 코드와 같음
        # temp_list = []
        # for color in CV_ENET_SHAPE_IMG_COLORS:
        #     color_list = color.split(',')
        #     #print(color_list)
        #     color_num_list = np.array(color_list).astype('int')
        #     temp_list.append(color_num_list)


        # 이제 색을 가지고 있는 변수가 됨
        CV_ENET_SHAPE_IMG_COLORS = np.array(temp_list)

        #그래서 색깔별로 , 클래스에 해당하는 숫자가 적힌 class_map을 
        # 각 숫자에 매핑되는 색깔로 셋팅을 해준 것임
        # 따라서 각 픽셀별 색깔 정보가 들어가게되는 것 RGB값으로 3차원으로 만들어주게 됨
        ## 데이터 억세스를 해서 3차원 이미지가 됨
        mask_class_map = CV_ENET_SHAPE_IMG_COLORS[class_map]


        ## 리사이즈 (원래 이미지 크기로 되돌리기 위해서 up scale)
        mask_class_map = cv2.resize(mask_class_map, (originalImg.shape[1], originalImg.shape[0] ),
                                                    interpolation = cv2.INTER_NEAREST )
        # interpolation을 INTER_NEAREST로 한 이유는 
        # 레이블 정보는 (0~19) 와 컬러정보(23, 100, 243) 는 둘 다 int 이므로 
        # 가장 가까운 픽셀 정보와 동일하게 셋팅해서 비는 공간을 정수로 만들어 주기위해서 
        class_map = cv2.resize(class_map, (originalImg.shape[1], originalImg.shape[0]), 
                                interpolation = cv2.INTER_NEAREST)


        # 원본 이미지랑 색 마스크 이미지를 합쳐서 보여준다
        # 가중치 비율을 줘서 보여준다
        # float이 안나오도록 uint8로 바꿔준다 
        # numpy array를 2개를 더하면 사진을 합치는 것인데
        ## 예를 들어 100 과 200 이 있다면 300 되어서 255를 넘어가기 때문에 
        ##비율을 조정해줘서 합치게 됨--> 하지만 소수점이 될 수 있기 때문에 
        ##astype('int')로 만들어 주는 것
        cv_enet_model_output = (( 0.4 * originalImg) + ( 0.6 * mask_class_map)).astype('uint8')


        # legend표시해주기 위해서 zeros로 만들기
        my_legend = np.zeros( ( len(label_values) * 25 , 300, 3), dtype='uint8' )
        # zip()는 2개의 리스트를 묶어서 처리할 수 있게함 (row를 묶어서 처리(튜플로))
        ## for 루프를 돌릴 때  enumerate는 i에 0번째로 인덱스 값을 주게 된다
        for ( i, (class_name, img_color)) in enumerate( zip(label_values, CV_ENET_SHAPE_IMG_COLORS)) :
            color_info = [ int(color) for color in img_color ] #리스트 컴프리핸션으로 int로 바꿈
            cv2.putText(my_legend, class_name, (5, (i * 25)+17), cv2.FONT_HERSHEY_COMPLEX , 0.5, (0, 255, 0),2)
            cv2.rectangle(my_legend, (100, (i*25)), (300, (i*25) + 25) , tuple(color_info), -1)



        originalImg = cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB)

        cv_enet_model_output = cv2.cvtColor(cv_enet_model_output, cv2.COLOR_BGR2RGB)
        
        my_legend = cv2.cvtColor(my_legend, cv2.COLOR_BGR2RGB)

        st.success('Semantic Segmentation에 성공했습니다.')
        return originalImg, cv_enet_model_output, my_legend




