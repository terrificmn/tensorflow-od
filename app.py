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

st.set_page_config(page_title='ml', page_icon=None, layout='centered', initial_sidebar_state='auto')


def main() :
    
    selectbox = st.sidebar.selectbox("선택하세요", ['test', 'Tensorflow-object-detection' ])

    if selectbox == 'test' :
        # 테스트 입니다.
        # my_bar = st.progress(0)
        # my_bar.progress(50)
        # my_bar.progress(100)

        # for percent_complete in range(100):
        #     
        pass    

    elif selectbox == 'Tensorflow-object-detection' :
        st.write('## Tensorflow Models Object Detection')
        st.write('')
        st.write('텐서 플로우 모델은 머신러닝으로 학습이 되어 있습니다.') 
        st.write('물체가 어떤 것인지 분류를 하면서 (예: 개인지 사람인지?)')
        st.write('동시에 어느 위치에 있는지까지 (박스를 만듬) 탑색 합니다.')
        st.write('')
        st.write('사진 파일을 업로드 하세요! AI가 분석해드립니다^^')
        upload_img_list = st.file_uploader('이미지 파일 업로드', type=['png', 'jpeg', 'jpg'], accept_multiple_files=False)
    
        if upload_img_list is not None:
        #print(upload_img_list)

            img = load_image(upload_img_list)

            # 여러장 받기 accept_multiple_files=True 로 바꿔주기
            # # 2-1 각 파일을 이미지로 바꿔줘야 한다.
            # image_list = []
            # # 2-2 모든 파일이 image_list에 이미지로 저장됨
            # for image_file in upload_img_list:
            #     img = load_image(image_file)
            #     image_list.append(img)

            #한장만 저장
            if st.button('저장 및 분석하기') :
                directory = 'data/images/user-upload'
                filename = save_uploaded_file(directory, img)
                
                # # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
                #PATH_TO_TEST_IMAGES_DIR = pathlib.Path('models/research/object_detection/test_images')
                
                PATH_TO_TEST_IMAGES_DIR = pathlib.Path('data/images/user-upload')
                TEST_IMAGE_PATHS = pathlib.Path(PATH_TO_TEST_IMAGES_DIR, filename)
                #print(TEST_IMAGE_PATHS)
                #TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))# 여러장일 때사용

                # # # 모델 불러오기 , 함수호출
                model_name = 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8'
                model_date = '20200711'
                detection_model = load_model(model_name, model_date)
                # #print(detection_model.signatures['serving_default'].output_dtypes)
                # #print(detection_model.signatures['serving_default'].output_shapes)

                show_inference(detection_model, TEST_IMAGE_PATHS)
                # for image_path in TEST_IMAGE_PATHS:
                #     show_inference(detection_model, image_path)



if __name__ == '__main__' :
    main()