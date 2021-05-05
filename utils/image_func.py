import streamlit as st
import os
from PIL import Image
from datetime import datetime
import random


# 이미지 처리 로드하는 함수
def load_image(image_file) :
    img = Image.open(image_file)
    return img

# 디렉토리와 파일을 주면, 해당 디렉토리에 이미지 파일을 저장하는 함수
def save_uploaded_file(directory, img):
    # 1. 디렉토리가 있는지 확인, 없으면 만든다
    if not os.path.exists(directory):
        os.makedirs(directory) # 없으면 만듬

    ext = img.format.lower() # 확장자 저장

    # 2. 이미지 저장 img.save()
    # datetime.now()을 이용해서 isoformat()으로 바꿔주기
    # : .은 에러가 남 ----> replace로 바꿔준다
    isoNowTime = str(datetime.now().isoformat()).replace(':', '-').replace('.', '-')
    # 2개 정도는 마이크로 시간 이후로 같은 파일로 생성이 되서 랜덤으로 만든 숫자 추가 만들어 주기
    random_nbr = '-'+ str(random.randint(1,10)) 
    
    filename = isoNowTime+random_nbr+'.'+ext
    
    img.save(directory + '/' + filename ) #시간랜덤으로만들어진 문자열+확장자 저장
    
    #st.success('{}이 {}에 파일이 저장 되었습니다.'.format(filename, directory) )
    
    return filename


