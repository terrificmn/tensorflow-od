import streamlit as st
import os

# # 파일 저장하는 함수

def save_uploaded_video(upload_video, directory):
    videoName = upload_video.name

    # 디렉토리가 있는지 확인
    if not os.path.exists(directory):
            os.makedirs(directory) # 없으면 만듬
    try:
        # upload_video='' #에러 테스트
        with open(os.path.join(directory, videoName),"wb") as f:
            f.write(upload_video.getbuffer())
            print ("파일을 저장했습니다. :{} in {}".format(videoName, directory))
        return videoName

    except :
        print('error while saving file')

