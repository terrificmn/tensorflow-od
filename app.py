# docker적용할 때 깃허브로 tensorflow/model를 받고 
# 소스파일을 컨테이너로 복사시켜서 해야지 실행이 됨
# 로컬에서 protoc 컴파일 후 api설치까지 해도.. 컨테이너에서 인식을 못하는거 같음
# 도커 컨테이너 안에 적용을 시킨 후 (protoc등..api설치까지)

############ 도커파일로 배포할 때는 아래코드 제거할 것-gpu 설정 코드임
# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ["PLAIDML_NATIVE_PATH"] = "/home/sgtocta/.local/lib/libplaidml.so"
# os.environ["RUNFILES_DIR"]="/home/sgtocta/.local/share/plaidml"
############ 도커파일로 배포할 때는 위의 코드 제거할 것

import streamlit as st

# 왼쪽 메뉴 불러오기
from tfod_menu import tfodDections
from ssd_menu import ssdDections
from yolo_menu import yoloDections
from seg_menu import segmentationDection


st.set_page_config(page_title='ml', page_icon=None, layout='centered', initial_sidebar_state='auto')

def main() :

    selectboxList = ['여기를 선택하세요', 'Tensorflow-object-detection', 'TF Video Object Detection',
                        'YOLO', 'SSD', 'Semantic Segmentation', 'aboutMe']
    selectbox = st.sidebar.selectbox("선택하세요", selectboxList)
    
    
    if selectbox == '여기를 선택하세요' :
        st.write('안녕하세요  😃 😀')
        st.write('')
        st.write('Tensorflow Models를 활용한 이미지/영상 물체 탐색 포트폴리오 사이트 입니다.')
        st.write('')
        st.write('- TensorFlow는 머신러닝을 위한 오픈소스 소프트웨어 입니다.')
        st.write('- 딥러닝으로 학습된 인공지능을 이용해서 사진 및 동영상 속의 물체를 판별할 수 있습니다.')
        st.write('')
        st.write('왼쪽의 메뉴를 선택하면 다양한 model로 Object Dectection을 할 수 있습니다. 💡')
        st.image('data/images/logo/vw-beetle-intro.jpg')
        st.write('감사합니다. 🤓')

    elif selectbox == 'Tensorflow-object-detection' :
        
        tfodDections()
        
    elif selectbox == 'TF Video Object Detection' :
        
        tfodDections(type='video')
        
    # SSD 메뉴 선택 
    elif selectbox == 'SSD' :
        ssdDections()

    # YOLO 메뉴 선택 시 
    elif selectbox == 'YOLO' :
        yoloDections()

    #시멘틱 세그멘테이션 
    elif selectbox == 'Semantic Segmentation' :
        segmentationDection()


    #aboutMe 페이지
    elif selectbox == 'aboutMe' :
        st.write('프로 삽질러가 되어 여기저기 파고 또 파는 주니어(젊은?) dev 지망생? 입니다! ')
        st.write('어쩌다가..')

        st.write('http://54.180.113.157', '개인 블로그: 소소하게 개발하고 블로그 포스트 합니다. 부족한게 많습니다.')

        st.write('')        
        st.write('이번 프로젝트에 사용한 고마운 프로그램들')
        
        
        st.image('data/images/logo/python_logo.png')
        st.write('현재 애플리케이션은 파이썬 언어로 개발했습니다.')

        
        st.image('data/images/logo/tensorflow_logo.png')
        st.write('tensorflow Model 똑똑한 능력자들이 만든~! 감사합니다!')

        
        st.image('data/images/logo/streamlit_logo.png')
        st.write('streamlit 프레임워크~ 깔끔한 구성을 할 수 있었습니다.')

        st.image('data/images/logo/awsec2_logo.png')
        st.write('AWS EC2 서버 입니다. free tier 이지만 \
                    서버로 활용하기에는 정말 훌륭하고 늘 배우고 있습니다.\
                    tensorflow model을 적용하기에는 어렵다는것도 배웁니다.')

        st.image('data/images/logo/ubuntu_logo.png')
        st.write('리눅스: 우분투18.04 ubuntu bionic beaver\
                    AWS 서버의 우분투 배포판 운영체제 입니다.')

        
        st.image('data/images/logo/centos_logo.png')
        st.write('리눅스: CentOS 8 ~ 로컬의 개발환경 OS, 개발할 때 사용했습니다~')


        
        st.image('data/images/logo/git_logo.png')
        st.write('git을 CLI 에서 실행 합니다.\
                    로컬에서 commit, push 후 AWS 서버에서 pull로 받습니다.')

        
        st.image('data/images/logo/github_logo.png')
        st.write('GitHub repository를 이용해서 서버에 배포 합니다.\
                    업데이트 수정사항이 생길 때마다 사용합니다.')
        
        
        st.image('data/images/logo/docker_logo.png')
        st.write('도커! 현재 이 application은 도커환경에서 실행되고 있습니다!\
                    동일한 개발환경에서도 컨테이너로 구성 합니다.')

        st.image('data/images/logo/plaidml_logo.png')
        st.write('인텔의 open소스 plaidml은 GPU로 연산 할 수 있게 도와줍니다.\
                    특히 Mac 이나 AMD 그래픽카드로 처리가 가능해 집니다.')

        st.image('data/images/logo/anaconda_logo.png')
        st.write('아나콘다! 파이썬 로고가 뱀이라서? 아나콘다도 뱀이였구나..?\
                    초기 가상환경을 셋팅하는데 기본 구성 테스트에 사용하였습니다.')





if __name__ == '__main__' :
    main()