import streamlit as st

from utils.ec2_warning import warningPrint
from utils.local import localImageShow
from utils.fake import fakeShow
from process.semanticSeg import makeSegmentation

def segmentationDection() :

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

    fileNameList = ['bike-640.jpg', 'hanoi-640.jpg', 'sport-640.jpg', 'traffic-640.jpg']
    # localImageShow() 라디오버튼 및 사진 미리보기 보여주는 함수
    # 두 번째 파라미터는 생략가능하나 리사이즈된 행렬이 필요함 (기본값은 'no')
    resizedImg, imgName = localImageShow(fileNameList, needResizedImgNp='yes')
    
    st.text('<위의 이미지는 opencv를 이용하여 리사이즈된 썸네일 입니다.>')
    if st.button('선택! 세그멘테이션 하기') :
        originalImg, cv_enet_model_output, my_legend = makeSegmentation(resizedImg, imgName)
        st.write('선택한 원본 이미지 입니다.')
        st.image(originalImg)

        st.write('시멘틱 세그맨테이션이 적용되었습니다.')
        st.image(cv_enet_model_output)

        st.write('색 분류 표를 참고하세요')
        st.image(my_legend)
