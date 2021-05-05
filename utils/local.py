import streamlit as st
from process.scaledown import imageResize

def localImageShow(fileNameList, needResizedImgNp='no') :
    # 미리 작업된 사진 고르기 
    st.write(' ')
    st.info('이미지를 선택해서 탐색 결과를 볼 수도 있습니다. 사전에 object dectection이 완료된 사진')
    
    st.write('사진을 선택해주세요')
    
    selectionNum = len(fileNameList)

    #넘어온 파일 수 만큼만 라디오 리스트 만들기 
    radioList = []
    for i, filename in  enumerate(fileNameList):
        radioName = 'pic'+ str(i+1)
        radioList.append(radioName)

    radioSelection = st.radio('샘플사진을 선택하세요', radioList)

    for i, imgName in enumerate(fileNameList):
        checkSelectedValue = 'pic' + str(i+1)

        if radioSelection == checkSelectedValue :
            # to resize and show
            resizedImg = imageResize(radioSelection, imgName)
            st.image(resizedImg)

            # 파라미터 resizedImg 기본은 no / True - False 값으로 지정을 하면은 에러가 발생
            if needResizedImgNp == 'no' :
                return imgName
            else :
                return resizedImg, imgName
