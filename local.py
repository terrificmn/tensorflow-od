import streamlit as st
from scaledown import imageResize

def localImageShow(fileNameList) :
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
            return imgName

    # if st.button('선택한 이미지 디텍션 하기') :
    #     fakeShow(imgName, addDir='yolo')
    #     st.text('<CPU의 한계로 인해.. 미리 작업이 완료된 사진 입니다.>')

    # elif radioSelection == 'pic2':
    #     imgName = 'city-640.jpg'
    #     resizedImg = imageResize(radioSelection, imgName)
    #     st.image(resizedImg)

    # elif radioSelection == 'pic3':
    #     imgName = 'bike-crosswalke.jpg'
    #     resizedImg = imageResize(radioSelection, imgName, 0.15, 0.15)
    #     st.image(resizedImg)
    
    # elif radioSelection == 'pic4':
    #     imgName = 'hanoi-ssd.jpg'
    #     resizedImg = imageResize(radioSelection, imgName)
    #     st.image(resizedImg)
    
    # elif radioSelection == 'pic5':
    #     imgName = 'ford-stopsign.jpg'
    #     resizedImg = imageResize(radioSelection, imgName)
    #     st.image(resizedImg)

    


# st.write('Object Dectection이 완료된 사진 또는 동영상 영상을 선택하세요')
# fakeSelection = st.radio('사진 또는 동영상을 선택하세요', ['image', 'video'])
# if fakeSelection == 'image' :
#     st.write('<로컬에서 확인한 YOLO 이미지 Dectection 영상>')
#     video_file = open('data/videos/show/yolo_image_prosessing.mp4', 'rb')
#     video_bytes = video_file.read()
#     st.video(video_bytes)   

#     ##### cpu문제로 이미지 보여주기용 입니다. #########
#     ##### cpu문제로 이미지 보여주기용 입니다. #########
    
#     # 미리 작업된 사진 고르기 
#     st.write(' ')
#     st.info('이미지를 선택해서 탐색 결과를 볼 수도 있습니다. 사전에 YOLO object dectection이 완료된 사진')
    
#     st.write('사진을 선택해주세요')

#     radioSelection = st.radio('샘플사진을 선택하세요', ['pic1', 'pic2', 'pic3', 'pic4', 'pic5' ])
    
#     if radioSelection == 'pic1':
#         imgName = 'ford-640.jpg'
#         resizedImg = imageResize(radioSelection, imgName)
#         st.image(resizedImg)

#     elif radioSelection == 'pic2':
#         imgName = 'students-640.jpg'
#         resizedImg = imageResize(radioSelection, imgName)
#         st.image(resizedImg)

#     elif radioSelection == 'pic3':
#         imgName = 'elder-1920.jpg'
#         resizedImg = imageResize(radioSelection, imgName, 0.15, 0.15)
#         st.image(resizedImg)
    
#     elif radioSelection == 'pic4':
#         imgName = 'girl-640.jpg'
#         resizedImg = imageResize(radioSelection, imgName)
#         st.image(resizedImg)
    
#     elif radioSelection == 'pic5':
#         imgName = 'bike-couple-640.jpg'
#         resizedImg = imageResize(radioSelection, imgName)
#         st.image(resizedImg)

#     if st.button('선택한 이미지 디텍션 하기') :
#         fakeShow(imgName, addDir='yolo')
#         st.text('<CPU의 한계로 인해.. 미리 작업이 완료된 사진 입니다.>')
    ##### cpu문제로 이미지 보여주기용 입니다. #########
    ##### cpu문제로 이미지 보여주기용 입니다. #########