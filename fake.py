import streamlit as st

def fakeShow(imageName, addDir) :
    imgDir = 'data/images/show' + '/' + addDir
    #print(imgDir)
    st.success('Object Dectection 성공!')
    st.image(imgDir + '/completed-' + imageName)
