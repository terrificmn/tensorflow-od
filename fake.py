import streamlit as st

def fakeShow(imageName) :
    imgDir = 'data/images/show'
    
    st.success('Object Dectection 성공!')
    st.image(imgDir + '/completed-' + imageName)
