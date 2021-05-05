import streamlit as st

def warningPrint(text='동영상') :


    st.warning('죄송합니다. 이미지 업로드 및 분석은 지원이 종료되었습니다.')
    st.info('현재 서버는 AWS EC2 t2.micro Free Tier 입니다. \
            CPU 1개인 하드웨어의 한계 때문에 Object Detection이 불가능 합니다. \
            실시간으로 처리가 불가능하여, 로컬에서 작동되는 모습을 확인할 수 있습니다. \
            {}(으)로 보실 수 있습니다. 감사합니다.'.format(text))
    st.write('')

