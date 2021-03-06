#from operator import truediv
import cv2
import time
from process.tensorflow_od import show_inference

##tfod video와 ssd video 만드는 함수
def reCaptureVideoTfod(type, videoPath, model) :

    cap = cv2.VideoCapture(videoPath)

    if cap.isOpened() == False:
        print("error occured to start to play a video")

    else:
        
        #######
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # #이미지 사이즈 줄이기
        # if int(frame_width / 2) % 2 ==0 : #짝수 
        #     frame_width = int(frame_width / 2)
        # else:
        #     frame_width = int(frame_width / 2) + 1 #홀수가 안되게 만들어 줌

        # if int(frame_height / 2) % 2 == 0 :
        #     frame_height = int(frame_height / 2)
        # else:
        #     frame_height = int(frame_height / 2 ) + 1

        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4v 성공
        #fourcc = cv2.VideoWriter_fourcc('X','2','6','4')  # x264 , h264 실패, 'X','2','6','4'
        # 일단, opencv가 h264지원을 안한다는거 같음, 윈도우에서는 방법이 있는거 같은데 
        # mp4v는 잘 되지만, 브라우저에서 지원을 안함 ㅋㅋㅋ
        
        # 내보내기 부분~ 저장될 디렉토리/파일명  (파일명은 type을 앞에 붙여주기)
        out = cv2.VideoWriter('data/videos/' + type + '_output.mp4', 
                                fourcc,
                                10, 
                                ( frame_width, frame_height) )
        ## 저장하는 코드 write()메소드 부분을 주석 해제할 것.. 아래코드
        #######
        
        totalTime = 0 #시간 기록 

        while cap.isOpened():
            ret, frame = cap.read() #동영상의 사진을 하나씩 frame에 넣어준다
            if ret == True:
                #cv2.imshow('Frame', frame)
                startTime = time.time()
                
                isImage = False # 비디오로 만들 것이기 때문에 False 를 준다. (기본은 True: 이미지에서도 사용함!)
                processedImg = show_inference(model, frame, isImage)
            
                endTime = time.time()
                # 처리 시간 출력
                precessTime = endTime-startTime
                print(precessTime)
                totalTime += precessTime
                #save resized image 
                out.write(processedImg)

                # if cv2.waitKey(25) & 0xFF == 27:  # 웹브라우저에서는 안됨
                #     break
            else:
                break
    
    cap.release()
    return True
