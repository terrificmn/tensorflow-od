import os
import time
import cv2
import numpy as np
from numpy.lib.utils import safe_eval

def process_image(img):
    """이미지 리사이즈하고 차원 확장
    img: 원본 사이즈
    결과는 (64, 64, 3) 으로 프로세싱된 이미지 반환"""

    image_org = cv2.resize(img, (416, 416), interpolation = cv2.INTER_CUBIC)
    image_org = np.array(image_org, dtype='float32')
    # normalizing
    image_org = image_org / 255.0
    image_org = np.expand_dims(image_org, axis = 0)

    return image_org



def get_classes(file) :
    """ 클래스의 이름을 가져온다 
        file 받으면 리스트로 클래스 이름을 반환한다 """
    
    with open(file) as f:
        name_of_class = f.readlines() # 전체 라인 가져오기
        #list comprehesion -- 전체 가지고 와서 strip()해주기
        name_of_class = [ class_name.strip() for class_name in name_of_class]

        return name_of_class


def box_draw(image, boxes, scores, classes, all_classes):
    """image: 오리지날 이미지
    boxes: 오브젝트의 박스데이터, ndarray
    classes: 오브젝트의 클래스 정보, ndarray
    scores : 오브젝트의 확률"""

    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        startTime = time.time()
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1.3, (0, 255, 0), 1,
                    cv2.LINE_AA)
        endTime = time.time()

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))
        print('processing time: ', endTime-startTime)
        print()



def detect_image( image, yolo, all_classes) :
    """ image : 오리지널 이미지
    yolo :  욜로 모델
    all_classes: 전체 클래스 이름

    변환된 이미지 리턴! """
    
    #process_image()함수 호출로 프로세싱된 이미지 받아온다
    pimage = process_image(image)

    image_boxes, image_classes, image_scores = yolo.predict(pimage, image.shape)

    ##오브젝트가 있다면
    if image_boxes is not None:
        box_draw(image, image_boxes, image_scores, image_classes, all_classes)

    return image



