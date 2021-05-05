import cv2

def imageResize (selection, imgName, scaleX=0.45, scaleY=0.45) :
    
    TEST_IMAGE_DIR = 'data/images/test'
    img = cv2.imread(TEST_IMAGE_DIR + '/' + imgName, 1)
    #print(img.shape)
    # 이미지 크기 비교 후 비율 조정
    if img.shape[0] >= 1024 :
        scaleX = 0.2
        scaleY = 0.2

    if selection == 'pic1':
        scaleDown = cv2.resize(img, None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_LINEAR)
        scaleDown = cv2.cvtColor(scaleDown, cv2.COLOR_BGR2RGB)
        return scaleDown
    
    elif selection == 'pic2':
    
        scaleDown = cv2.resize(img, None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_LINEAR)
        scaleDown = cv2.cvtColor(scaleDown, cv2.COLOR_BGR2RGB)
        return scaleDown
    
    elif selection == 'pic3':

        scaleDown = cv2.resize(img, None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_LINEAR)
        scaleDown = cv2.cvtColor(scaleDown, cv2.COLOR_BGR2RGB)
        return scaleDown
    
    elif selection == 'pic4':
    
        scaleDown = cv2.resize(img, None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_LINEAR)
        scaleDown = cv2.cvtColor(scaleDown, cv2.COLOR_BGR2RGB)
        return scaleDown
    
    elif selection == 'pic5':

        scaleDown = cv2.resize(img, None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_LINEAR)
        scaleDown = cv2.cvtColor(scaleDown, cv2.COLOR_BGR2RGB)
        return scaleDown
    
    elif selection == 'pic6':
    
        scaleDown = cv2.resize(img, None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_LINEAR)
        scaleDown = cv2.cvtColor(scaleDown, cv2.COLOR_BGR2RGB)
        return scaleDown

    