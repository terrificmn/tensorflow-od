import cv2

def imageResize (selection, imgName, scaleX=0.4, scaleY=0.4) :
    
    TEST_IMAGE_DIR = 'data/images/test'
    
    if selection == 'pic1':
        img = cv2.imread(TEST_IMAGE_DIR + '/' + imgName, 1) 
        scaleDown = cv2.resize(img, None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_LINEAR)
        scaleDown = cv2.cvtColor(scaleDown, cv2.COLOR_BGR2RGB)

        return scaleDown
    
    if selection == 'pic2':
        img = cv2.imread(TEST_IMAGE_DIR + '/' + imgName, 1) 
    
        scaleDown = cv2.resize(img, None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_LINEAR)
        scaleDown = cv2.cvtColor(scaleDown, cv2.COLOR_BGR2RGB)
        return scaleDown
    
    if selection == 'pic3':
        img = cv2.imread(TEST_IMAGE_DIR + '/' + imgName, 1) 
    
        scaleDown = cv2.resize(img, None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_LINEAR)
        scaleDown = cv2.cvtColor(scaleDown, cv2.COLOR_BGR2RGB)
        return scaleDown
    
    if selection == 'pic4':
        img = cv2.imread(TEST_IMAGE_DIR + '/' + imgName, 1) 
    
        scaleDown = cv2.resize(img, None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_LINEAR)
        scaleDown = cv2.cvtColor(scaleDown, cv2.COLOR_BGR2RGB)
        return scaleDown

    