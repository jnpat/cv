import cv2

videoCaptureObject = cv2.VideoCapture(0)
result = True
while(result):
    ret,frame = videoCaptureObject.read()
    cv2.imshow('camera',frame) #uint8
    cv2.imwrite("NewPicture.jpg",frame)
    if cv2.waitKey(10):
        break
    result = False
videoCaptureObject.release()
cv2.destroyAllWindows()