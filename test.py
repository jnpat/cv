import cv2

videoCaptureObject = cv2.VideoCapture(0)
result = True
i=20
while(True):
    ret,frame = videoCaptureObject.read()
    cv2.imshow('camera',frame) #uint8
    cv2.imwrite("im/NewPicture.jpg",frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("imm/i"+str(i)+".jpg",frame)
        if i >= 0:
            i -=1;
        else:
            break
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break
videoCaptureObject.release()
cv2.destroyAllWindows()