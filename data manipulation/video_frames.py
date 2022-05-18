import cv2

cap = cv2.VideoCapture("rawdataset/Governador/1.mp4")

if (cap.isOpened()== False):
    print("Error opening video stream or file")

i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('kang'+str(i)+'.jpg',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()