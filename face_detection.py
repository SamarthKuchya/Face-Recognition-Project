import cv2

# connect camera to open cv

cap=cv2.VideoCapture(0) # 0 means default webcam chanege no accourding to available webcams
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
    ret,frame=cap.read() # returns tuple True if connected and frame as video captures
    grey_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret==False:
        print('device not found')
        break

    faces=face_cascade.detectMultiScale(grey_frame,1.3,5) #uses to detect faces by box method and returns the length and width in tuple
    
    # cv2.imshow('Video frame',frame)
    
    # cv2.imshow('B&W frame',grey_frame)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #open cv method to plot rectangle in camera

    
    cv2.imshow('Video frame',frame)
    

#     wait for user input - q , then yoou will stop the loop
    keypressed=cv2.waitKey(1) & 0xFF
    if keypressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()