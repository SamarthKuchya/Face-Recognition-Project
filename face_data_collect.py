import cv2
import numpy as np
import matplotlib.pyplot as plt

file_name=input('Enter The Name Of Person : ')

# Init camera
cap=cv2.VideoCapture(0)

# Face Detetction
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
Skip=0
face_data=[]
dataset_path='./data/'
while True:
    ret,frame=cap.read()

    if ret==False:
        break

    grey_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    faces=face_cascade.detectMultiScale(frame,1.3,5)

    # print(faces)
    faces=sorted(faces,key=lambda f:f[2]*f[3])
    face_section=frame
    img=frame
    # pick last face because last face is largest
    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        # Extract (Crop out the required part) : Region Of Interest
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        
        Skip+=1
        if Skip%10==0:
            face_data.append(face_section)
            print(len(face_data))   

    cv2.imshow('Frame',frame)
    cv2.imshow('Face Section',face_section)
    img=face_section
    # plt.imshow(face_section)
    # plt.show()
        
    
    key_pressed=cv2.waitKey(1) & 0xFF #converts 32 bit to 8 bit integer
    if key_pressed==ord('q'):
        break
    
# # save as image
# cv2.imwrite(dataset_path+file_name+'.npy', 1, img)  

# convert our face list array into a numpy array
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# convert this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully Saved At "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()