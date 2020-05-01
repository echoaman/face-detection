import cv2
import numpy as np

cap = cv2.VideoCapture(0)

eyes_detect = cv2.CascadeClassifier('haarcascade_eye.xml')
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
LBFmodel = 'lbfmodel.yaml'

landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

# while True:
#     ret, img = cap.read()
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     face_list = face_detect.detectMultiScale(gray_img, 1.3, 5)

#     for (x,y,w,h) in face_list:
#         cv2.rectangle(img,(x,y), (x+w,y+h), (0,0,255), 1)
#         roi = gray_img[y:y+h, x:x+w]
#         roi_col = img[y:y+h, x:x+w]
#         eye_list = eyes_detect.detectMultiScale(roi)

#         for (eye_x, eye_y, eye_w, eye_h) in eye_list:
#             cv2.rectangle(roi_col, (eye_x, eye_y), (eye_x + eye_w , eye_h + eye_y), (255,0,0) , 1) 
    
#     cv2.imshow('aman',img)

#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break


# cap.release()
# cv2.destroyAllWindows()


while(True):
    # read webcam
    _, frame = cap.read()

    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = face_detect.detectMultiScale(gray, 1.05, 5)    

    for (x,y,w,d) in faces:
        # Detect landmarks on "gray"
        _, landmarks = landmark_detector.fit(gray, np.array(faces))

        for landmark in landmarks:
            temp = landmark[0][36:42]
            topx,topy = temp[1]
            rightx,righty = temp[2]
            bottomx,bottomy = temp[4]
            leftx,lefty = temp[5]

            print('top left',topy - lefty)
            print('botttom right',righty - bottomy)


            for x,y in temp:
                # display landmarks on "frame/image,"
                # with blue colour in BGR and thickness 2
                cv2.circle(frame, (x, y), 1, (255, 0, 0), 2)

    # # save last instance of detected image
    # cv2.imwrite('face-detect.jpg', frame)    
    
    # Show image
    cv2.imshow("frame", frame)

    # terminate the capture window
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()