import numpy as np
import cv2
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()
cap = cv2.VideoCapture(0)
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))
while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
       
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    out.write(frame.astype('uint8'))
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
#1st,  import the necessary libraries — OpenCV for computer vision tasks and NumPy for handling arrays.
#2nd create a HOG descriptor, which is a feature extractor that works well for detecting human shapes. Then, we load a pre-trained SVM model that knows how to detect people.
#3rd open the webcam so that the system can capture live video frames in real time.
#4th set up a video writer to save the output video. This means the final video will show all the detected people with rectangles drawn around them.
#5th a loop, we read each video frame from the webcam, resize it to make detection faster, and convert it to grayscale if needed to speed up processing.
#6th use the HOG descriptor to scan each frame and detect people. The detector returns the position of each person it finds.
#7th For every detected person, we draw a rectangle around them so that the human movement is clearly highlighted.
#8th display the live video with highlighted boxes and also save it to a file so it can be viewed later
#9th The program keeps running until we press the ‘Q’ key on the keyboard, which stops the loop.
#10th close the webcam, save the video file properly, and close any windows opened by OpenCV