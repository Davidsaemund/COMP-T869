import numpy as np
import cv2 as cv
import time
cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
time_before = 0
time_after = 0
fps = 0
font = cv.FONT_HERSHEY_SIMPLEX
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])

while True:
    max_value = 0
    # Capture frame-by-frame
    ret, frame = cap.read()
    #e1 = cv.getTickCount()
    time_before = time.time()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #Find the brightest spot
    #(minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)

    # Step 6 for loop
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i,j] >= max_value :
                max_value = gray[i,j]
                maxLoc = (i,j)

    # Brightest circle
    cv.circle(frame, maxLoc, 7, (255,255,255),2)


    #Reddest circle
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_red,upper_red)
    res = cv.bitwise_and(frame,frame, mask= mask)
    (redminVal, redmaxVal, redminLoc, redmaxLoc) = cv.minMaxLoc(res[:,:,1])
    cv.circle(frame, redmaxLoc, 7, (0,0,255),2)

    #Calculate and display FPS 
    fps = 1.0 / (time_before-time_after)
    fps = int(fps)
    fps = str(fps)
    time_after = time_before
    cv.putText(frame, fps, (7, 70), font, 3, (255, 255, 255), 3, cv.LINE_AA)
    print(fps)
    # Display the resulting frame
    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()