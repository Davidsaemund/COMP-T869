import numpy as np
import cv2 as cv
import time
from ransac import ransac
cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
time_before = 0
time_after = 0
fps = 0
font = cv.FONT_HERSHEY_SIMPLEX


while True:
    try: 
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
        width = int(frame.shape[1]* 0.5)
        height = int(frame.shape[0]* 0.5)
        dsize = (width,height)

        new_frame = cv.resize(frame,dsize)
        gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray,5)
        edge = cv.Canny(gray,100,180)
        y,x = np.where(edge == 255)
        list_edge = []
        for i in range(len(x)):
            list_edge.append((x[i],y[i]))
        #print(list_edge)   
        best_line = ransac(list_edge, 100, 1)
        #print(ransac_1)
        # point_1 = (points[0], points[1])
        # point_2 = (points[2], points[3])
        # plt.scatter(y,x)
        # plt.plot(line_X, line_y_ransac, color='yellow')
        # plt.show()
        x = np.array([0, new_frame.shape[1]])
        y = best_line[0] * x + best_line[1]
 
        cv.line(new_frame,(int(x[0]), int(y[0])), (int(x[1]), int(y[1])),(0,0,255),4)

        
        

        #Calculate and display FPS 
        fps = 1.0 / (time_before-time_after)
        fps = int(fps)
        fps = str(fps)
        time_after = time_before
        cv.putText(new_frame, fps, (7, 70), font, 3, (255, 255, 255), 3, cv.LINE_AA)
        
        # Display the resulting frame
        cv.imshow('frame', new_frame)
        cv.imshow('edge', edge)
        if cv.waitKey(1) == ord('q'):
            break
    except:
        pass
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()