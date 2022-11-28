import numpy as np
import cv2 as cv
# import time
import matplotlib.pyplot as plt
import warnings

# turn off warnings
warnings.filterwarnings("ignore")

def ransac(data, n_samples, threshold, ):
    best_number_inliners = 0
    best_line = np.array(2)
    #best_points = []
    for n in range(n_samples):
        rand_1= np.random.randint(0,len(data), size=1)
        rand_2 = np.random.randint(0,len(data), size=1) 
        sample_1 = data[int(rand_1)] 
        sample_2 = data[int(rand_2)]
        line = np.polyfit(sample_1,sample_2, 1)
        number_inliners = 0
        for point in data:
            dist = abs(line[0] * point[0] + line[1] - point[1]) / np.sqrt(np.square(line[0]) +1)
            if dist < threshold: 
                number_inliners += 1
        
        if number_inliners > best_number_inliners:
            best_number_inliners = number_inliners
            best_line = line
            #best_points = [sample_1[0],sample_1[1],sample_2[0],sample_2[1]]
    #print(best_points)
        
    return best_line

def ransac_4_lines(data, n_samples, threshold, ):
    best_number_inliners = 0
    best_line = np.array(2)
    #best_points = []
    # new_data = data
    # size_data = len(data)
    # print(size_data)
    for n in range(n_samples):
        rand_1= np.random.randint(0,len(data), size=1)
        rand_2 = np.random.randint(0,len(data), size=1)  
        sample_1 = data[int(rand_1)] 
        sample_2 = data[int(rand_2)]
        line = np.polyfit(sample_1,sample_2, 1)
        number_inliners = 0
        for point in data:
            dist = abs(line[0] * point[0] + line[1] - point[1]) / np.sqrt(np.square(line[0]) +1)
            if dist < threshold: 
                number_inliners += 1
                
            
        if number_inliners > best_number_inliners:
            best_number_inliners = number_inliners
            best_line = line
            #best_points = [sample_1[0],sample_1[1],sample_2[0],sample_2[1]]
    #print(best_points)
    for point in data:
            dist = abs(best_line[0] * point[0] + best_line[1] - point[1]) / np.sqrt(np.square(best_line[0]) +1)
            if dist < threshold: 
                data.remove(point)

    return best_line, data
# cap = cv.VideoCapture(0)
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
# time_before = 0
# time_after = 0
# fps = 0
# font = cv.FONT_HERSHEY_SIMPLEX



# while True:
#     max_value = 0
#     Capture frame-by-frame
#     ret, frame = cap.read()
#     e1 = cv.getTickCount()
#     time_before = time.time()
#     if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     Our operations on the frame come here
#     edge = cv.Canny(frame,100,180)
#     x,y = np.where(edge)
#     list_edge = []
#     for i in range(len(x)):
#         list_edge.append((x[i],y[i]))
#     print(list_edge)

    
 

#     plt.scatter(y,x)
#     plt.plot(line_X, line_y_ransac, color='yellow')
#     plt.show()

#     cv.line(frame,(,),(,),(0,0,255),9)
#     Display the resulting frame
#     cv.imshow('frame', frame)

#     if cv.waitKey(1) == ord('q'):
#         break
# When everything done, release the capture
# cap.release()
# cv.destroyAllWindows()