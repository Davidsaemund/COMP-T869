import numpy as np
import cv2 
import time
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
time_before = 0
time_after = 0
fps = 0
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
font = cv2.FONT_HERSHEY_SIMPLEX
classesFile = 'coco.names'
classNames = []
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

modelConfiguration = 'yolov3-tiny.cfg'
modelWeigths = 'yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeigths)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



def findObejects(outputs,frame):
    hT, wT, cT = frame.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:    
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT) - w/2), int((det[1]*hT)- h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    #print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    #print(indices[0])
    for i in indices:
        #i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),2)
    for i in indices:
        #i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),2)
        class_name = classNames[classIds[i]].upper()
        conf_value = int(confs[i]*100)
        cv2.putText(frame,f'{class_name} {conf_value}%',(x,y-10),font,0.6,(255,0,255),2)


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
    blob = cv2.dnn.blobFromImage(frame, 1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    #print(layerNames)
    #layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    # for i in net.getUnconnectedOutLayers():
    #     outputNames.append(layerNames[i-1])
    #print(net.getUnconnectedOutLayers())
    #print(outputNames)

    outputs = net.forward(outputNames)
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])
    findObejects(outputs,frame)




    #Calculate and display FPS 
    fps = 1.0 / (time_before-time_after)
    fps = int(fps)
    fps = str(fps)
    time_after = time_before
    cv2.putText(frame, fps, (7, 70), font, 3, (255, 255, 255), 3, cv2.LINE_AA)
    #print(fps)
    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()