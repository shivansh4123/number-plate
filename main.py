import cv2

cap=cv2.VideoCapture("demo.mp4")

npCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

count=0

while True:
    success, frame = cap.read()
    frame = cv2.resize(frame, (640,480))
    if success:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        np = npCascade.detectMultiScale(frame_gray,1.1, 10)
        for x,y,w,h in np:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,255), 3)
            cv2.putText(frame, "number plate", (x, y-5), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2)
            frameROI=frame[y:y+h, x:x+w]
            cv2.imshow("output 2", frameROI)
        cv2.imshow("output",frame)

        if cv2.waitKey(1) & 0xFF==ord("p"):
            cv2.imwrite("plate/np"+str(count)+".jpg", frameROI)
            cv2.rectangle(frame, (0,200), (640,300), (0,255,0), cv2.FILLED)
            cv2.putText(frame, "scan saved", (150,265), cv2.FONT_HERSHEY_DUPLEX, 2, (255,0,0), 2)
            cv2.imshow("output", frame)
            cv2.waitKey(500)
            count+=1

    else:
        break

