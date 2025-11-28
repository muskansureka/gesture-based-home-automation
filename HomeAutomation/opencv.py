import cv2 as cv
import time


def objects(frame):
    #cv.circle(frame, (150,150), 50, (0, 0, 255), cv.FILLED)
    #cv.line(frame, (300, 300), (400, 300), (0, 0, 255), thickness=2)
    #cv.rectangle(frame, (200, 200), (350, 250), (0, 255, 0), thickness=2)
    cv.putText(frame, "Name: " + "Sudip", (50, 135), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),thickness=5)


video = cv.VideoCapture(0)
pTime = cTime = 0

while True:
    _, frame = video.read() # '_' allocates less memory
    frame=cv.flip(frame,1)
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    objects(frame)

    cTime = time.time()
    fps = 1 // (cTime - pTime)
    pTime = cTime

    cv.putText(frame, "Fps"+str(fps), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), thickness=2)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break
    cv.imshow("LIVE", frame)
video.release()
cv.destroyAllWindows()