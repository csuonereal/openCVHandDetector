import cv2
import mediapipe as mp
import time



cap = cv2.VideoCapture(0)


mpHands = mp.solutions.hands
hands = mpHands.Hands()#this needs to rgb images so we will convert frame to rgb
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

run = True
while run:
    ret, frame = cap.read()
    rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgbImg)

    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for hand_lmarks in  results.multi_hand_landmarks:
            for id, lm in enumerate(hand_lmarks.landmark):
                #print(id,lm)#this gives the locations as a deciamal ratio value we can find exact location by multiplying height and width
                height, width, channel = frame.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                print(id ,cx, cy)
                if id == 4:
                    cv2.circle(frame, (cx,cy), 20,(255,0,0), cv2.FILLED)
            mpDraw.draw_landmarks(frame, hand_lmarks, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(frame, "FPS:"+str(+int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255), 1)

    cv2.imshow("frame",frame)
    if cv2.waitKey(1) == ord("q"):
        run = False

cap.release()
cv2.destroyAllWindows()