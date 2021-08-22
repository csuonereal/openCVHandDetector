import cv2
import mediapipe as mp
import time



class HandDetector:
    def __init__(self, static_mode=False,maxHands=2, detectionConfident=0.5, trackConfident=0.5):
        self.static_mode = static_mode
        self.maxHands = maxHands
        self.detectionConfident = detectionConfident
        self.trackConfident = trackConfident
   
        self.pTime = 0
        self.cTime = 0

    def show_fps(self,frame):
        self.cTime = time.time()
        self.fps = 1/(self.cTime-self.pTime)
        self.pTime = self.cTime


    def find_hands(self,results,mpDraw,mpHands,frame):
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
        self.show_fps(frame)
        cv2.putText(frame, "FPS:"+str(+int(self.fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255), 1)
  

def main():
    hand_detector = HandDetector()
    cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()#this needs to rgb images so we will convert frame to rgb
    mpDraw = mp.solutions.drawing_utils

    run = True
    while run:
        ret, frame = cap.read()
        rgbImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgbImg)
        hand_detector.find_hands(results,mpDraw,mpHands,frame)

        cv2.imshow("frame",frame)
        if cv2.waitKey(1) == ord("q"):
            run = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()