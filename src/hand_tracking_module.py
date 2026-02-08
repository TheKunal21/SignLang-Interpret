import cv2 as cv
import mediapipe as mp
import time

class hand_detector:
    def __init__(
        self,
        mode=False,
        maxhands=2,
        modelcomplexity=1,
        min_detectionconfidence=0.5,
        max_track_confidence=0.5,
    ):
        self.mode = mode
        self.maxhands = maxhands
        self.modelcomplexity = modelcomplexity
        self.min_detectionconfidence = min_detectionconfidence
        self.max_track_confidence = max_track_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxhands,
            model_complexity=self.modelcomplexity,
            min_detection_confidence=self.min_detectionconfidence,
            min_tracking_confidence=self.max_track_confidence,
        )
        self.mpsketch = mp.solutions.drawing_utils
        self.results = None

    def findhands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(self.results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpsketch.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img 

    def findposition(self, img, handno=0, draw=True):
        lmlist = []
        if self.results and self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)
        return lmlist

def main():
    Ptime = 0
    cap = cv.VideoCapture(0)
    detector = hand_detector()

    while True:
        success, img = cap.read()
        img = detector.findhands(img)
        lmlist = detector.findposition(img)
        if len(lmlist) != 0:
            print(lmlist[4])
        Ctime = time.time()
        fps = 1 / (Ctime - Ptime)
        Ptime = Ctime
        cv.putText(
            img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_TRIPLEX, 3, (0, 255, 0), 3
        )
        cv.imshow("image", img)
        if cv.waitKey(1) & 0xFF == ord("d"):
            break

if __name__ == "__main__":
    main()