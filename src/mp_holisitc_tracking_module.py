import cv2 as cv
import mediapipe as mp
import time

class HolisticDetector:
    def __init__(
        self,
        mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        detectionCon=0.5,
        trackCon=0.5,
    ):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHolistic = mp.solutions.holistic
        self.holistic = self.mpHolistic.Holistic(
            static_image_mode=self.mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHolistic(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.holistic.process(imgRGB)

        if draw:
            if self.results.pose_landmarks:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpHolistic.POSE_CONNECTIONS
                )
            if self.results.face_landmarks:
                self.mpDraw.draw_landmarks(
                    img, self.results.face_landmarks, self.mpHolistic.FACEMESH_TESSELATION
                )
            if self.results.left_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    img, self.results.left_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS
                )
            if self.results.right_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    img, self.results.right_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS
                )
        return img

    def findHandPositions(self, img, hand='right', draw=True):
        lmlist = []
        hand_landmarks = (
            self.results.right_hand_landmarks if hand == 'right' else self.results.left_hand_landmarks
        )

        if hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        return lmlist

def main():
    pTime = 0
    cap = cv.VideoCapture(0)
    detector = HolisticDetector()

    while True:
        success, img = cap.read()
        img = detector.findHolistic(img)
        lmlist = detector.findHandPositions(img, hand='right')
        if len(lmlist) != 0:
            print("Right Hand Tip (id=4):", lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, f"FPS: {int(fps)}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv.imshow("Holistic Detection", img)
        if cv.waitKey(1) & 0xFF == ord('d'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
