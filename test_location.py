import cv2
import mediapipe as mp
import time


class HandsDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):  # Same parameters as class Hands
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode, self.max_num_hands,
                                        self.model_complexity, self.min_detection_confidence,
                                        self.min_tracking_confidence)  # Generate hands image
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
        self.results = self.hands.process(imageRGB)  # Process RGB image to get hands information

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

        return image

    def find_positions(self, image, hand_num=0, draw=True):
        landmark_list = []

        if self.results.multi_hand_landmarks:
            my_hand_landmarks = self.results.multi_hand_landmarks[hand_num]  # Get landmarks of one particular hand

            for id, landmark in enumerate(my_hand_landmarks.landmark):  # Ergodic 21 landmarks
                h, w, c = image.shape  # Height, width and channel of image
                cx, cy = int(landmark.x * w), int(landmark.y * h)  # Compute location of landmark
                landmark_list.append([id, cx, cy])
                if draw:
                    if id == 4:
                        cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                    if id == 8:
                        cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                    if id == 12:
                        cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                    if id == 16:
                        cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                    if id == 20:
                        cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        return landmark_list


def main():
    pTime = 0  # Previous time
    cTime = 0  # Current time
    cap = cv2.VideoCapture(0)  # Generate webcam
    detector = HandsDetector()  # Generate hands detector

    while True:
        success, image = cap.read()
        image = detector.find_hands(image)
        landmark_list = detector.find_positions(image)

        if landmark_list:

            if (landmark_list[2][1] > landmark_list[4][1] and landmark_list[6][2] > landmark_list[8][2] and
                    landmark_list[10][2] < landmark_list[12][2] and landmark_list[14][2] < landmark_list[16][2] and
                    landmark_list[18][2] < landmark_list[20][2]):
                cv2.putText(image, "1", (500, 400), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8)

            if (landmark_list[2][1] > landmark_list[4][1] and landmark_list[6][2] > landmark_list[8][2] and
                    landmark_list[10][2] > landmark_list[12][2] and landmark_list[14][2] < landmark_list[16][2] and
                    landmark_list[18][2] < landmark_list[20][2]):
                cv2.putText(image, "2", (500, 400), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8)

            if (landmark_list[2][1] > landmark_list[4][1] and landmark_list[6][2] > landmark_list[8][2] and
                    landmark_list[10][2] > landmark_list[12][2] and landmark_list[14][2] > landmark_list[16][2] and
                    landmark_list[18][2] < landmark_list[20][2]):
                cv2.putText(image, "3", (500, 400), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8)

            if (landmark_list[2][1] > landmark_list[4][1] and landmark_list[6][2] > landmark_list[8][2] and
                    landmark_list[10][2] > landmark_list[12][2] and landmark_list[14][2] > landmark_list[16][2] and
                    landmark_list[18][2] > landmark_list[20][2]):
                cv2.putText(image, "4", (500, 400), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8)

            if (landmark_list[2][1] < landmark_list[4][1] and landmark_list[6][2] > landmark_list[8][2] and
                    landmark_list[10][2] > landmark_list[12][2] and landmark_list[14][2] > landmark_list[16][2] and
                    landmark_list[18][2] > landmark_list[20][2]):
                cv2.putText(image, "5", (500, 400), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8)

        cTime = time.time()
        fps = 1 / (cTime - pTime)  # Compute fps
        pTime = cTime

        cv2.putText(image, str(int(fps)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)  # Display fps on screen

        cv2.imshow("Image", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
