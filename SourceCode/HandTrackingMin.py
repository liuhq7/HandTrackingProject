import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # Generate webcam

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # Generate hands image
mpDraw = mp.solutions.drawing_utils

pTime = 0  # Previous time
cTime = 0  # Current time

while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
    results = hands.process(imageRGB)  # Process RGB image to get hands information

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, landmark in enumerate(hand_landmarks.landmark):  # Ergodic 21 landmarks
                print(id, landmark)
                h, w, c = image.shape  # Height, width and channel of image
                cx, cy = int(landmark.x * w), int(landmark.y * h)  # Compute location of landmark
                print(cx, cy)

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

            mpDraw.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)  # Compute fps
    pTime = cTime

    cv2.putText(image, str(int(fps)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)  # Display fps on screen

    cv2.imshow("Image", image)
    cv2.waitKey(1)
