import cv2
import time
import HandTrackingModule as htm

pTime = 0  # Previous time
cTime = 0  # Current time

cam_width = 648
cam_height = 480

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)  # Set the width of camera
cap.set(4, cam_height)  # Set the height of camera

detector = htm.HandsDetector()  # Generate hands detector

while True:
    success, image = cap.read()
    image = detector.find_hands(image, draw=False)
    landmark_list = detector.find_positions(image)

    if landmark_list:
        angle_list = htm.hand_angle(landmark_list)

        if (landmark_list[3][2] > landmark_list[4][2] > landmark_list[18][2] > landmark_list[14][2] and
                angle_list[0] < 50 and angle_list[1] > 100 and angle_list[2] > 100 and angle_list[3] > 100 and
                angle_list[4] > 100):
            cv2.putText(image, "O", (500, 400), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8)

        if (landmark_list[19][2] > landmark_list[18][2] > landmark_list[14][2] >
                landmark_list[10][2] > landmark_list[4][2] and landmark_list[15][2] > landmark_list[14][2] and
                landmark_list[17][2] > landmark_list[13][2] > landmark_list[9][2] and angle_list[0] < 50 and
                angle_list[1] < 20 and angle_list[2] < 20 and angle_list[3] > 150 and angle_list[4] > 150):
            cv2.putText(image, "K", (500, 400), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8)

        if (landmark_list[6][2] > landmark_list[5][2] and landmark_list[10][2] > landmark_list[9][2] and
                landmark_list[14][2] > landmark_list[13][2] and landmark_list[18][2] > landmark_list[17][2] and
                angle_list[0] > 150 and angle_list[1] < 100 and angle_list[2] < 100 and angle_list[3] < 50 and
                angle_list[4] < 100):
            cv2.putText(image, "YES", (400, 400), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8)

        if (landmark_list[8][2] > landmark_list[4][2] and landmark_list[12][2] > landmark_list[4][2] and
                landmark_list[15][2] > landmark_list[14][2] and landmark_list[19][2] > landmark_list[18][2] and
                angle_list[0] < 100 and angle_list[1] < 100 and angle_list[2] < 100 and angle_list[3] > 100 and
                angle_list[4] > 100):
            cv2.putText(image, "NO", (450, 400), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 8)

    cTime = time.time()
    fps = 1 / (cTime - pTime)  # Compute fps
    pTime = cTime

    cv2.putText(image, "FPS: " + str(int(fps)), (20, 40),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)  # Display fps on screen

    cv2.imshow("Image", image)
    cv2.waitKey(1)
