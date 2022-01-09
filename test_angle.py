import cv2
import mediapipe as mp
import time
import math
import HandTrackingModule as htm


def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle = math.degrees(math.acos(
            (v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))))
    except:
        angle = 65535.
    if angle > 180.:
        angle = 65535.
    return angle


def hand_angle(landmark_list):
    angle_list = []

    angle = vector_2d_angle(
        ((int(landmark_list[0][1]) - int(landmark_list[2][1])),
         (int(landmark_list[0][2]) - int(landmark_list[2][2]))),
        ((int(landmark_list[3][1]) - int(landmark_list[4][1])),
         (int(landmark_list[3][2]) - int(landmark_list[4][2])))
    )  # Thumb angle
    angle_list.append(angle)
    print(angle)

    angle = vector_2d_angle(
        ((int(landmark_list[0][1]) - int(landmark_list[6][1])),
         (int(landmark_list[0][2]) - int(landmark_list[6][2]))),
        ((int(landmark_list[7][1]) - int(landmark_list[8][1])),
         (int(landmark_list[7][2]) - int(landmark_list[8][2])))
    )  # Index angle
    angle_list.append(angle)
    print(angle)

    angle = vector_2d_angle(
        ((int(landmark_list[0][1]) - int(landmark_list[10][1])),
         (int(landmark_list[0][2]) - int(landmark_list[10][2]))),
        ((int(landmark_list[11][1]) - int(landmark_list[12][1])),
         (int(landmark_list[11][2]) - int(landmark_list[12][2])))
    )  # Middle angle
    angle_list.append(angle)
    print(angle)

    angle = vector_2d_angle(
        ((int(landmark_list[0][1]) - int(landmark_list[14][1])),
         (int(landmark_list[0][2]) - int(landmark_list[14][2]))),
        ((int(landmark_list[15][1]) - int(landmark_list[16][1])),
         (int(landmark_list[15][2]) - int(landmark_list[16][2])))
    )  # Ring angle
    angle_list.append(angle)
    print(angle)

    angle = vector_2d_angle(
        ((int(landmark_list[0][1]) - int(landmark_list[18][1])),
         (int(landmark_list[0][2]) - int(landmark_list[18][2]))),
        ((int(landmark_list[19][1]) - int(landmark_list[20][1])),
         (int(landmark_list[19][2]) - int(landmark_list[20][2])))
    )  # Pink angle
    angle_list.append(angle)
    print(angle)

    return angle_list


def h_gesture(angle_list):
    thr_angle = 65.
    thr_angle_thumb = 53.
    thr_angle_s = 49.
    gesture_str = None
    if 65535. not in angle_list:
        if (angle_list[0] > thr_angle_thumb) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
            gesture_str = "fist"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s):
            gesture_str = "five"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
            gesture_str = "gun"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s):
            gesture_str = "love"
        elif (angle_list[0] > 5) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
            gesture_str = "one"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s):
            gesture_str = "six"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (
                angle_list[2] < thr_angle_s) and (angle_list[3] < thr_angle_s) and (angle_list[4] > thr_angle):
            gesture_str = "three"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
            gesture_str = "thumbUp"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (
                angle_list[2] < thr_angle_s) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
            gesture_str = "yeah"
    return gesture_str


def main():
    pTime = 0  # Previous time
    cTime = 0  # Current time
    cap = cv2.VideoCapture(0)  # Generate webcam
    detector = htm.HandsDetector()  # Generate hands detector

    while True:
        success, image = cap.read()
        image = detector.find_hands(image)
        landmark_list = detector.find_positions(image)

        if landmark_list:
            angle_list = hand_angle(landmark_list)
            gesture_str = h_gesture(angle_list)
            cv2.putText(image, gesture_str, (0, 100), 0, 1.3, (0, 0, 255), 3)

        cTime = time.time()
        fps = 1 / (cTime - pTime)  # Compute fps
        pTime = cTime

        cv2.putText(image, str(int(fps)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)  # Display fps on screen

        cv2.imshow("Image", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
