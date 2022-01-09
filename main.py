import cv2
import time
import HandTrackingModule as htm


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
            print(landmark_list[2])

        cTime = time.time()
        fps = 1 / (cTime - pTime)  # Compute fps
        pTime = cTime

        cv2.putText(image, "FPS: " + str(int(fps)), (20, 40),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)  # Display fps on screen

        cv2.imshow("Image", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
