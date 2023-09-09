import cv2
import mediapipe as mp
import math


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

    def find_positions(self, image, hand_num=0):  # For just one hand in image
        landmark_list = []

        if self.results.multi_hand_landmarks:
            my_hand_landmarks = self.results.multi_hand_landmarks[hand_num]  # Get landmarks of one particular hand

            for id, landmark in enumerate(my_hand_landmarks.landmark):  # Ergodic 21 landmarks
                h, w, c = image.shape  # Height, width and channel of image
                cx, cy = int(landmark.x * w), int(landmark.y * h)  # Compute location of landmark
                landmark_list.append([id, cx, cy])

        return landmark_list


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

    angle = vector_2d_angle(
        ((int(landmark_list[0][1]) - int(landmark_list[6][1])),
         (int(landmark_list[0][2]) - int(landmark_list[6][2]))),
        ((int(landmark_list[7][1]) - int(landmark_list[8][1])),
         (int(landmark_list[7][2]) - int(landmark_list[8][2])))
    )  # Index angle
    angle_list.append(angle)

    angle = vector_2d_angle(
        ((int(landmark_list[0][1]) - int(landmark_list[10][1])),
         (int(landmark_list[0][2]) - int(landmark_list[10][2]))),
        ((int(landmark_list[11][1]) - int(landmark_list[12][1])),
         (int(landmark_list[11][2]) - int(landmark_list[12][2])))
    )  # Middle angle
    angle_list.append(angle)

    angle = vector_2d_angle(
        ((int(landmark_list[0][1]) - int(landmark_list[14][1])),
         (int(landmark_list[0][2]) - int(landmark_list[14][2]))),
        ((int(landmark_list[15][1]) - int(landmark_list[16][1])),
         (int(landmark_list[15][2]) - int(landmark_list[16][2])))
    )  # Ring angle
    angle_list.append(angle)

    angle = vector_2d_angle(
        ((int(landmark_list[0][1]) - int(landmark_list[18][1])),
         (int(landmark_list[0][2]) - int(landmark_list[18][2]))),
        ((int(landmark_list[19][1]) - int(landmark_list[20][1])),
         (int(landmark_list[19][2]) - int(landmark_list[20][2])))
    )  # Pink angle
    angle_list.append(angle)

    return angle_list
