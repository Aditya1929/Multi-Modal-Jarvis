import cv2
import mediapipe as mp
import pyautogui as pag
import numpy as np

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode = False,
                       max_num_hands=2,
                       min_detection_confidence=0.1,
                       min_tracking_confidence=0.1
                       )

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

M = np.load("M.npy")

width, height = 854, 480



while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    output_image = np.zeros((height, width, 3), np.uint8)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            landmark_choords = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                landmark_choords.append([x,y])
            
            landmark_choords = np.array(landmark_choords, dtype=np.float32)

            transformed_choords = cv2.perspectiveTransform(np.array([landmark_choords]), M)[0]

            for i, (x,y) in enumerate(transformed_choords):
                cv2.circle(output_image, (int(x), int(y)), 5, (0, 255,0), -1)

    cv2.namedWindow("Final Image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Final Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Final Image", output_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



