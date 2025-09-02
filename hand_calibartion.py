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

width, height = 854, 480
target_points = [(100, 100), (754, 100), (754, 380), (100, 380)]
calibration_points = []

def capture_hand_landmarks():
    global calibration_points
    for i, point in enumerate(target_points):
        while True:
            calibration_image = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.circle(calibration_image, point, 20, (0, 255, 0), -1)

            cv2.namedWindow("Calibration Targets", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Calibration Targets", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Calibration Targets", calibration_image)

            ret, frame = cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            
            cv2.imshow("Camera View", frame)

            key = cv2.waitKey(1)
            if key == 13:
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:

                        x = int(hand_landmarks.landmark[8].x * frame.shape[1])
                        y = int(hand_landmarks.landmark[8].y * frame.shape[0])
                        calibration_points.append((x,y))
                        print(f"Captured landmark for point {i+1} at: ({x}, {y})")
                else:
                    print("Error")
                    continue
                break


capture_hand_landmarks()
cv2.destroyAllWindows()

if len(calibration_points) == 4:
    target_points_np = np.array(target_points, dtype=np.float32)
    calibration_points_np = np.array(calibration_points, dtype=np.float32)
    M, _ = cv2.findHomography(calibration_points_np, target_points_np)
    np.save("M.npy", M)
    print("Successful")
else:
    print("Error, points not calibrated")

