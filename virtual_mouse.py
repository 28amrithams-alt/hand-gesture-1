import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize mediapipe and screen
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
screen_w, screen_h = pyautogui.size()

# Variables for smooth movement
prev_x, prev_y = 0, 0
smoothening = 5
last_click_time = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            index_finger_tip = lm[8]
            middle_finger_tip = lm[12]
            index_finger_pip = lm[6]
            middle_finger_pip = lm[10]

            # Coordinates
            ix, iy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            mx, my = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)

            # Convert to screen space
            screen_x = screen_w * index_finger_tip.x
            screen_y = screen_h * index_finger_tip.y

            # Smooth movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Draw points
            cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1)
            cv2.circle(frame, (mx, my), 8, (0, 0, 255), -1)

            # Check if both fingers are up
            index_up = index_finger_tip.y < index_finger_pip.y
            middle_up = middle_finger_tip.y < middle_finger_pip.y

            # Distance between finger tips
            distance = math.hypot(mx - ix, my - iy)

            # Click condition: both fingers up and close enough
            if index_up and middle_up and distance < 40:
                if time.time() - last_click_time > 0.7:
                    pyautogui.click()
                    last_click_time = time.time()
                    cv2.putText(frame, "Click", (ix, iy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Move", (ix, iy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
