import cv2
import mediapipe as mp
import math

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Define thresholds
STABLE_THRESHOLD = 15  # frames
MOVEMENT_THRESHOLD = 0.14

# Gesture names and their actions
GESTURE_ACTIONS = {
    "FIVE": "Send",
    "FIST": "Select",
    "PEACE": "Cancel",
    "THUMB_UP": "Confirm",
    "OK": "Copy"
}

# Helper function to find distance
def calc_distance(a, b):
    return math.hypot(b.x - a.x, b.y - a.y)

# Recognize gesture
def recognize_gesture(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    total = fingers.count(1)

    if total == 5:
        return "FIVE"
    elif total == 0:
        return "FIST"
    elif fingers[1] and fingers[2] and not fingers[0] and not fingers[3] and not fingers[4]:
        return "PEACE"
    elif fingers[0] and not any(fingers[1:]):
        return "THUMB_UP"
    elif fingers[1] and fingers[0] and not any(fingers[2:]):
        return "OK"
    else:
        return "UNKNOWN"

# Main loop
cap = cv2.VideoCapture(0)
stable_gesture = None
stable_count = 0

print("Starting camera... Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    gesture_name = "UNKNOWN"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            gesture_name = recognize_gesture(handLms)

    # Stability check
    if gesture_name == stable_gesture:
        stable_count += 1
    else:
        stable_gesture = gesture_name
        stable_count = 0

    if stable_count > STABLE_THRESHOLD and stable_gesture in GESTURE_ACTIONS:
        action = GESTURE_ACTIONS[stable_gesture]
        cv2.putText(img, f"Action: {action}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    else:
        cv2.putText(img, f"Gesture: {gesture_name}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

    cv2.imshow("Hand Gesture Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
