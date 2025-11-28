import mediapipe as mp
import numpy as np
import cv2 as cv
import time
import math
import csv
import os
import sys

# === FUNCTIONS ===

def findHands(results, draw=True):
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            if draw:
                mpDraw.draw_landmarks(frame, handLms, mpHand.HAND_CONNECTIONS)
    return frame

def findPosition(frame, results, handNo=0, draw=True):
    lmList = []
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
            height, width, _ = frame.shape
            cx, cy = int(lm.x * width), int(lm.y * height)
            lmList.append([id, cx, cy])
            if draw:
                cv.circle(frame, (cx, cy), 10, (0, 0, 255), cv.FILLED)
    return lmList

def countFinger(lmList, tipIds, led=False):
    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[4][1] < lmList[3][1]:
            fingers.append(1)
            if led: simulateLight(4, mode=True)
        else:
            fingers.append(0)
            if led: simulateLight(4)

        # Other 4 fingers
        for key in tipIds.keys():
            if key > 4:
                if lmList[key][2] < lmList[key - 2][2]:
                    fingers.append(1)
                    if led: simulateLight(key, mode=True)
                else:
                    fingers.append(0)
                    if led: simulateLight(key)
        return fingers.count(1)
    return 0

def fingerLength(lmList, frame, servo=False):
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        x3, y3 = (x1 + x2) // 2, (y1 + y2) // 2

        if servo:
            cv.circle(frame, (x1, y1), 7, (0, 0, 255), cv.FILLED)
            cv.circle(frame, (x2, y2), 7, (0, 0, 255), cv.FILLED)
            cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            cv.circle(frame, (x3, y3), 7, (0, 0, 255), cv.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        servoDeg = int(np.interp(length, [50, 320], [0, 180]))

        if servo:
            print(f"[Simulated Servo] Distance: {length:.2f}, Angle: {servoDeg}")
            cv.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), thickness=2)
            servoBar = int(np.interp(length, [50, 320], [400, 150]))
            cv.rectangle(frame, (50, servoBar), (85, 400), (0, 255, 0), cv.FILLED)
            cv.putText(frame, "DEG " + str(servoDeg), (50, 135),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)

        if length < 50:
            cv.circle(frame, (x3, y3), 7, (0, 255, 0), cv.FILLED)

def simulateLight(finger_id, mode=False):
    state = "ON" if mode else "OFF"
    print(f"[Simulated LED] Finger {finger_id}: {state}")

# ✅ Action mapping function
def performAction(count):
    if count == 0:
        return "💡 Light OFF"
    elif count == 1:
        return "🌀 Fan ON"
    elif count == 4:
        return "🌀 Fan OFF"
    elif count == 5:
        return "💡 Light ON"
    else:
        return "Unknown Gesture"

# === CSV Setup ===
csv_file = "gesture_results.csv"

# Create file with header if not exists
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "gesture_actual", "gesture_detected", "detection_time"])

# Save result to CSV
def save_result(user_id, actual, detected, time_taken):
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, actual, detected, round(time_taken, 3)])

# === INIT ===
tipIds = {4: b'a', 8: b'b', 12: b'c', 16: b'd', 20: b'e'}
pTime = cTime = 0

video = cv.VideoCapture(0)
mpHand = mp.solutions.hands
hands = mpHand.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# ✅ Cooldown variables
last_action_time = 0
cooldown_active = False
cooldown_duration = 15  # seconds

# Auto-assign user ID
user_id = 1

try:
    while True:
        _, frame = video.read()
        frame = cv.flip(frame, 1)
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frameRGB)

        frame = findHands(results)
        lmList = findPosition(frame, results, draw=True)
        count = countFinger(lmList, tipIds, led=False)
        fingerLength(lmList, frame, servo=False)

        action_text = ""

        # ✅ Check cooldown
        current_time = time.time()
        if not cooldown_active and count in [0, 1, 4, 5]:
            start_time = time.time()
            action_text = performAction(count)
            print(f"Predicted Gesture: {count} → {action_text}")

            # Auto-label gesture as actual = detected
            actual = count
            detection_time = time.time() - start_time
            save_result(user_id, actual, count, detection_time)

            cooldown_active = True
            last_action_time = current_time

        elif cooldown_active:
            action_text = "⏳ Waiting for next gesture..."

        # ✅ Reset after cooldown
        if cooldown_active and (current_time - last_action_time) >= cooldown_duration:
            print("⏳ Ready for next gesture input...")
            cooldown_active = False

        cTime = time.time()
        fps = 1 // (cTime - pTime) if cTime != pTime else 0
        pTime = cTime

        # ✅ Display information on screen
        cv.putText(frame, "Press 'Q' to Quit", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv.putText(frame, f"FPS: {fps}", (10, 55),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.putText(frame, f"Predicted Gesture: {count}", (10, 85),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv.putText(frame, action_text, (10, 115),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv.rectangle(frame, (30, 425), (190, 470), (0, 0, 255), thickness=2)

        cv.imshow("LIVE", frame)
        if cv.waitKey(20) & 0xFF == ord('q'):
            print("✅ Exiting program safely...")
            break

except KeyboardInterrupt:
    print("\n⚠️ Program interrupted manually. Closing safely...")

finally:
    video.release()
    cv.destroyAllWindows()
    sys.exit(0)
