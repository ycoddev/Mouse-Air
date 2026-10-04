import cv2 as cv
import mediapipe as mp
import math
import subprocess

video = cv.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

prev_x = None
prev_y = None
speed = 6
clicked = False

while True:
    ret, frame = video.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand,side in zip(results.multi_hand_landmarks,results.multi_handedness):
            if side.classification[0].label == "Right":
                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )
                h, w, _ = frame.shape
                index = hand.landmark[8]
                index_x = int(index.x * w)
                index_y = int(index.y * h)
                cv.circle(frame, (index_x, index_y), 10, (0, 0, 0), cv.FILLED)
                if prev_x is not None:
                    dx = (index_x - prev_x) * speed
                    dy = (index_y - prev_y) * speed
                    subprocess.run([
                        "ydotool",
                        "mousemove",
                        "--",
                        str(int(dx)),
                        str(int(dy))
                    ])
                prev_x = index_x
                prev_y = index_y
            if side.classification[0].label == "Left":
                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )
                h, w, _ = frame.shape
                thumb = hand.landmark[4]
                index = hand.landmark[8]
                thumb_x = int(thumb.x * w)
                thumb_y = int(thumb.y * h)
                index_x = int(index.x * w)
                index_y = int(index.y * h)
                cv.circle(frame, (thumb_x, thumb_y), 8, (0, 255, 0), cv.FILLED)
                cv.circle(frame, (index_x, index_y), 8, (0, 0, 255), cv.FILLED)
                distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
                if distance < 20 and not clicked:
                    subprocess.run([
                        "ydotool",
                        "click",
                        "0xC0"
                    ])
                    clicked = True
                if distance > 30:
                    clicked = False
    cv.imshow("Air Mouse", frame)
    if cv.waitKey(1) == ord("q"):
        break
video.release()
cv.destroyAllWindows()