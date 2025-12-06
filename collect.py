#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2, csv, os, itertools, copy
import mediapipe as mp
import numpy as np

SAVE_PATH = "project/data/keypoints.csv"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

current_label = None   # None = not capturing


def normalize(lm):
    temp = copy.deepcopy(lm)
    base_x, base_y = temp[0]
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y
    temp = list(itertools.chain.from_iterable(temp))
    m = max(map(abs, temp))
    return [v/m if m!=0 else 0 for v in temp]


def label_from_key(key):
    c = chr(key)
    if c.isdigit(): return int(c)
    if 'a'<=c<='z': return 10+(ord(c)-97)
    return None


def main():
    global current_label
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(max_num_hands=2,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.5) as hands:

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break  # ESC

            # toggle capture state
            if (ord('0') <= key <= ord('9')) or (ord('a') <= key <= ord('z')):
                new_label = label_from_key(key)
                if current_label == new_label:
                    print("STOP capturing")
                    current_label = None
                else:
                    print("START capturing:", new_label)
                    current_label = new_label

            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            # show landmarks
            if res.multi_hand_landmarks:
                for handlms in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, handlms, mp_hands.HAND_CONNECTIONS)

            # show capture status
            if current_label is not None:
                cv2.putText(frame, f"CAPTURING: {current_label}",
                            (10,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            cv2.imshow("Collect", frame)

            # write sample if capturing
            if current_label is not None and res.multi_hand_landmarks:
                left=[0]*42; right=[0]*42
                for h,hd in zip(res.multi_hand_landmarks,res.multi_handedness):
                    name = hd.classification[0].label
                    pts = normalize([[lm.x,lm.y] for lm in h.landmark])
                    if name=="Left": left=pts
                    else: right=pts
                row=[current_label]+left+right
                with open(SAVE_PATH,"a",newline="") as f:
                    csv.writer(f).writerow(row)

    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main() 