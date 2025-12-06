import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
import threading
import time
from collections import deque, Counter
from tensorflow import keras
import win32com.client

# ------------------ PATHS ------------------
MODEL_PATH = r"project\project\model\sign_lstm_right_part2morecc.h5"
LABELS_PATH = r"C:\Users\VICTUS\Desktop\project\project\project\data\labels.csv"

# ------------------ PARAMETERS --------------
FPS_THROTTLE = 4
SMOOTH_K = 12
SMOOTH_MIN_VOTES = 7
CONF_THRESHOLD = 0.80
COOLDOWN_MS = 1000

# Speech engine
speaker = win32com.client.Dispatch("SAPI.SpVoice")
sapi_lock = threading.Lock()


def speak_sapi_async(text):
    def run():
        with sapi_lock:
            speaker.Speak(text)
    threading.Thread(target=run, daemon=True).start()


def normalize_frame(lm):
    """
    Same as training normalize_sample:
    21 landmarks -> (x,y) -> subtract wrist -> / max_abs -> flatten (42,)
    """
    pts = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)  # (21,2)
    wrist = pts[0]
    pts = pts - wrist
    max_val = np.max(np.abs(pts))
    if max_val != 0:
        pts = pts / max_val
    return pts.flatten().astype("float32")   # (42,)


# ------------------ MAIN DETECTION FUNCTION ------------------
def start_sign_detection(video_label_widget, output_text_widget):
    """
    video_label_widget -> CTkLabel where video feed is displayed
    output_text_widget -> CTkTextbox where predicted text is shown
    """

    print("Loading sign detection model...")
    model = keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = [l.strip() for l in f if l.strip()]

    num_features = model.input_shape[1]
    num_classes = model.output_shape[1]
    print("Model expects features:", num_features)
    print("Model outputs classes:", num_classes, "| labels:", len(labels))

    if len(labels) != num_classes:
        print("WARNING: labels count != model output units. Check labels.csv")

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # Open Webcam
    cap = cv2.VideoCapture(0)

    pred_buffer = deque(maxlen=SMOOTH_K)

    typed = ""
    last_label = None
    last_pred_time = 0
    last_accept_time_ms = 0

    # --------------- BACKSPACE HANDLER ---------------
    def on_backspace(event):
        nonlocal typed, last_label, pred_buffer

        if typed:
            typed = typed[:-1]
            output_text_widget.delete("1.0", "end")
            output_text_widget.insert("1.0", typed)

        last_label = None
        pred_buffer.clear()
        return "break"

    output_text_widget.bind("<BackSpace>", on_backspace)
    output_text_widget.focus_set()

    # --------------- UPDATE LOOP ---------------
    def update():
        nonlocal typed, last_label, last_pred_time, last_accept_time_ms

        ret, frame = cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        now = time.time()

        if res.multi_hand_landmarks and res.multi_handedness:
            # use right hand (must match how you collected training data)
            used_lm = None
            for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                if handed.classification[0].label == "Right":
                    used_lm = lm
                    break

            if used_lm is not None:
                mp_draw.draw_landmarks(frame, used_lm, mp_hands.HAND_CONNECTIONS)

                if now - last_pred_time >= 1.0 / FPS_THROTTLE:
                    last_pred_time = now

                    flat = normalize_frame(used_lm)             # (42,)
                    x = flat.reshape(1, num_features, 1)        # (1, 42, 1)

                    probs = model.predict(x, verbose=0)[0]      # (num_classes,)
                    idx = int(np.argmax(probs))
                    conf = float(probs[idx])

                    pred_buffer.append((idx, conf))

                    votes = [p[0] for p in pred_buffer]
                    top_label, top_votes = Counter(votes).most_common(1)[0]
                    avg_conf = np.mean([p[1] for p in pred_buffer if p[0] == top_label])
                    now_ms = int(time.time() * 1000)

                    if (top_votes >= SMOOTH_MIN_VOTES and
                        avg_conf >= CONF_THRESHOLD and
                        (now_ms - last_accept_time_ms) >= COOLDOWN_MS):

                        detected = labels[top_label]

                        if detected != last_label:
                            if detected.lower() == "space":
                                typed += " "
                            else:
                                typed += detected

                            output_text_widget.delete("1.0", "end")
                            output_text_widget.insert("1.0", typed)

                            speak_sapi_async(typed)

                        last_label = detected
                        last_accept_time_ms = now_ms
                        pred_buffer.clear()
        else:
            # no hand – clear prediction buffer
            pred_buffer.clear()
            last_label = None

        # ---- Convert frame → Tkinter ----
        color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(color_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label_widget.imgtk = imgtk
        video_label_widget.configure(image=imgtk)

        video_label_widget.after(10, update)

    update()
