import cv2, csv, os, itertools, copy
import mediapipe as mp

INPUT_DIR = r"C:\Users\VICTUS\Desktop\project\ASL_Dataset\Train"
OUTPUT_CSV = r"C:\Users\VICTUS\Desktop\project\data\letters.csv"

mp_hands = mp.solutions.hands

# your label list matching labels.csv you gave me
LABELS = [
    "a","b","c","d","e","f","g","h","i","j",
    "k","l","m","n","o","p","q","r","s","space",
    "t","u","v","w","x","y","z"
]

def normalize_landmarks(lm):
    temp = copy.deepcopy(lm)
    base_x, base_y = temp[0]
    for i in range(len(temp)):
        temp[i][0] -= base_x
        temp[i][1] -= base_y
    flat = list(itertools.chain.from_iterable(temp))
    m = max(map(abs, flat)) if flat else 1
    return [v/m for v in flat]   # 42 floats

def label_to_id(name:str):
    name = name.lower()
    if name not in LABELS:
        raise ValueError(f"folder '{name}' not found in LABELS list")
    return LABELS.index(name)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        for cls in os.listdir(INPUT_DIR):
            cls_path = os.path.join(INPUT_DIR, cls)
            if not os.path.isdir(cls_path): continue

            label_id = label_to_id(cls)  # << MATCHES labels.csv EXACTLY

            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                img = cv2.imread(img_path)
                if img is None: continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)

                left = [0.0]*42
                right= [0.0]*42

                if res.multi_hand_landmarks and res.multi_handedness:
                    for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                        hand = hd.classification[0].label
                        pts  = normalize_landmarks([[p.x,p.y] for p in lm.landmark])
                        if hand=="Left": left=pts
                        else:            right=pts

                row = [label_id] + left + right
                writer.writerow(row)
                print("saved:", cls, label_id, img_path)
