import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from sign_detection import start_sign_detection
from text_to_sign import show_sign_images
import win32com.client
import threading


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Sign Language App")
app.geometry("1200x720")

# ✅ allow maximize/minimize
app.resizable(True, True)


# ---------------- PAGE SWITCHING ----------------
def show_frame(frame):
    frame.tkraise()


container = ctk.CTkFrame(app)
container.pack(fill="both", expand=True)
container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)

home_page = ctk.CTkFrame(container)
sign_page = ctk.CTkFrame(container)
tts_page = ctk.CTkFrame(container)
text_to_sign_page = ctk.CTkFrame(container)

for page in (home_page, sign_page, tts_page, text_to_sign_page):
    page.grid(row=0, column=0, sticky="nsew")


# ---------------------------------------------------------
# UNIVERSAL BACK BUTTON (TOP LEFT)
# ---------------------------------------------------------
def add_back_button(parent_page):
    back_btn = ctk.CTkButton(
        parent_page,
        text="⬅ Back",
        width=120,
        height=45,
        fg_color="#333333",
        command=lambda: show_frame(home_page)
    )
    back_btn.place(x=20, y=20)


# ---------------------------------------------------------
# HOME PAGE — VERTICAL BUTTONS
# ---------------------------------------------------------

ctk.CTkLabel(home_page, text="Sign Language Communication App",
             font=("Arial", 42, "bold")).pack(pady=40)

# Vertical alignment
ctk.CTkButton(home_page, text="✋  Sign → Text",
              width=380, height=100, font=("Arial", 26, "bold"),
              fg_color="#8c4dff",
              command=lambda: show_frame(sign_page)).pack(pady=20)

ctk.CTkButton(home_page, text="🔤 Text → Sign Images",
              width=380, height=100, font=("Arial", 26, "bold"),
              fg_color="#ff8c00",
              command=lambda: show_frame(text_to_sign_page)).pack(pady=20)

ctk.CTkButton(home_page, text="🔊 Text → Speech",
              width=380, height=100, font=("Arial", 26, "bold"),
              fg_color="#1dbf73",
              command=lambda: show_frame(tts_page)).pack(pady=20)


# ---------------------------------------------------------
# PAGE 1 — SIGN → TEXT PAGE
# ---------------------------------------------------------

add_back_button(sign_page)

ctk.CTkLabel(sign_page, text="Sign → Text (Live Detection)",
             font=("Arial", 32, "bold")).pack(pady=70)

# 🟣 Camera placement UP and Bigger
video_label = ctk.CTkLabel(sign_page, text="")
video_label.pack(pady=5)   # smaller padding → moves up

output_box = ctk.CTkTextbox(sign_page, width=800, height=150, font=("Arial", 18))
output_box.pack(pady=20)


def start_detection_thread():
    threading.Thread(target=start_sign_detection,
                     args=(video_label, output_box),
                     daemon=True).start()


ctk.CTkButton(sign_page, text="Start Detection",
              width=300, height=70, fg_color="#8c57f0",
              font=("Arial", 26),
              command=start_detection_thread).pack(pady=10)


# ---------------------------------------------------------
# PAGE 2 — TEXT → SIGN IMAGES PAGE
# ---------------------------------------------------------

add_back_button(text_to_sign_page)

ctk.CTkLabel(text_to_sign_page, text="Text → Sign Images",
             font=("Arial", 32, "bold")).pack(pady=70)

text_box = ctk.CTkTextbox(text_to_sign_page, width=700, height=100, font=("Arial", 20))
text_box.pack(pady=10)

# Horizontal scroll frame for sign images
sign_frame = ctk.CTkScrollableFrame(
    text_to_sign_page, width=850, height=200, orientation="horizontal"
)
sign_frame.pack(pady=15)


def convert_to_signs():
    text = text_box.get("1.0", tk.END).strip()
    if text:
        show_sign_images(text, sign_frame)
    else:
        messagebox.showwarning("Empty", "Please type something!")


ctk.CTkButton(text_to_sign_page, text="Convert to Signs",
              fg_color="#ff8c00", width=300, height=70,
              font=("Arial", 26),
              command=convert_to_signs).pack(pady=15)


# ---------------------------------------------------------
# PAGE 3 — TEXT → SPEECH PAGE
# ---------------------------------------------------------

add_back_button(tts_page)

ctk.CTkLabel(tts_page, text="Text → Speech",
             font=("Arial", 32, "bold")).pack(pady=70)

tts_input = ctk.CTkTextbox(tts_page, width=700, height=200, font=("Arial", 18))
tts_input.pack(pady=20)


def speak_tts():
    text = tts_input.get("1.0", tk.END).strip()
    if text:
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Speak(text)
    else:
        messagebox.showwarning("Empty", "Write something first")


ctk.CTkButton(tts_page, text="Speak",
              width=250, height=70, fg_color="#1dbf73",
              font=("Arial", 26),
              command=speak_tts).pack(pady=10)


# ---------------- SHOW START PAGE ----------------
show_frame(home_page)
app.mainloop()
