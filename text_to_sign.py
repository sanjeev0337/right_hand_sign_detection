# text_to_sign.py
import os
from PIL import Image, ImageTk
import customtkinter as ctk

SIGN_IMAGE_FOLDER = r"C:\Users\VICTUS\Desktop\project\project\sign_images"
  # folder with a.jpg, b.jpg, space.jpg ...

def show_sign_images(text, parent_frame):
    """
    Show sign images horizontally (left-to-right) with horizontal scroll.
    """

    # Clear old images
    for widget in parent_frame.winfo_children():
        widget.destroy()

    text = text.lower()

    # Create a horizontal inner frame inside scrollable frame
    inner = ctk.CTkFrame(parent_frame)
    inner.pack(anchor="w")

    for ch in text:

        if ch == " ":
            img_name = "space.JPG"
        else:
            img_name = f"{ch}.JPG"

        img_path = os.path.join(SIGN_IMAGE_FOLDER, img_name)

        if not os.path.exists(img_path):
            print("Missing image:", img_path)
            continue

        img = Image.open(img_path).resize((150, 150))
        img_tk = ImageTk.PhotoImage(img)

        lbl = ctk.CTkLabel(inner, text="", image=img_tk)
        lbl.image = img_tk
        lbl.pack(side="left", padx=10)
