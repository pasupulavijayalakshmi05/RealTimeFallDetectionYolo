import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import os

MODEL_PATH = "runs/detect/train/weights/best.pt"
model = YOLO(MODEL_PATH)

root = tk.Tk()
root.title("Fall Detection - Image Prediction")
root.geometry("900x700")
image_label = tk.Label(root)
image_label.pack(pady=10)
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)
def upload_and_predict():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    results = model.predict(source=file_path, conf=0.3, save=False)
    img = cv2.imread(file_path)
    detected_classes = set()
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detected_classes.add(label)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((700, 500))
    img_tk = ImageTk.PhotoImage(img_pil)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    if detected_classes:
        result_label.config(
            text="Detected: " + ", ".join(detected_classes),
            fg="green",)
    else:
        result_label.config(
            text="No person detected",
            fg="red",)
upload_btn = tk.Button(
    root,
    text="Upload Image & Predict",
    command=upload_and_predict,
    font=("Arial", 14),
    bg="blue",
    fg="white",)
upload_btn.pack(pady=20)
root.mainloop()
