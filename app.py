from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
GRAPH_FOLDER = "static/yolo_graphs"
MODEL_PATH = "runs/detect/train/weights/best.pt"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = YOLO(MODEL_PATH)
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_image = None
    detected = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = str(uuid.uuid4()) + ".jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            img = cv2.imread(filepath)
            results = model.predict(img, conf=0.3, verbose=False)
            classes = set()
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])
                    classes.add(label)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,)
            out_path = os.path.join(UPLOAD_FOLDER, "pred_" + filename)
            cv2.imwrite(out_path, img)
            prediction_image = "/" + out_path.replace("\\", "/")
            detected = ", ".join(classes) if classes else "No detection"
    return render_template(
        "index.html",
        prediction_image=prediction_image,
        detected=detected,)
if __name__ == "__main__":
    app.run(debug=True)
