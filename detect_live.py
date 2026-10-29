
import cv2
from ultralytics import YOLO

# Load your trained PPE model
model = YOLO(r"runs/detect/runs/ppe/train/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    raise SystemExit

# PPE classes that count as detected equipment
ppe_keywords = ["helmet", "vest", "glove", "boot", "shoe"]

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    # Run detection
    results = model(frame, conf=0.35, verbose=False)
    result = results[0]

    # Store detected PPE boxes and confidence
    ppe_boxes = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id].lower()
        confidence = float(box.conf[0]) * 100

        # Ignore classes representing missing PPE
        if class_name.startswith("no_"):
            continue

        if any(word in class_name for word in ppe_keywords):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            ppe_boxes.append(
                (x1, y1, x2, y2, class_name, confidence)
            )

    # Find people detected in the frame
    person_boxes = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id].lower()

        if class_name == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            person_confidence = float(box.conf[0]) * 100
            person_boxes.append(
                (x1, y1, x2, y2, person_confidence)
            )

    # Draw one box around each person
    for px1, py1, px2, py2, person_confidence in person_boxes:
        detected_items = []

        # Check whether PPE box centers fall inside this person box
        for x1, y1, x2, y2, class_name, confidence in ppe_boxes:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if px1 <= center_x <= px2 and py1 <= center_y <= py2:
                detected_items.append(class_name)

        if detected_items:
            color = (0, 255, 0)  # Green
            label = f"PPE DETECTED | Person {person_confidence:.1f}%"
        else:
            color = (0, 0, 255)  # Red
            label = f"NO PPE DETECTED | Person {person_confidence:.1f}%"

        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 3)

        cv2.putText(
            frame,
            label,
            (px1, max(py1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # Draw individual PPE boxes with confidence percentages
    for x1, y1, x2, y2, class_name, confidence in ppe_boxes:
        label = f"{class_name} {confidence:.1f}%"

        # Blue box around each PPE item
        cv2.rectangle(
            frame, (x1, y1), (x2, y2), (255, 0, 0), 2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )

    cv2.imshow("Industrial PPE Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
