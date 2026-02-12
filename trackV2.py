from collections import defaultdict
import cv2
import numpy as np

from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel


# -----------------------
# YOLO + SAHI setup
# -----------------------
model_path = r"C:\Users\cpear\Documents\VisualAnalysis\runs\detect\train10\weights\best.pt"

# SAHI-wrapped model (used for sliced inference)
sahi_model = UltralyticsDetectionModel(
    model_path=model_path,
    confidence_threshold=0.50,
    device="cuda",
)

# Native YOLO model just for class names, etc.
yolo_model = YOLO(model_path)

# --- find squirrel class id ---
squirrel_id = None
for cid, name in yolo_model.names.items():
    if "squirrel" in name.lower():
        squirrel_id = cid
        break

print("Squirrel class id:", squirrel_id)

# -----------------------
# Video I/O
# -----------------------
video_path = r"C:\Users\cpear\Documents\VisualAnalysis\squirrelmapping.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

cv2.namedWindow("SAHI YOLO Squirrel Tracking", cv2.WINDOW_NORMAL)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30  # fallback

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_path = r"C:\Users\cpear\Documents\VisualAnalysis\squirrel_sahi_annotated.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# -----------------------
# Simple IoU-based tracker
# -----------------------

def iou_xyxy(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    if areaA <= 0 or areaB <= 0:
        return 0.0

    return inter / (areaA + areaB - inter + 1e-6)


# tracks[id] = {"bbox": [x1,y1,x2,y2], "history": [(cx,cy), ...], "last_seen": int}
tracks = {}
next_track_id = 0

IOU_THRESH = 0.3      # match threshold between frames
MAX_AGE = 30          # frames to keep "dead" tracks around

# OPTIONAL: if you still want external track_history dict you can keep it
track_history = defaultdict(list)


# -----------------------
# Main loop
# -----------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # -----------------------
    # SAHI sliced detection
    # -----------------------
    prediction_result = get_sliced_prediction(
        image=frame,
        detection_model=sahi_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    # Convert SAHI predictions into array of [x1, y1, x2, y2, score]
    detections = []
    for pred in prediction_result.object_prediction_list:
        cid = pred.category.id

        # keep only squirrels if class known
        if squirrel_id is not None and cid != squirrel_id:
            continue

        x1, y1, x2, y2 = pred.bbox.to_xyxy()
        score = float(pred.score.value)
        detections.append([x1, y1, x2, y2, score])

    detections = np.array(detections, dtype=np.float32) if len(detections) else np.empty((0, 5), dtype=np.float32)

    # -----------------------
    # Update tracks via greedy IoU matching
    # -----------------------
    assigned = set()

    # First, try to match existing tracks to new detections
    for tid, trk in list(tracks.items()):
        best_iou = 0.0
        best_j = -1

        for j, det in enumerate(detections):
            if j in assigned:
                continue
            i = iou_xyxy(trk["bbox"], det[:4])
            if i > best_iou:
                best_iou = i
                best_j = j

        if best_j >= 0 and best_iou >= IOU_THRESH:
            # update track with this detection
            x1, y1, x2, y2, score = detections[best_j]
            trk["bbox"] = [float(x1), float(y1), float(x2), float(y2)]
            trk["last_seen"] = 0

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            trk["history"].append((cx, cy))

            track_history[tid].append((cx, cy))  # keep external history too
            assigned.add(best_j)
        else:
            # no matching detection, age the track
            trk["last_seen"] += 1
            if trk["last_seen"] > MAX_AGE:
                # drop old tracks
                del tracks[tid]

    # Any detections that weren't used become new tracks
    for j, det in enumerate(detections):
        if j in assigned:
            continue
        x1, y1, x2, y2, score = det
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        tracks[next_track_id] = {
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "history": [(cx, cy)],
            "last_seen": 0,
        }
        track_history[next_track_id].append((cx, cy))
        next_track_id += 1

    # -----------------------
    # Draw tracks
    # -----------------------
    for tid, trk in tracks.items():
        x1, y1, x2, y2 = map(int, trk["bbox"])

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

        # ID label
        cv2.putText(
            frame,
            f"ID {tid}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # path / polyline
        hist = track_history[tid]
        if len(hist) >= 2:
            pts = np.array(hist, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=(230, 0, 230), thickness=10)

    # -----------------------
    # Display + write
    # -----------------------
    cv2.imshow("SAHI YOLO Squirrel Tracking", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Saved SAHI-tracking video to:", out_path)
