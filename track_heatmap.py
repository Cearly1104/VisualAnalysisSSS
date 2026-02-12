from collections import defaultdict
import os
import cv2
import numpy as np
from ultralytics import YOLO


def next_available_filename(base, ext):
    idx = 1
    while True:
        candidate = f"{base}_{idx:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


# Load the YOLO model
model = YOLO(r"C:\Users\cpear\Documents\VisualAnalysis\runs\detect\train11\best.pt")
model.to("cuda")

# --- find squirrel class id ---
squirrel_id = None
for cid, name in model.names.items():
    if "squirrel" in name.lower():
        squirrel_id = cid
        break

print("Squirrel class id:", squirrel_id)

video_path = r"C:\Users\cpear\Documents\VisualAnalysis\squirrelmapping.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

cv2.namedWindow("YOLO Squirrel Tracking", cv2.WINDOW_NORMAL)
cv2.namedWindow("YOLO Squirrel Heatmap", cv2.WINDOW_NORMAL)

# Get video properties for writer
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30  # fallback if metadata is weird

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video paths (auto-increment)
annotated_out_path = next_available_filename("squirrel_annotated", ".mp4")
heatmap_out_path = next_available_filename("squirrel_annotated_heatmap", ".mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
annotated_out = cv2.VideoWriter(annotated_out_path, fourcc, fps, (width, height))
heatmap_out = cv2.VideoWriter(heatmap_out_path, fourcc, fps, (width, height))

track_history = defaultdict(list)
heatmap_acc = np.zeros((height, width), dtype=np.float32)
heatmap_radius = 18
heatmap_blur = 31
last_frame = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    last_frame = frame

    # Run tracking on this frame (no internal saving)
    result = model.track(
        frame,
        persist=True,
        classes=[squirrel_id] if squirrel_id is not None else None,
        conf=0.25,
        tracker="bytetrack.yaml",
    )[0]

    # Let Ultralytics draw boxes/IDs first
    annotated = result.plot()

    if result.boxes is not None and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))

            # Accumulate heatmap at box center (larger footprint for visibility)
            cx, cy = int(x), int(y)
            if 0 <= cx < width and 0 <= cy < height:
                cv2.circle(heatmap_acc, (cx, cy), heatmap_radius, 1.0, -1)

            if len(track) >= 2:
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    annotated,
                    [points],
                    isClosed=False,
                    color=(230, 0, 230),
                    thickness=10,
                )

    # Create a heatmap overlay for this frame
    heatmap_smooth = cv2.GaussianBlur(heatmap_acc, (heatmap_blur, heatmap_blur), 0)
    if heatmap_smooth.max() > 0:
        heatmap_norm = cv2.normalize(heatmap_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        heatmap_norm = np.zeros_like(heatmap_smooth, dtype=np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_INFERNO)
    heatmap_overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    # Show live
    display_annotated = cv2.resize(annotated, (1280, 720))
    display_heatmap = cv2.resize(heatmap_overlay, (1280, 720))
    cv2.imshow("YOLO Squirrel Tracking", display_annotated)
    cv2.imshow("YOLO Squirrel Heatmap", display_heatmap)

    # Write outputs
    annotated_out.write(annotated)
    heatmap_out.write(heatmap_overlay)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
annotated_out.release()
heatmap_out.release()
cv2.destroyAllWindows()

print("Saved annotated video to:", annotated_out_path)
print("Saved heatmap video to:", heatmap_out_path)

if last_frame is not None and np.any(heatmap_acc):
    heatmap_smooth = cv2.GaussianBlur(heatmap_acc, (heatmap_blur, heatmap_blur), 0)
    if heatmap_smooth.max() > 0:
        heatmap_norm = cv2.normalize(heatmap_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        heatmap_norm = np.zeros_like(heatmap_smooth, dtype=np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_INFERNO)
    final_frame = cv2.addWeighted(last_frame, 0.6, heatmap_color, 0.4, 0)
    image_out_path = next_available_filename("squirrel_heatmap_final", ".png")
    cv2.imwrite(image_out_path, final_frame)
    print("Saved final heatmap image to:", image_out_path)
