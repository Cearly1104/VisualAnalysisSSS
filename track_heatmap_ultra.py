import os

import cv2
import numpy as np
from ultralytics import solutions


def next_available_filename(base, ext):
    idx = 1
    while True:
        candidate = f"{base}_{idx:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


video_path = r"C:\Users\cpear\Downloads\Trail Cam Tuesdays  4 (Eastern Gray Squirrel) [9CVmV3aG29s].mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

# Video writer
w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH,
    cv2.CAP_PROP_FRAME_HEIGHT,
    cv2.CAP_PROP_FPS,
))
video_out_path = next_available_filename("heatmap_output", ".avi")
video_writer = cv2.VideoWriter(
    video_out_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps if fps > 0 else 30,
    (w, h),
)

# Initialize heatmap (Ultralytics example style)
heatmap = solutions.Heatmap(
    show=False,
    model="C:\\Users\\cpear\\Documents\\VisualAnalysis\\runs\\detect\\train10\\weights\\best.pt",
    colormap=cv2.COLORMAP_INFERNO,
    device="cuda",
)

cv2.namedWindow("YOLO Heatmap", cv2.WINDOW_NORMAL)
last_frame = None

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    last_frame = im0

    results = heatmap(im0)
    out_frame = results.plot_im

    video_writer.write(out_frame)
    display_frame = cv2.resize(out_frame, (1280, 720))
    cv2.imshow("YOLO Heatmap", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("Saved heatmap video to:", video_out_path)

if last_frame is not None and hasattr(heatmap, "heatmap") and heatmap.heatmap is not None:
    heatmap_norm = cv2.normalize(heatmap.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, heatmap.colormap)
    final_frame = cv2.addWeighted(last_frame, 0.5, heatmap_color, 0.5, 0)
    image_out_path = next_available_filename("heatmap_final", ".png")
    cv2.imwrite(image_out_path, final_frame)
    print("Saved final heatmap image to:", image_out_path)
