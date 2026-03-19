import os
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import csv

# Hardcoded paths
VIDEO_DIR = r"D:\Wildfire_VD\wildfire_videos"
FRAME_SAVE_DIR = r"D:\Wildfire_VD\labeled_output\frames"
CSV_SAVE_PATH = r"D:\Wildfire_VD\labeled_output\frame_labels.csv"

# Ensure directories exist
os.makedirs(FRAME_SAVE_DIR, exist_ok=True)

# Load video files
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
video_index = 0
frame_index = 0
current_video = None
cap = None
current_frame_number = 0
frame_list = []

# Create or open CSV
csv_file = open(CSV_SAVE_PATH, 'a', newline='')
csv_writer = csv.writer(csv_file)
if os.stat(CSV_SAVE_PATH).st_size == 0:
    csv_writer.writerow(['filename', 'smoke', 'density'])

# --- TKINTER GUI SETUP ---
root = tk.Tk()
root.title("Wildfire Smoke Labeling Tool")

# UI Elements
image_label = tk.Label(root)
image_label.pack()

smoke_var = tk.BooleanVar()
tk.Checkbutton(root, text="Smoke Present", variable=smoke_var).pack()

density_var = tk.StringVar(value="none")
for level in ["none", "low", "medium", "high"]:
    ttk.Radiobutton(root, text=level.capitalize(), variable=density_var, value=level).pack(anchor=tk.W)

status_label = tk.Label(root, text="")
status_label.pack()

# Load next video and select 4 frames
def load_next_video():
    global cap, current_video, video_index, frame_index, frame_list

    if cap:
        cap.release()

    if video_index >= len(video_files):
        status_label.config(text="All videos labeled!")
        root.quit()
        return

    current_video = video_files[video_index]
    video_path = os.path.join(VIDEO_DIR, current_video)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list = [
        total_frames // 5,
        total_frames // 2,
        total_frames * 3 // 5,
        total_frames * 4 // 5
    ]
    frame_index = 0
    show_frame()

# Show the current frame
def show_frame():
    global current_frame_number

    if frame_index >= len(frame_list):
        # Move to next video
        global video_index
        video_index += 1
        load_next_video()
        return

    current_frame_number = frame_list[frame_index]
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
    success, frame = cap.read()

    if not success:
        status_label.config(text=f"Failed to load frame {current_frame_number}")
        return

    # Convert to display image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    resized = img.resize((600, 450))
    tk_image = ImageTk.PhotoImage(resized)
    image_label.imgtk = tk_image
    image_label.config(image=tk_image)

# Save the current frame and label
def save_and_next():
    global frame_index

    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
    success, frame = cap.read()

    if not success:
        frame_index += 1
        show_frame()
        return

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fname = f"vid_{video_index}_frame_{frame_index}.jpg"
    img.save(os.path.join(FRAME_SAVE_DIR, fname))

    # Save label
    csv_writer.writerow([fname, 'yes' if smoke_var.get() else 'no', density_var.get()])
    csv_file.flush()

    frame_index += 1
    show_frame()

# Skip frame
def skip_frame():
    global frame_index
    frame_index += 1
    show_frame()

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Save Label", command=save_and_next).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Skip Frame", command=skip_frame).pack(side=tk.LEFT, padx=5)

# Start
load_next_video()
root.mainloop()

# Cleanup
if cap:
    cap.release()
csv_file.close()
