
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2, time, threading, queue, os, sys, traceback
from main import process_image


FRAME_W, FRAME_H = 640, 480
DELAY_MS = 30  
DEFAULT_VIDEO = "input2.mp4"


cap = None
video_path = DEFAULT_VIDEO
playing = False
saving = False
stop_thread = False
status_q = queue.Queue() 
save_progress_q = queue.Queue()


def safe_resize(img, size):
    try:
        return cv2.resize(img, size)
    except Exception:
        return None

def set_status(text):
    status_var.set(text)


last_time = time.time()
frame_count = 0
fps = 0.0

def update_frames():
    global cap, playing, last_time, frame_count, fps

    
    try:
        while True:
            msg = status_q.get_nowait()
            if msg == "SAVED_OK":
                messagebox.showinfo("Done", "Saved processed video.")
            else:
                
                set_status(msg)
    except queue.Empty:
        pass

    try:
        prog = save_progress_q.get_nowait()
        progress_var.set(prog.get("percent", 0))
        progress_text_var.set(prog.get("text", ""))
    except queue.Empty:
        pass

    if not playing:
        root.after(DELAY_MS, update_frames)
        return

    if cap is None or not cap.isOpened():
        try:
            cap = cv2.VideoCapture(video_path)
        except Exception as e:
            set_status(f"Failed open: {e}")
            root.after(DELAY_MS, update_frames)
            return

    ret, frame = cap.read()
    if not ret or frame is None:
       
        release_capture()
        set_status("End of video")
        play_pause_btn.config(text="Play")
        playing = False
        root.after(DELAY_MS, update_frames)
        return

    # update FPS
    frame_count += 1
    now = time.time()
    dt = now - last_time
    if dt >= 1.0:
        fps = frame_count / dt
        frame_count = 0
        last_time = now

    
    raw_resized = safe_resize(frame, (FRAME_W, FRAME_H))
    if raw_resized is not None:
        try:
            raw_rgb = cv2.cvtColor(raw_resized, cv2.COLOR_BGR2RGB)
            im_raw = Image.fromarray(raw_rgb)
            tk_raw = ImageTk.PhotoImage(im_raw)
            lbl_raw.imgtk = tk_raw
            lbl_raw.configure(image=tk_raw)
        except Exception as e:
            print("Error showing raw:", e, file=sys.stderr)

   
    try:
        in_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = process_image(in_rgb)
        if processed is not None:
            proc_resized = safe_resize(processed, (FRAME_W, FRAME_H))
            if proc_resized is not None:
                im_proc = Image.fromarray(proc_resized)
                tk_proc = ImageTk.PhotoImage(im_proc)
                lbl_proc.imgtk = tk_proc
                lbl_proc.configure(image=tk_proc)
    except Exception as e:
        print("Processing error:", e, file=sys.stderr)
        traceback.print_exc()

  
    status_text = f"File: {os.path.basename(video_path)}  |  FPS: {fps:.1f}  |  Playing"
    status_var.set(status_text)

    root.after(DELAY_MS, update_frames)

def release_capture():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
    cap = None


def open_file():
    global video_path
    path = filedialog.askopenfilename(title="Select video", filetypes=[("MP4","*.mp4"),("All files","*.*")])
    if not path:
        return
    video_path = path
    set_status(f"Loaded {os.path.basename(video_path)}")
   
    release_capture()
    play()

def toggle_play_pause():
    global playing
    if saving:
        set_status("Busy saving — cannot play/pause")
        return
    playing = not playing
    play_pause_btn.config(text="Pause" if playing else "Play")
    if playing:
        root.after(0, update_frames)

def play():
    global playing
    if saving:
        set_status("Busy saving — cannot play")
        return
    playing = True
    play_pause_btn.config(text="Pause")
    root.after(0, update_frames)

def pause():
    global playing
    playing = False
    play_pause_btn.config(text="Play")


def save_processed_video():
    global saving
    if saving:
        messagebox.showinfo("Saving", "Already saving.")
        return
    out_path = filedialog.asksaveasfilename(defaultextension=".mp4",
                                            filetypes=[("MP4","*.mp4")],
                                            title="Save processed video as")
    if not out_path:
        return
   
    saving = True
    open_btn.config(state="disabled")
    play_pause_btn.config(state="disabled")
    save_btn.config(state="disabled")
    progress_var.set(0)
    progress_text_var.set("Starting...")
   
    t = threading.Thread(target=_saver_thread, args=(video_path, out_path), daemon=True)
    t.start()

def _saver_thread(in_path, out_path):
    try:
        capw = cv2.VideoCapture(in_path)
        if not capw.isOpened():
            status_q.put(f"Failed to open input: {in_path}")
            _saver_done()
            return
        fps = capw.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(capw.get(cv2.CAP_PROP_FRAME_WIDTH)) or FRAME_W
        h = int(capw.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_H
        total = int(capw.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            status_q.put(f"Failed to open output: {out_path}")
            capw.release()
            _saver_done()
            return

        idx = 0
        last_report = time.time()
        while True:
            ret, frame = capw.read()
            if not ret or frame is None:
                break
            try:
                in_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = process_image(in_rgb)
                if processed is None:
                    out_bgr = frame
                else:
                    if processed.dtype != frame.dtype:
                        processed = processed.astype(frame.dtype)
                    out_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                if (out_bgr.shape[1], out_bgr.shape[0]) != (w, h):
                    out_bgr = cv2.resize(out_bgr, (w, h))
                writer.write(out_bgr)
            except Exception as e:
                print("Saver frame error:", e, file=sys.stderr)
                traceback.print_exc()
                writer.write(frame)
            idx += 1
           
            if time.time() - last_report > 0.5:
                pct = int((idx / total) * 100) if total else None
                save_progress_q.put({"percent": pct or 0, "text": f"Frames written: {idx}"})
                last_report = time.time()

        writer.release()
        capw.release()
        status_q.put("SAVED_OK")
    except Exception as e:
        status_q.put(f"Save failed: {e}")
        traceback.print_exc()
    finally:
        _saver_done()

def _saver_done():
    global saving
    saving = False
    
    def _reenable():
        open_btn.config(state="normal")
        play_pause_btn.config(state="normal")
        save_btn.config(state="normal")
        progress_var.set(100)
        progress_text_var.set("Complete")
    root.after(0, _reenable)


root = tk.Tk()
root.title("Lane Detection — Enhanced GUI")
root.geometry(f"{FRAME_W*2 + 420}x{FRAME_H + 180}")


toolbar = ttk.Frame(root)
toolbar.pack(side="top", fill="x", padx=6, pady=6)

open_btn = ttk.Button(toolbar, text="Open", command=open_file)
open_btn.pack(side="left", padx=6)
play_pause_btn = ttk.Button(toolbar, text="Play", command=toggle_play_pause)
play_pause_btn.pack(side="left", padx=6)
save_btn = ttk.Button(toolbar, text="Save", command=save_processed_video)
save_btn.pack(side="left", padx=6)
ttk.Button(toolbar, text="Quit", command=root.quit).pack(side="right", padx=6)


display = ttk.Frame(root)
display.pack(fill="both", expand=True, padx=8, pady=8)

lbl_raw = ttk.Label(display)
lbl_raw.pack(side="left", padx=8, pady=8)
lbl_proc = ttk.Label(display)
lbl_proc.pack(side="right", padx=8, pady=8)


bottom = ttk.Frame(root)
bottom.pack(side="bottom", fill="x", padx=8, pady=8)

status_var = tk.StringVar(value="Ready")
status_label = ttk.Label(bottom, textvariable=status_var)
status_label.pack(side="left", padx=6)

progress_var = tk.IntVar(value=0)
progress_text_var = tk.StringVar(value="")
progress = ttk.Progressbar(bottom, orient="horizontal", length=300, mode="determinate", variable=progress_var)
progress.pack(side="right", padx=6)
progress_label = ttk.Label(bottom, textvariable=progress_text_var)
progress_label.pack(side="right", padx=6)


if os.path.exists(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
    except Exception:
        cap = None

play_pause_btn.config(text="Play")
set_status("Ready — loaded " + os.path.basename(video_path) if os.path.exists(video_path) else "Ready")
root.protocol("WM_DELETE_WINDOW", lambda: (setattr(sys.modules[__name__], 'stop_thread', True), release_capture(), root.destroy()))


def poll_queues():
    try:
        while True:
            msg = status_q.get_nowait()
            if msg == "SAVED_OK":
                messagebox.showinfo("Done", "Saved processed video.")
            else:
                status_var.set(msg)
    except queue.Empty:
        pass
    try:
        prog = save_progress_q.get_nowait()
        progress_var.set(prog.get("percent", 0))
        progress_text_var.set(prog.get("text", ""))
    except queue.Empty:
        pass
    root.after(200, poll_queues)

root.after(200, poll_queues)

root.mainloop()
