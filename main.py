import numpy as np
import cv2
import os
import math

# module-level state so importing and testing won't fail
cache = None
first_frame = 1

def interested_region(img, vertices):
    """
    Mask the image to keep only the polygon defined by vertices.
    vertices should be a list/array as used by cv2.fillPoly.
    """
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def lines_drawn(img, lines, color=[255, 0, 0], thickness=10):
    """
    Average left/right Hough lines and draw two smoothed lane lines.
    Uses module-level cache and first_frame for temporal smoothing.
    """
    global cache, first_frame
    α = 0.2  # smoothing factor

    if lines is None:
        return

    left_lines = []
    right_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # slope guard against division by zero
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if 0.5 < slope < 2:      # right lane (positive slope)
                right_lines.append((x1, y1, x2, y2))
            elif -2 < slope < -0.5:  # left lane (negative slope)
                left_lines.append((x1, y1, x2, y2))

    if len(left_lines) == 0 or len(right_lines) == 0:
        return

    # Average the left and right line coordinates
    left_avg = np.mean(left_lines, axis=0).astype(int)
    right_avg = np.mean(right_lines, axis=0).astype(int)

    x1_l, y1_l, x2_l, y2_l = left_avg
    x1_r, y1_r, x2_r, y2_r = right_avg

    present_frame = np.array(
        [x1_l, y1_l, x2_l, y2_l, x1_r, y1_r, x2_r, y2_r],
        dtype="float32"
    )

    if first_frame == 1:
        next_frame = present_frame
        first_frame = 0
    else:
        prev_frame = cache
        if prev_frame is None:
            next_frame = present_frame
        else:
            next_frame = (1 - α) * prev_frame + α * present_frame

    # Draw lines on the image
    cv2.line(img,
             (int(next_frame[0]), int(next_frame[1])),
             (int(next_frame[2]), int(next_frame[3])),
             color, thickness)
    cv2.line(img,
             (int(next_frame[4]), int(next_frame[5])),
             (int(next_frame[6]), int(next_frame[7])),
             color, thickness)

    cache = next_frame

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Run probabilistic Hough transform and draw averaged lines onto a blank image.
    """
    lines = cv2.HoughLinesP(
        img, rho, theta, threshold, np.array([]),
        minLineLength=min_line_len, maxLineGap=max_line_gap
    )
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lines_drawn(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    """
    image: expected to be an RGB numpy array (MoviePy uses RGB frames).
    Returns: RGB image with lane lines overlayed.
    """
    global first_frame, cache

    # Convert assuming input is RGB (MoviePy). If using cv2.VideoCapture (BGR), convert first.
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

    gauss_gray = cv2.GaussianBlur(mask_yw_image, (5, 5), 0)
    canny_edges = cv2.Canny(gauss_gray, 50, 150)

    imshape = image.shape
    lower_left = [int(imshape[1] / 9), imshape[0]]
    lower_right = [int(imshape[1] - imshape[1] / 9), imshape[0]]
    top_left = [int(imshape[1] / 2 - imshape[1] / 8), int(imshape[0] / 2 + imshape[0] / 10)]
    top_right = [int(imshape[1] / 2 + imshape[1] / 8), int(imshape[0] / 2 + imshape[0] / 10)]
    vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
    roi_image = interested_region(canny_edges, vertices)

    theta = np.pi / 180
    line_image = hough_lines(roi_image, 4, theta, 30, 100, 180)
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)
    return result


if __name__ == "__main__":
    # Try to use MoviePy's VideoFileClip if available and supports fl_image.
    # Otherwise fallback to a robust OpenCV-based processor that uses process_image.
    import os, traceback
    INPUT = "input1.mp4"
    OUTPUT = "processed_output1.mp4"

    # prefer to import VideoFileClip but tolerate failure
    try:
        from moviepy.editor import VideoFileClip
    except Exception:
        VideoFileClip = None

    # helper: OpenCV fallback writer
    def opencv_process(in_path, out_path):
        import cv2, time
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            print(f"ERROR: cannot open input video: {in_path}")
            return 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            print(f"ERROR: cannot open writer for: {out_path}")
            cap.release()
            return 1

        idx = 0
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            try:
                # convert BGR->RGB for process_image (your function expects RGB)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out_rgb = process_image(rgb)
                if out_rgb is None:
                    out_bgr = frame
                else:
                    # ensure dtype and shape
                    import numpy as np
                    arr = np.asarray(out_rgb)
                    if arr.dtype != frame.dtype:
                        arr = arr.astype(frame.dtype)
                    # convert RGB->BGR for writer
                    out_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    # if size mismatch, resize to writer size
                    if (out_bgr.shape[1], out_bgr.shape[0]) != (w, h):
                        out_bgr = cv2.resize(out_bgr, (w, h))
                writer.write(out_bgr)
            except Exception as e:
                print("[opencv_process] frame error:", e)
                traceback.print_exc()
                writer.write(frame)
            idx += 1
            if idx % 100 == 0:
                print(f"[opencv_process] frames written: {idx}")
        writer.release()
        cap.release()
        dt = time.time() - t0
        print(f"[opencv_process] finished. frames={idx} time={dt:.1f}s")
        return 0

    # If we have a VideoFileClip and it has fl_image, use it (best-effort)
    try:
        if VideoFileClip is not None:
            clip = VideoFileClip(INPUT)
            if hasattr(clip, "fl_image"):
                print("Using MoviePy.fl_image pipeline")
                processed_clip = clip.fl_image(process_image)
                # write without audio for simplicity
                try:
                    processed_clip.write_videofile(OUTPUT, audio=False, codec='libx264')
                except Exception as e:
                    print("MoviePy write_videofile failed:", e)
                    print("Falling back to OpenCV writer...")
                    opencv_process(INPUT, OUTPUT)
            else:
                print("VideoFileClip has no fl_image — using OpenCV fallback")
                opencv_process(INPUT, OUTPUT)
        else:
            print("MoviePy not available — using OpenCV fallback")
            opencv_process(INPUT, OUTPUT)
    except Exception as e:
        print("Processing pipeline exception:", e)
        traceback.print_exc()
        print("Falling back to OpenCV processing...")
        opencv_process(INPUT, OUTPUT)
