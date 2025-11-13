#!/usr/bin/env python3
import os
import cv2

VIDEO_PATH = "/home/ivan/Downloads/drive_dataset_kvasir/labelled_videos/videos"
OUTPUT = "/home/ivan/Downloads/jose_luis_act/output"

def get_video_frame(frame_number, video_name, base_path=VIDEO_PATH, default_ext="mp4"):
    """
    Returns a specific frame from a video as a NumPy array (BGR format).

    Parameters
    ----------
    frame_number : int
        Index of the frame to extract (0-based).
    video_name : str
        Video file name, with or without extension.
    base_path : str, optional
        Folder where videos are stored. Defaults to global X.
    default_ext : str, optional
        Default extension to use if none is provided (without the dot).
    """
    root, ext = os.path.splitext(video_name)
    if ext == "":
        video_name = f"{video_name}.{default_ext}"

    video_path = os.path.join(base_path, video_name)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        raise ValueError(f"Cannot read frame {frame_number} from {video_path}")

    return frame


def get_video_frame_and_second(frame_number, video_name, base_path=VIDEO_PATH, default_ext="mp4"):
    """
    Returns
    -------
    frame : numpy.ndarray
        The requested frame in BGR format.
    time_sec : float
        Time (in seconds) corresponding to that frame.
    """
    root, ext = os.path.splitext(video_name)
    if ext == "":
        video_name = f"{video_name}.{default_ext}"

    video_path = os.path.join(base_path, video_name)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS ({fps}) for video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if not success or frame is None:
        cap.release()
        raise ValueError(f"Cannot read frame {frame_number} from {video_path}")

    time_sec = frame_number / fps

    cap.release()
    return frame, time_sec

if __name__ == "__main__":
    if not os.path.isdir(VIDEO_PATH):
        raise FileNotFoundError(f"Input folder does not exist: {VIDEO_PATH}")
    video_name = "7a47e8eacea04e64"
    frame_num = 52355
    frame = get_video_frame(frame_num, video_name)
    out_path = os.path.join(OUTPUT, f"{video_name}_frame_{frame_num}.png")
    cv2.imwrite(out_path, frame)


    frame, t = get_video_frame_and_second(frame_num, video_name)
    print(f"Time (s): {t:.3f}")
    print(f"Saved frame to: {out_path}")
