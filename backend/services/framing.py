from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from backend.models import FramingKeyframe


def _load_cascades() -> Tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    haar_dir = cv2.data.haarcascades
    frontal = cv2.CascadeClassifier(str(Path(haar_dir) / "haarcascade_frontalface_default.xml"))
    profile = cv2.CascadeClassifier(str(Path(haar_dir) / "haarcascade_profileface.xml"))
    return frontal, profile


def analyze_subject_trajectory(
    video_path: Path,
    start_time: float,
    end_time: float,
    sample_interval_sec: float = 0.5,
) -> Tuple[List[FramingKeyframe], Dict[str, any]]:
    """
    Sample frames between [start_time, end_time] and locate the primary subject's horizontal
    center of gravity (normalized 0.0 to 1.0) using:
      1. OpenCV frontal + profile Haar Cascade face detection
      2. Inter-frame motion + Sobel edge saliency center-of-mass fallback
    Applies temporal EMA + deadband smoothing to prevent camera jitter.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (
            [FramingKeyframe(timestamp=0.0, center_x_norm=0.5, detection_method="center_fallback")],
            {"mode": "static_center", "face_hit_ratio": 0.0, "mean_center_x": 0.5},
        )

    frontal_cascade, profile_cascade = _load_cascades()
    clip_dur = max(0.5, end_time - start_time)
    sample_times = np.arange(start_time, end_time, max(0.25, sample_interval_sec))
    if len(sample_times) == 0:
        sample_times = np.array([start_time])

    raw_keyframes: List[FramingKeyframe] = []
    prev_gray = None
    face_hits = 0

    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        h, w = frame.shape[:2]
        scale = min(1.0, 640.0 / float(max(w, h)))
        small_w = max(64, int(w * scale))
        small_h = max(64, int(h * scale))
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        # 1. Frontal & Profile Face Detection
        faces = frontal_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.12, minNeighbors=5, minSize=(int(small_h * 0.08), int(small_h * 0.08))
        )
        method = "face_frontal"
        if len(faces) == 0:
            faces = profile_cascade.detectMultiScale(
                gray_eq, scaleFactor=1.12, minNeighbors=5, minSize=(int(small_h * 0.08), int(small_h * 0.08))
            )
            method = "face_profile"

        if len(faces) > 0:
            # Pick largest detected face (primary speaker)
            best_face = max(faces, key=lambda rect: rect[2] * rect[3])
            fx, fy, fw, fh = best_face
            cx_norm = float(fx + fw * 0.5) / float(small_w)
            face_hits += 1
        else:
            # 2. Motion + Sobel Visual Saliency Fallback
            sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edge_mag = np.hypot(sobel_x, sobel_y)

            if prev_gray is not None and prev_gray.shape == gray.shape:
                motion = cv2.absdiff(gray, prev_gray).astype(np.float32)
                saliency = 0.65 * motion + 0.35 * (edge_mag / (np.max(edge_mag) + 1e-5) * 255.0)
                method = "motion_saliency"
            else:
                saliency = edge_mag
                method = "edge_saliency"

            # Weight vertical middle band (where human subjects/speakers typically appear)
            y_weights = np.exp(-0.5 * ((np.linspace(-1.0, 1.0, small_h) - 0.0) / 0.65) ** 2)[:, None]
            weighted_sal = saliency * y_weights
            col_energy = np.sum(weighted_sal, axis=0)

            # Find peak window of width equal to 9:16 crop aspect
            win_w = max(8, int(small_h * (9.0 / 16.0)))
            if win_w < small_w and np.sum(col_energy) > 1e-3:
                kernel = np.ones(win_w, dtype=np.float32) / float(win_w)
                smoothed_cols = np.convolve(col_energy, kernel, mode="valid")
                best_left = int(np.argmax(smoothed_cols))
                cx_norm = float(best_left + win_w * 0.5) / float(small_w)
            else:
                cx_norm = 0.5

        prev_gray = gray
        raw_keyframes.append(
            FramingKeyframe(
                timestamp=round(float(t - start_time), 3),
                center_x_norm=round(float(np.clip(cx_norm, 0.05, 0.95)), 4),
                detection_method=method,
            )
        )

    cap.release()

    if not raw_keyframes:
        return (
            [FramingKeyframe(timestamp=0.0, center_x_norm=0.5, detection_method="center_fallback")],
            {"mode": "static_center", "face_hit_ratio": 0.0, "mean_center_x": 0.5},
        )

    # Apply temporal EMA smoothing + deadband hysteresis to eliminate camera shake
    smoothed: List[FramingKeyframe] = []
    ema_x = raw_keyframes[0].center_x_norm
    alpha = 0.32
    deadband = 0.035

    for kf in raw_keyframes:
        delta = kf.center_x_norm - ema_x
        if abs(delta) > deadband:
            ema_x = ema_x + alpha * delta
        smoothed.append(
            FramingKeyframe(
                timestamp=kf.timestamp,
                center_x_norm=round(float(np.clip(ema_x, 0.05, 0.95)), 4),
                detection_method=kf.detection_method,
            )
        )

    xs = [k.center_x_norm for k in smoothed]
    span = max(xs) - min(xs)
    face_ratio = round(face_hits / max(1, len(raw_keyframes)), 2)
    mode = "dynamic_subject_pan" if span >= 0.06 and len(smoothed) >= 3 else "locked_subject_crop"

    return smoothed, {
        "mode": mode,
        "face_hit_ratio": face_ratio,
        "mean_center_x": round(float(np.median(xs)), 4),
        "pan_span_norm": round(float(span), 4),
        "samples": len(smoothed),
    }


def build_ffmpeg_framing_filter(
    src_width: int,
    src_height: int,
    target_width: int,
    target_height: int,
    keyframes: List[FramingKeyframe],
    framing_info: Dict[str, any],
) -> str:
    """
    Build an FFmpeg video filter string that converts any input resolution/aspect ratio
    into a 9:16 vertical canvas (`target_width x target_height`) with intelligent subject framing.
    """
    target_aspect = float(target_width) / float(target_height)  # 0.5625 (9:16)
    src_aspect = float(src_width) / float(src_height)

    # Case 1: Source is wider than 9:16 (e.g., 16:9 landscape, 4:3, 1:1) -> Smart Horizontal Crop
    if src_aspect > target_aspect + 0.01:
        crop_h = src_height - (src_height % 2)
        crop_w = int(round(crop_h * target_aspect))
        crop_w = min(src_width, crop_w - (crop_w % 2))
        max_x = max(0, src_width - crop_w)

        if max_x == 0 or framing_info.get("mode") != "dynamic_subject_pan":
            cx = float(framing_info.get("mean_center_x", 0.5))
            crop_x = int(round(cx * src_width - crop_w * 0.5))
            crop_x = max(0, min(max_x, crop_x))
            return (
                f"crop={crop_w}:{crop_h}:{crop_x}:0,"
                f"scale={target_width}:{target_height}:flags=lanczos,"
                f"setsar=1,format=yuv420p"
            )

        # Piecewise-linear smooth pan expression x(t) across up to 8 anchor points
        step = max(1, len(keyframes) // 7)
        anchors = keyframes[::step]
        if anchors[-1].timestamp != keyframes[-1].timestamp:
            anchors.append(keyframes[-1])

        def _x_px(center_norm: float) -> int:
            val = int(round(center_norm * src_width - crop_w * 0.5))
            return max(0, min(max_x, val))

        if len(anchors) <= 1:
            x_expr = str(_x_px(anchors[0].center_x_norm))
        else:
            # Construct nested if(lt(t, t1), x0 + (x1-x0)*(t-t0)/(t1-t0), ...)
            expr = str(_x_px(anchors[-1].center_x_norm))
            for i in range(len(anchors) - 2, -1, -1):
                t0 = anchors[i].timestamp
                t1 = max(t0 + 0.05, anchors[i + 1].timestamp)
                x0 = _x_px(anchors[i].center_x_norm)
                x1 = _x_px(anchors[i + 1].center_x_norm)
                slope = round((x1 - x0) / (t1 - t0), 2)
                seg_expr = f"({x0}+{slope}*(t-{t0:.2f}))"
                expr = f"if(lt(t\\,{t1:.2f})\\,{seg_expr}\\,{expr})"
            x_expr = f"clip({expr}\\,0\\,{max_x})"

        return (
            f"crop=w={crop_w}:h={crop_h}:x='{x_expr}':y=0,"
            f"scale={target_width}:{target_height}:flags=lanczos,"
            f"setsar=1,format=yuv420p"
        )

    # Case 2: Source is narrower than 9:16 -> Crop vertically around upper-middle subject
    if src_aspect < target_aspect - 0.01:
        crop_w = src_width - (src_width % 2)
        crop_h = int(round(crop_w / target_aspect))
        crop_h = min(src_height, crop_h - (crop_h % 2))
        crop_y = max(0, int((src_height - crop_h) * 0.35))
        return (
            f"crop={crop_w}:{crop_h}:0:{crop_y},"
            f"scale={target_width}:{target_height}:flags=lanczos,"
            f"setsar=1,format=yuv420p"
        )

    # Case 3: Already 9:16 aspect ratio
    return f"scale={target_width}:{target_height}:flags=lanczos,setsar=1,format=yuv420p"
