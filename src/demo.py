"""
Sign Language Detection - Real-Time Demo
=========================================
Uses a trained LSTM model + MediaPipe Holistic to detect sign language
actions in real-time from webcam feed.

Actions supported: hello, thanks, okay, iloveyou
Press 'd' to quit.
"""

import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import time

# ──────────────────────────────────────────────
# MediaPipe setup
# ──────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Action labels (must match training order)
actions = np.array(['hello', 'thanks', 'okay', 'iloveyou'])


def mediapipe_detection(img, model):
    """Run MediaPipe detection on a BGR frame."""
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = model.process(img)
    img.flags.writeable = True
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    return img, results


def draw_styled_landmarks(img, results):
    """Draw holistic landmarks with styled colors."""
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            img, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


def extracting_keypoints(results):
    """Extract pose + hand keypoints (no face — it's noise for sign language).
    Returns a 258-dim vector: pose(132) + left_hand(63) + right_hand(63)."""
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])


def main():
    # ──────────────────────────────────────────────
    # Load trained model
    # ──────────────────────────────────────────────
    try:
        import tensorflow as tf
        model_path = os.path.join(os.path.dirname(__file__), 'action.h5')
        model = tf.keras.models.load_model(model_path)
        print(f"[INFO] Model loaded from {model_path}")
        model_loaded = True
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
        print("[INFO] Running in landmark-visualization-only mode.")
        model_loaded = False

    # ──────────────────────────────────────────────
    # Config
    # ──────────────────────────────────────────────
    THRESHOLD = 0.6
    STABILITY_FRAMES = 10
    COLORS = {
        'hello':    (0, 255, 0),
        'thanks':   (255, 165, 0),
        'okay':     (0, 255, 255),
        'iloveyou': (255, 0, 255),
    }

    # ──────────────────────────────────────────────
    # State
    # ──────────────────────────────────────────────
    sequence = []
    current_sign = ""
    current_conf = 0.0
    stability_counter = 0
    last_candidate = ""
    session_log = []
    frame_count = 0
    res = np.zeros(len(actions))

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    start_time = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Detection
            img, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(img, results)

            # Prediction (only if model is loaded)
            if model_loaded:
                keypoints = extracting_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                    top_idx = np.argmax(res)
                    top_conf = res[top_idx]
                    candidate = actions[top_idx]

                    # Stability filter
                    if candidate == last_candidate and top_conf > THRESHOLD:
                        stability_counter += 1
                    else:
                        stability_counter = 1
                        last_candidate = candidate

                    if stability_counter >= STABILITY_FRAMES:
                        if candidate != current_sign:
                            elapsed = time.time() - start_time
                            session_log.append((elapsed, candidate, top_conf))
                            current_sign = candidate
                            current_conf = top_conf
                            print(f"[{elapsed:6.1f}s] Detected: {candidate} (conf: {top_conf:.2f})")

                    if candidate == current_sign:
                        current_conf = top_conf

            # --- Visualization ---
            h, w = img.shape[:2]

            # Top bar: current prediction
            cv.rectangle(img, (0, 0), (w, 50), (30, 30, 30), -1)
            if current_sign:
                color = COLORS.get(current_sign, (255, 255, 255))
                cv.putText(img, current_sign.upper(), (10, 38),
                           cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv.LINE_AA)
                # Confidence bar
                bar_x, bar_w, bar_h, bar_y = 280, 200, 20, 15
                cv.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
                fill_w = int(bar_w * current_conf)
                bar_color = (0, 255, 0) if current_conf > 0.8 else (0, 255, 255) if current_conf > 0.6 else (0, 0, 255)
                cv.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
                cv.putText(img, f"{current_conf:.0%}", (bar_x + bar_w + 10, bar_y + 16),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
            else:
                cv.putText(img, "Show a sign...", (10, 35),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2, cv.LINE_AA)

            # Bottom bar: class probabilities
            if model_loaded and len(sequence) == 30:
                bottom_y = h - 40
                cv.rectangle(img, (0, bottom_y - 5), (w, h), (30, 30, 30), -1)
                bar_total_w = w - 20
                x_offset = 10
                for i, act in enumerate(actions):
                    prob = res[i]
                    act_w = bar_total_w // len(actions) - 5
                    act_color = COLORS.get(act, (200, 200, 200))
                    cv.rectangle(img, (x_offset, bottom_y), (x_offset + act_w, h - 5), (60, 60, 60), -1)
                    fill = int(act_w * prob)
                    cv.rectangle(img, (x_offset, bottom_y), (x_offset + fill, h - 5), act_color, -1)
                    cv.putText(img, f"{act[:3]} {prob:.0%}", (x_offset + 2, h - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)
                    x_offset += act_w + 5

            # FPS
            fps = frame_count / (time.time() - start_time + 0.001)
            cv.putText(img, f"FPS: {fps:.0f}", (w - 100, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv.LINE_AA)

            cv.imshow('Sign Language Detection', img)

            if cv.waitKey(10) & 0xFF == ord('d'):
                break

    cap.release()
    cv.destroyAllWindows()

    # --- Session Summary ---
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"SESSION SUMMARY")
    print(f"{'='*50}")
    print(f"Duration: {total_time:.1f}s | Frames: {frame_count} | Avg FPS: {frame_count/total_time:.1f}")
    print(f"Total signs detected: {len(session_log)}")
    if session_log:
        print(f"\nTimeline:")
        for t, sign, conf in session_log:
            print(f"  [{t:6.1f}s] {sign:<12s} (confidence: {conf:.2f})")
        from collections import Counter
        sign_counts = Counter(s for _, s, _ in session_log)
        print(f"\nSign frequency:")
        for sign, count in sign_counts.most_common():
            print(f"  {sign}: {count} times")
        avg_conf = np.mean([c for _, _, c in session_log])
        print(f"\nAverage confidence: {avg_conf:.2f}")
    else:
        print("No signs were detected during this session.")


if __name__ == "__main__":
    main()


