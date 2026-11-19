from __future__ import annotations
import argparse
import sys
from pathlib import Path
import cv2
import face_recognition
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UNKNOWN = "UNKNOWN"


def load_known_faces(faces_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    """Encode every image in ``faces_dir``; the file name (without extension) is the person's name."""
    if not faces_dir.is_dir():
        sys.exit(f"Known-faces folder not found: {faces_dir}")
    encodings, names = [], []
    for path in sorted(faces_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            image = face_recognition.load_image_file(str(path))  
        except Exception as exc:  
            print(f"[WARN] Could not read {path.name} ({type(exc).__name__}) - skipped")
            continue
        found = face_recognition.face_encodings(image)
        if not found:
            print(f"[WARN] No face found in {path.name} - skipped")
            continue
        encodings.append(found[0])
        names.append(path.stem)
        print(f"[INFO] Encoded {path.stem}")
    if not encodings:
        sys.exit(f"No usable face images in {faces_dir}. Add one clear, front-facing photo per person.")
    return encodings, names


def recognize_frame(frame_bgr, known_encodings, known_names, tolerance: float, scale: float):
    """Detect and label faces in a BGR frame (drawn in place). Returns the list of names seen."""
    small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(small_rgb)
    encodings = face_recognition.face_encodings(small_rgb, locations)

    seen = []
    inv = 1.0 / scale
    for encoding, (top, right, bottom, left) in zip(encodings, locations):
        distances = face_recognition.face_distance(known_encodings, encoding)
        best = int(np.argmin(distances))
        name = known_names[best].upper() if distances[best] <= tolerance else UNKNOWN
        seen.append(name)

        top, right, bottom, left = (int(v * inv) for v in (top, right, bottom, left))
        color = (0, 0, 255) if name == UNKNOWN else (0, 200, 0)
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame_bgr, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame_bgr, name, (left + 6, bottom - 8),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    return seen


def run_image(args, encodings, names) -> None:
    frame = cv2.imread(str(args.image))
    if frame is None:
        sys.exit(f"Could not read image: {args.image}")
    seen = recognize_frame(frame, encodings, names, args.tolerance, args.scale)
    print(f"Faces found: {seen if seen else 'none'}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), frame)
        print(f"Saved annotated image to {args.output}")


def run_camera(args, encodings, names) -> None:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera index {args.camera}")
    last_seen: set[str] = set()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read a frame from the camera")
                break
            seen = set(recognize_frame(frame, encodings, names, args.tolerance, args.scale))
            if seen != last_seen: 
                print(f"In view: {sorted(seen) if seen else 'nobody'}")
                last_seen = seen
            cv2.imshow("Face Recognition (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Face recognition from webcam or a single image.")
    parser.add_argument("--faces-dir", type=Path, default=PROJECT_ROOT / "known_faces")
    parser.add_argument("--image", type=Path, help="recognise faces in this image instead of the webcam")
    parser.add_argument("--output", type=Path, help="save the annotated image here (with --image)")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--tolerance", type=float, default=0.6,
                        help="max face distance to count as a match; lower = stricter (default 0.6)")
    parser.add_argument("--scale", type=float, default=None,
                        help="frame down-scaling for speed (default: 0.25 for webcam, 1.0 for --image)")
    args = parser.parse_args()
    if args.scale is None:
        args.scale = 1.0 if args.image else 0.25

    encodings, names = load_known_faces(args.faces_dir)
    print(f"[INFO] {len(names)} known face(s) loaded: {names}")
    (run_image if args.image else run_camera)(args, encodings, names)


if __name__ == "__main__":
    main()
