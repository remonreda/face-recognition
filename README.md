# 🙂 Face Recognition

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-dlib-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Recognise known people from a **webcam stream or a still image** using OpenCV and the [`face_recognition`](https://github.com/ageitgey/face_recognition) library (dlib's 128-d face embeddings). Add one photo per person, run the script, and recognised faces are boxed and labelled with their name; unrecognised faces are labelled `UNKNOWN`.

## Project structure

```
face-recognition/
├── src/face_recognition_app.py
├── known_faces/                # put one photo per person here (alice.jpg, bob.png, …)
├── requirements.txt
└── LICENSE
```

## Getting started

```bash
git clone https://github.com/<your-username>/face-recognition.git
cd face-recognition
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`dlib` needs CMake and a C++ compiler to build. If installation fails, use the prebuilt wheel instead:

```bash
pip install dlib-bin opencv-python numpy setuptools
pip install --no-deps face_recognition face_recognition_models
```

## Usage

1. Add a clear, front-facing photo of each person to `known_faces/`. The **file name is the label** (`alice.jpg` → `ALICE`). Files without a detectable face are skipped with a warning.
2. Run:

```bash
python src/face_recognition_app.py                                   # webcam, press q to quit
python src/face_recognition_app.py --image group.jpg --output outputs/group.jpg   # still image
```

| Option | Default | Description |
|---|---|---|
| `--faces-dir` | `known_faces/` | folder of reference photos |
| `--image` | – | recognise faces in this image instead of the webcam |
| `--output` | – | save the annotated image (with `--image`) |
| `--camera` | `0` | webcam index |
| `--tolerance` | `0.6` | max face distance for a match — lower is stricter |
| `--scale` | `0.25` webcam / `1.0` image | frame down-scaling for speed |

## How it works

1. Each reference photo is converted to a 128-dimensional face encoding.
2. Every frame is down-scaled, faces are located and encoded.
3. Each face is compared against the known encodings; the closest one is accepted if its distance is within `--tolerance`.
4. A box and the person's name are drawn on the frame.

## Privacy

`known_faces/` contains personal photos — **do not commit real people's photos** to a public repository. The folder is kept in git only via an empty `.gitkeep`; add your own photos locally.

## License

[MIT](LICENSE)
