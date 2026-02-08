# Sign Language Detection using MediaPipe + LSTM

Real-time sign language action detection using **MediaPipe Holistic** for landmark extraction and an **LSTM neural network** for sequence classification.

## Supported Actions

| Action     | Description             |
|------------|-------------------------|
| `hello`    | Waving hand gesture     |
| `thanks`   | Thank you gesture       |
| `okay`     | OK hand sign            |
| `iloveyou` | I Love You hand sign    |

## Project Structure

```
├── README.md
├── requirement.txt
├── data/                       # (placeholder for additional datasets)
└── src/
    ├── project.ipynb           # Main notebook: data collection → training → evaluation
    ├── demo.py                 # Real-time inference script (webcam)
    ├── hand_tracking_module.py # Reusable hand-tracking class (MediaPipe Hands)
    ├── mp_holisitc_tracking_module.py  # Reusable holistic-tracking class
    ├── testing.ipynb           # YOLO hand-detection experiment
    ├── action.h5               # Trained LSTM model (generated after training)
    └── MP_Data/                # Collected keypoint sequences (30 videos × 30 frames × 4 actions)
        ├── hello/
        ├── thanks/
        ├── okay/
        └── iloveyou/
```

## How It Works

1. **Landmark Extraction** – MediaPipe Holistic detects 33 pose, 468 face, and 21+21 hand landmarks per frame → **1662 values** per frame.
2. **Data Collection** – 30 video sequences per action, each 30 frames long, are captured and saved as `.npy` arrays.
3. **LSTM Training** – A 3-layer LSTM model learns temporal patterns from the sequences.
4. **Real-Time Inference** – The webcam feed is processed frame-by-frame; the last 30 frames are fed to the model to predict the current action.

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd Open-Cv_Project

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirement.txt
```

## Usage

### Run the full pipeline (notebook)

Open `src/project.ipynb` in VS Code / Jupyter and run cells in order:
- **Cells 1-8** – Imports, helpers, live landmark preview
- **Cells 9-13** – Keypoint extraction functions
- **Cells 14-16** – Folder setup & data collection (webcam required)
- **Cells 17-24** – Preprocessing & train/test split
- **Cells 25-33** – Build, compile, train LSTM model
- **Cells 34-38** – Evaluation (confusion matrix + accuracy)
- **Cell 39** – Real-time sign language detection demo

### Run the standalone demo

```bash
cd src
python demo.py
```
> Press **d** to quit the webcam window.  
> If no trained model (`action.h5`) is found, it runs in visualization-only mode.

## Training Details

| Parameter            | Value                  |
|----------------------|------------------------|
| Sequences per action | 30                     |
| Frames per sequence  | 30                     |
| Feature vector size  | 1662                   |
| LSTM layers          | 3 (64 → 128 → 64)     |
| Dense layers         | 2 (64 → 32)           |
| Output activation    | Softmax (4 classes)    |
| Optimizer            | Adam (lr=0.001)        |
| Loss                 | Categorical Crossentropy |
| Early Stopping       | patience=10, restore_best_weights=True |

## License

This project is for educational and personal use.