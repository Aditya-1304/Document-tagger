# Video Authenticity Checker

A web-based application that classifies videos as **Real** or **Fake** using a pre-trained EfficientNet v2 B0 model. The system extracts key frames from a provided video URL, processes them through the model, and aggregates the predictions to determine the video's authenticity.

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Video URL Submission:** Users can input a video URL directly on the website
- **Frame Extraction:** Extracts key frames from the video using OpenCV
- **Model Inference:** Uses a pre-trained EfficientNet v2 B0 model for classifying each frame
- **Prediction Aggregation:** Aggregates frame-level predictions for final decision
- **Simple & Responsive UI:** Built with HTML, CSS, and JavaScript

## Architecture Overview

```
[User Browser]
    ↓
[Front-end UI (HTML/CSS/JS)]
    ↓ Video URL submission
[Back-end API (Flask)]
    ↓ Downloads video, extracts frames
[Model Inference Engine (PyTorch)]
    ↓
[Aggregation & Response]
    ↓
[Result Displayed to User]
```

## Demo

> **Note:** For demonstration purposes, the app currently processes videos stored locally or accessible via direct URLs.

## Installation

### Prerequisites

- Python 3.7+
- pip
- Git
- FFmpeg (optional)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/video-authenticity-checker.git
cd video-authenticity-checker
```

2. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:

```bash
python app.py
```

2. Open http://127.0.0.1:5000 in your browser
3. Submit a video URL and click "Analyze Video"

## Project Structure

```
video-authenticity-checker/
├── app.py                     # Main Flask application
├── templates/
│   └── index.html            # Front-end UI
├── static/
│   ├── css/
│   └── js/
├── efficientnet_v2_b0.pth    # Pre-trained model
└── requirements.txt
```

## Deployment

Deploy on free platforms:

- Heroku
- Render
- Replit

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m "Add feature"`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgements

- EfficientNet v2 developers
- Flask, OpenCV, and PyTorch maintainers
