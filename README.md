# 🌊 Underwater Waste Detection & Threat Scoring System

An AI-powered computer vision system for detecting and classifying underwater waste with environmental threat assessment, built for marine conservation efforts.

<img width="1602" height="620" alt="image" src="https://github.com/user-attachments/assets/fb67a515-4f03-425f-b670-5a7f1169c647" />


## 🎯 Project Overview

This project uses **YOLOv8** deep learning model to detect and classify underwater waste into multiple categories (plastic, metal, glass, etc.) and assigns threat scores based on environmental impact and decomposition time.

### Key Features
- ✅ Multi-class waste detection (5+ categories)
- ✅ Real-time object detection using YOLOv8
- ✅ Environmental threat scoring (0-10 scale)
- ✅ Batch image analysis
- ✅ Visual analytics dashboard
- ✅ Trained on 3,628 underwater images

## 📊 Results

- **Training Dataset:** 3,628 images
- **Validation Dataset:** 1,001 images
- **Model:** YOLOv8 Nano
- **Training Epochs:** 50
- **mAP@50:**<img width="747" height="372" alt="image" src="https://github.com/user-attachments/assets/a362359a-b94d-4078-ae40-5476e877fef7" />
- **Inference Speed:** ~8ms per image

### Threat Scoring System

| Waste Type | Threat Score | Decomposition Time | Environmental Impact |
|------------|--------------|-------------------|---------------------|
| Plastic    | 9/10         | 450-1000 years    | Microplastics, toxic to marine life |
| Metal      | 7/10         | 50-200 years      | Heavy metal contamination |
| Glass      | 5/10         | 1 million years   | Sharp edges, chemically inert |
| Paper      | 3/10         | 2-6 weeks         | Minimal harm, biodegradable |

## 🖼️ Detection Examples

### Before & After Detection
![Detection Example 1](<img width="1528" height="582" alt="image" src="https://github.com/user-attachments/assets/109140dc-c1d6-405e-9c05-c6681bda7934" />)
![Detection Example 2](<img width="1522" height="568" alt="image" src="https://github.com/user-attachments/assets/1e71b10c-14bb-46da-a4df-db7787092014" />
)

### Threat Analysis Dashboard
![Dashboard](<img width="712" height="273" alt="image" src="https://github.com/user-attachments/assets/81bb6dfd-89e4-42f5-a086-c383b8969659" />)
<img width="735" height="518" alt="image" src="https://github.com/user-attachments/assets/a2c7993f-6bfc-439b-be41-13fd349a2032" />


## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/underwater-waste-detection.git
cd underwater-waste-detection

# Install dependencies
pip install -r requirements.txt

# Download trained model (if not included)
# Place best.pt in model/ folder
```

## 💻 Usage

### Quick Start - Detection on Single Image

```python
from ultralytics import YOLO

# Load model
model = YOLO('model/best.pt')

# Run detection
results = model('path/to/image.jpg')

# Display results
results[0].show()
```

### Batch Analysis with Threat Scoring

```python
# See notebooks/underwater_waste_detection.ipynb for complete code
```

### Using the Inference Script

```bash
python src/detect.py --image path/to/image.jpg --model model/best.pt
```

## 📁 Project Structure
