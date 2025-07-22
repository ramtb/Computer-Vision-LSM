# Visión artificial para el reconocimiento de la Lengua de señas Mexicana (LSM) 🤟  

## 📚 Description 

A cutting-edge computer vision system for **Mexican Sign Language (LSM) recognition and vocalization**, built with:  
- **MediaPipe** for real-time pose detection  
- **TensorFlow/Keras** for deep learning models  
- **Coqui TTS** for natural Spanish speech synthesis  
- **PySide6** for desktop GUI  

## 🚀 Key Features  
- ✅ **94% accuracy** in static sign recognition (A-Y alphabet)  
- 🤖 **87% accuracy** in dynamic sign detection (phrases like "Hola", "Gracias")  
- 😊 **Facial expression analysis** (happy, sad, neutral, surprised)  
- 🔊 **Real-time vocalization** with emotion-aware TTS  
- 📊 **Interactive GUI** with sign history and confidence metrics  
- 📷 **Camera calibration** for optimal angle detection  

## 🛠️ Tech Stack  
- **Computer Vision**: MediaPipe, OpenCV  
- **Deep Learning**:  
  - CNN for static signs (94% acc)  
  - Bi-LSTM for dynamic signs (87% acc)  
- **Voice Synthesis**: Coqui AI XTTSv2 (Spanish)  
- **GUI**: PySide6 (Qt for Python)  
- **Data Pipeline**: Pandas, Scikit-learn  

## 📦 Installation  
```bash
git clone https://github.com/ramtb/Computer-Vision-LSM.git
cd Computer-vision-LSM
pip install -r requirements.txt
```


## ▶️ Launch System  
```bash
cd GUI
python main.py  # Starts GUI with camera feed
```

## 📊 Model Performance  
| Model | Accuracy | AUC-ROC |  
|-------|----------|---------|  
| Static Signs (NN) | 94% | 0.99 |  
| Dynamic Signs (Bi-LSTM) | 87% | 0.96 |  
| Facial Expressions | 94% | 0.98 |  

## 🌍 Deployment Options  

### Docker  
```dockerfile
FROM python:3.11-slim
COPY . /app
RUN pip install -r /app/requirements.txt
CMD ["python", "/GUI/main.py"]
```

## 🤝 How to Contribute  
1. Report issues with unusual signs  
2. Improve TTS emotional intonation  
3. Add support for regional LSM variations  


---  
*"Breaking communication barriers with AI vision"* 👁️🗨️💬