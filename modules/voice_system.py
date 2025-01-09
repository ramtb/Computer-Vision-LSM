import torch
from TTS.api import TTS
import sounddevice as sd

# Detectar si hay una GPU disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo en uso: {device}")

# Cargar el modelo Bark en TTS
tts = TTS(model_name="tts_models/multilingual/multi-dataset/bark", progress_bar=False).to(device)

while True:
    # Texto a convertir en audio
    texto = input("Introduce el texto que deseas convertir en audio: ")

    # Generar el audio con Bark
    audio = tts.tts(texto)  # El modelo ya incluye el vocoder y la generación de voz

    # Definir la frecuencia de muestreo (22050 Hz para este modelo)
    sample_rate = 22050

    # Reproducir el audio al instante
    sd.play(audio, samplerate=sample_rate)
    sd.wait()  # Esperar a que termine la reproducción

    # Preguntar si se desea repetir el proceso
    respuesta = input("¿Desea repetir el proceso? (s/n): ")
    if respuesta.lower() != "s":
        break
