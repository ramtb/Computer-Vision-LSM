import torch
from TTS.api import TTS
import sounddevice as sd
from bark import generate_audio  # Asegúrate de que este import es correcto

# Detectar si hay una GPU disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo en uso: {device}")

# Cargar el modelo TTS Bark (Asegúrate de que el modelo esté bien instalado)
# bark_model = TTS(model_name="tts_models/multilingual/multi-dataset/bark", progress_bar=False).to(device)

# El modelo de Bark puede tener diferentes configuraciones para emociones (este es un ejemplo general)
# Puedes explorar más sobre los diferentes parámetros en la documentación de Bark.

def generar_audio_con_emocion(texto):
    """
    Función para generar audio con emoción usando Bark.
    :param texto: Texto a convertir en voz.
    :param emocion: Emoción deseada ('neutral', 'happy', 'sad', 'angry', etc.)
    """
    # Aquí se genera el audio usando el modelo Bark
    # El parámetro 'emotion' se usa para modificar la emoción del discurso
    audio = generate_audio(texto)  # 'emotion' puede cambiarse a la emoción deseada
    
    # Reproducir el audio al instante
    sample_rate = 22050  # Frecuencia de muestreo estándar para muchos modelos TTS
    sd.play(audio, samplerate=sample_rate)
    sd.wait()  # Esperar a que termine la reproducción

while True:
    # Solicitar texto al usuario
    texto = input("Introduce el texto que deseas convertir en audio: ")
    

    # Generar y reproducir el audio con la emoción seleccionada
    generar_audio_con_emocion(texto)
    
    # Preguntar si se desea repetir el proceso
    respuesta = input("¿Desea repetir el proceso? (s/n): ")
    if respuesta.lower() != "s":
        break
