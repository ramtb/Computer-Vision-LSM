import torch
from TTS.api import TTS
import sounddevice as sd

# Detectar si hay una GPU disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo en uso: {device}")

# # Load the TTS model
# 1: tts_models/multilingual/multi-dataset/xtts_v2
#  2: tts_models/multilingual/multi-dataset/xtts_v1.1
#  3: tts_models/multilingual/multi-dataset/your_tts
#  4: tts_models/multilingual/multi-dataset/bark [already downloaded]
#  5: tts_models/bg/cv/vits
#  6: tts_models/cs/cv/vits
#  7: tts_models/da/cv/vits
#  8: tts_models/et/cv/vits
#  9: tts_models/ga/cv/vits
#  10: tts_models/en/ek1/tacotron2
#  11: tts_models/en/ljspeech/tacotron2-DDC
#  12: tts_models/en/ljspeech/tacotron2-DDC_ph
#  13: tts_models/en/ljspeech/glow-tts
#  14: tts_models/en/ljspeech/speedy-speech
#  15: tts_models/en/ljspeech/tacotron2-DCA
#  16: tts_models/en/ljspeech/vits
#  17: tts_models/en/ljspeech/vits--neon
#  18: tts_models/en/ljspeech/fast_pitch
#  19: tts_models/en/ljspeech/overflow
#  20: tts_models/en/ljspeech/neural_hmm
#  21: tts_models/en/vctk/vits
#  22: tts_models/en/vctk/fast_pitch
#  23: tts_models/en/sam/tacotron-DDC
#  24: tts_models/en/blizzard2013/capacitron-t2-c50
#  25: tts_models/en/blizzard2013/capacitron-t2-c150_v2
#  26: tts_models/en/multi-dataset/tortoise-v2
#  27: tts_models/en/jenny/jenny
#  28: tts_models/es/mai/tacotron2-DDC [already downloaded]
#  29: tts_models/es/css10/vits [already downloaded]
#  30: tts_models/fr/mai/tacotron2-DDC
#  31: tts_models/fr/css10/vits
#  32: tts_models/uk/mai/glow-tts
#  33: tts_models/uk/mai/vits
#  34: tts_models/zh-CN/baker/tacotron2-DDC-GST
#  35: tts_models/nl/mai/tacotron2-DDC
#  36: tts_models/nl/css10/vits
#  37: tts_models/de/thorsten/tacotron2-DCA
#  38: tts_models/de/thorsten/vits
#  39: tts_models/de/thorsten/tacotron2-DDC
#  40: tts_models/de/css10/vits-neon
#  41: tts_models/ja/kokoro/tacotron2-DDC
#  42: tts_models/tr/common-voice/glow-tts
#  43: tts_models/it/mai_female/glow-tts
#  44: tts_models/it/mai_female/vits
#  45: tts_models/it/mai_male/glow-tts
#  46: tts_models/it/mai_male/vits
#  47: tts_models/ewe/openbible/vits
#  48: tts_models/hau/openbible/vits
#  49: tts_models/lin/openbible/vits
#  50: tts_models/tw_akuapem/openbible/vits
#  51: tts_models/tw_asante/openbible/vits
#  52: tts_models/yor/openbible/vits
#  53: tts_models/hu/css10/vits
#  54: tts_models/el/cv/vits
#  55: tts_models/fi/css10/vits
#  56: tts_models/hr/cv/vits
#  57: tts_models/lt/cv/vits
#  58: tts_models/lv/cv/vits
#  59: tts_models/mt/cv/vits
#  60: tts_models/pl/mai_female/vits
#  61: tts_models/pt/cv/vits
#  62: tts_models/ro/cv/vits
#  63: tts_models/sk/cv/vits
#  64: tts_models/sl/cv/vits
#  65: tts_models/sv/cv/vits
#  66: tts_models/ca/custom/vits
#  67: tts_models/fa/custom/glow-tts
#  68: tts_models/bn/custom/vits-male
#  69: tts_models/bn/custom/vits-female
#  70: tts_models/be/common-voice/glow-tts

#  Name format: type/language/dataset/model
#  1: vocoder_models/universal/libri-tts/wavegrad
#  2: vocoder_models/universal/libri-tts/fullband-melgan [already downloaded]
#  3: vocoder_models/en/ek1/wavegrad
#  4: vocoder_models/en/ljspeech/multiband-melgan
#  5: vocoder_models/en/ljspeech/hifigan_v2
#  6: vocoder_models/en/ljspeech/univnet
#  7: vocoder_models/en/blizzard2013/hifigan_v2
#  8: vocoder_models/en/vctk/hifigan_v2
#  9: vocoder_models/en/sam/hifigan_v2
#  10: vocoder_models/nl/mai/parallel-wavegan
#  11: vocoder_models/de/thorsten/wavegrad
#  12: vocoder_models/de/thorsten/fullband-melgan
#  13: vocoder_models/de/thorsten/hifigan_v1
#  14: vocoder_models/ja/kokoro/hifigan_v1
#  15: vocoder_models/uk/mai/multiband-melgan
#  16: vocoder_models/tr/common-voice/hifigan
#  17: vocoder_models/be/common-voice/hifigan
what_model = input("Introduce el modelo de TTS que deseas utilizar: \n (R: rapido pero menos calidad L:lento pero mas calidad) ")
what_model = what_model.upper()
if what_model != "R" and what_model != "L":
    print("Modelo no reconocido")
    exit()



elif what_model == "L":
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
    # tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
    


while True:
    # Texto a convertir en audio
    texto = input("Introduce el texto que deseas convertir en audio: ")
    genre = input("Introduce el genero del hablante (M: masculino, F: femenino): ")
    genre = genre.upper()
    
    if what_model == "L":

    # Xtts 
    #['Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence', 'Annmarie Nele',
    # 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie',
    # 'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin',
    # 'Craig Gutsy', 'Damien Black', 'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 'Torcull Diarmuid',
    # 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland', 'Szofi Granger',
    # 'Camilla Holmström', 'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 'Alma María', 'Rosemary Okafor', 
    # 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Eugenio Mataracı', 'Ferran Simen', 'Xavier Hayasaka',
    # 'Luis Moray', 'Marcos Rudaski']

        
        if genre == "F":
            audio = tts.tts(texto, language="es", speaker = 'Alma María', speed = 0.1)
        elif genre == "M":
            audio = tts.tts(texto, language="es", speaker = 'Luis Moray', speed = 0.1)
    #### Luis moray is the best speaker for spanish for male
    #### Alma maria is the best speaker for spanish
    elif what_model == "R":
        if genre == "F":
            tts = TTS(model_name="tts_models/es/mai/tacotron2-DDC", progress_bar=False).to(device)
        elif genre == "M":
            tts = TTS(model_name="tts_models/es/css10/vits", progress_bar=False).to(device)
        audio = tts.tts(texto, speed  = 0.1)
        
    sample_rate = 22050  

    
    sd.play(audio, samplerate=sample_rate)
    sd.wait()  # Esperar a que termine la reproducción
    
    
    respuesta = input("¿Desea repetir el proceso? (s/n): ")
    if respuesta.lower() != "s":
        break


"""for tts.tts Convert text to speech.

        Args:
            text (str):
                Input text to synthesize.
            speaker (str, optional):
                Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
            language (str): Language of the text. If None, the default language of the speaker is used. Language is only
                supported by `XTTS` model.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            emotion (str, optional):
                Emotion to use for 🐸Coqui Studio models. If None, Studio models use "Neutral". Defaults to None.
            speed (float, optional):
                Speed factor to use for 🐸Coqui Studio models, between 0 and 2.0. If None, Studio models use 1.0.
                Defaults to None.
            split_sentences (bool, optional):
                Split text into sentences, synthesize them separately and concatenate the file audio.
                Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
                applicable to the 🐸TTS models. Defaults to True.
            kwargs (dict, optional):
                Additional arguments for the model.
        """