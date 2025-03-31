import pyttsx3

def sintetizar_emocion(emocion, texto):
    engine = pyttsx3.init()
    
    if emocion == "alegría":
        engine.setProperty('rate', 200)
        engine.setProperty('volume', 0.9) 
    elif emocion == "tristeza":
        engine.setProperty('rate', 120)  
        engine.setProperty('volume', 0.6)  
    elif emocion == "ira":
        engine.setProperty('rate', 210)  
        engine.setProperty('volume', 1.0) 
    elif emocion == "neutral":
        engine.setProperty('rate', 150)  
        engine.setProperty('volume', 0.8)  
    else:
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
    
    engine.say(texto)
    engine.runAndWait()


