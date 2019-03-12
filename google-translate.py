from googletrans import Translator
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 150)
translator = Translator()
def say(text):
    engine.say(text)
    engine.runAndWait()
text = "boy dog"
translated_text = translator.translate(text,dest="hi")
say(text)
say(translated_text.text)
