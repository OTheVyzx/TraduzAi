from deep_translator import GoogleTranslator
try:
    source_text = "..., 그, 그런 짓은 하면 안 되는 거야"
    translated = GoogleTranslator(source='ko', target='pt').translate(source_text)
    print(f"Result: {translated}")
except Exception as e:
    print(f"Error: {e}")
