from deep_translator import GoogleTranslator
try:
    source_text = "달라 붙으면... 아하하... 피타..."
    translated = GoogleTranslator(source='auto', target='pt').translate(source_text)
    print(f"Result: {translated}")
except Exception as e:
    print(f"Error: {e}")
