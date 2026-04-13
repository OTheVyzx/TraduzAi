from deep_translator import GoogleTranslator
try:
    source_text = "달라 붙으면... 아하하... 피타..."
    translated = GoogleTranslator(source='en', target='pt').translate(source_text)
    with open("result_ko_as_en.txt", "w", encoding="utf-8") as f:
        f.write(str(translated))
except Exception as e:
    with open("result_ko_as_en.txt", "w", encoding="utf-8") as f:
        f.write(f"Error: {e}")
