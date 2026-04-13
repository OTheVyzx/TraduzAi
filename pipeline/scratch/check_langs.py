from deep_translator import GoogleTranslator
try:
    langs = GoogleTranslator().get_supported_languages(as_dict=True)
    print(langs.get("korean"))
    print(langs.get("japanese"))
    print(langs.get("chinese (simplified)"))
except Exception as e:
    print(f"Error: {e}")
