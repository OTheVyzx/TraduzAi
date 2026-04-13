from deep_translator import GoogleTranslator
try:
    langs = GoogleTranslator().get_supported_languages(as_dict=True)
    print(f"Portuguese code: {langs.get('portuguese')}")
except Exception as e:
    print(f"Error: {e}")
