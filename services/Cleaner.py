import re
import html

class TextCleaner:
    @staticmethod
    def clean(text):
        text = str(text).lower()
        text = html.unescape(text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
