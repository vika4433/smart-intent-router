from langdetect import detect, DetectorFactory
try:
    import langid
    langid_available = True
except ImportError:
    langid_available = False
    print("Warning: langid not available, using langdetect only")

# Ensure reproducibility for langdetect
DetectorFactory.seed = 0

def detect_language(text: str) -> str:
    """
    Detect the language of the given text.
    Uses langdetect for short texts and langid for long texts.

    Args:
        text (str): The input text to analyze.

    Returns:
        str: The detected language code (ISO 639-1), e.g., 'en' for English.
             Returns 'unknown' if detection fails or confidence is too low.
    """
    if not text or len(text.strip()) < 3:
        return "unknown"

    # You can tune this threshold!
    threshold = 200

    try:
        if len(text) < threshold:
            # Use langdetect for short texts
            return detect(text)
        else:
            # Use langid for long texts if available, otherwise fallback to langdetect
            if langid_available:
                lang, confidence = langid.classify(text)
                if confidence < 0.8:
                    return "unknown"
                return lang
            else:
                # Fallback to langdetect for long texts
                return detect(text)
    except Exception:
        return "unknown"
