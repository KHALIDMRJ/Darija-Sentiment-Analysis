"""
Darija Text Preprocessor
=========================
Cleans and normalizes Moroccan Arabic (Darija) text
for sentiment analysis.

Handles:
- Mixed Arabic/Latin/French scripts
- Arabic diacritics removal
- Letter normalization
- Darija-specific slang
- Emoji sentiment encoding
- Noise removal

Author: Khalid Morjan
Project: Darija Sentiment Analysis
"""

import re
import pandas as pd


# ─── DARIJA SLANG DICTIONARY ──────────────────────────────────────────────────
# Common Darija words and their sentiment signals
# Expand this as you collect more data

POSITIVE_SLANG = {
    "مزيان": "جيد",       # mzyan = good
    "مزين": "جيد",
    "واو": "رائع",         # wow
    "بركة": "كافي",
    "خويا": "أخي",         # brother (friendly)
    "حبيبي": "عزيزي",
    "والو": "لاشيء",
    "بزاف": "كثيراً",      # bzzaf = a lot
    "شحال": "كم",
    "كيفاش": "كيف",
    "فين": "أين",
    "واش": "هل",
}

# Emoji sentiment mapping
EMOJI_SENTIMENT = {
    "😍": " ايجابي_جداً ",
    "❤️": " ايجابي_جداً ",
    "🥰": " ايجابي_جداً ",
    "👍": " ايجابي ",
    "😊": " ايجابي ",
    "😁": " ايجابي ",
    "🙏": " ايجابي ",
    "😂": " ايجابي ",
    "🤣": " ايجابي ",
    "👏": " ايجابي ",
    "💪": " ايجابي ",
    "😡": " سلبي_جداً ",
    "🤬": " سلبي_جداً ",
    "😠": " سلبي ",
    "👎": " سلبي ",
    "😒": " سلبي ",
    "😢": " سلبي ",
    "😭": " سلبي ",
    "🤮": " سلبي_جداً ",
    "😤": " سلبي ",
    "💔": " سلبي ",
    "🤔": " محايد ",
    "😐": " محايد ",
}


# ─── PREPROCESSING FUNCTIONS ──────────────────────────────────────────────────

def remove_diacritics(text):
    """Remove Arabic tashkeel (diacritics)."""
    diacritics_pattern = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u0674]')
    return diacritics_pattern.sub('', text)


def normalize_arabic_letters(text):
    """Normalize Arabic letter variants to standard forms."""
    # Normalize alef variants
    text = re.sub(r'[إأآٱا]', 'ا', text)
    # Normalize yaa
    text = re.sub(r'[ىي]', 'ي', text)
    # Normalize taa marbuta
    text = re.sub(r'ة', 'ه', text)
    # Normalize waw
    text = re.sub(r'ؤ', 'و', text)
    # Normalize hamza
    text = re.sub(r'ئ', 'ي', text)
    return text


def encode_emojis(text):
    """Replace emojis with sentiment tokens."""
    for emoji, token in EMOJI_SENTIMENT.items():
        text = text.replace(emoji, token)
    return text


def remove_urls(text):
    """Remove URLs."""
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def remove_mentions_hashtags(text):
    """Remove @mentions and #hashtags."""
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    return text


def normalize_repeated_chars(text):
    """
    Normalize repeated characters (common in Darija expressions).
    e.g. 'مزياааان' → 'مزيان', 'hhhhhh' → 'hhh'
    """
    # Arabic repeated chars
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text


def remove_punctuation_noise(text):
    """Remove excessive punctuation while keeping sentence structure."""
    # Keep . ! ? but remove excessive repetition
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    # Remove other noise punctuation
    text = re.sub(r'[،؛؟,;:(){}\[\]«»"\'`~^&*+=|\\/<>]', ' ', text)
    return text


def normalize_whitespace(text):
    """Clean up whitespace."""
    return re.sub(r'\s+', ' ', text).strip()


def detect_script(text):
    """
    Detect dominant script in text.
    Returns: 'arabic', 'latin', 'mixed'
    """
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    
    total = arabic_chars + latin_chars
    if total == 0:
        return 'unknown'
    
    arabic_ratio = arabic_chars / total
    if arabic_ratio > 0.7:
        return 'arabic'
    elif arabic_ratio < 0.3:
        return 'latin'
    else:
        return 'mixed'


def clean_darija(text, keep_emojis_as_tokens=True):
    """
    Full Darija preprocessing pipeline.
    
    Steps:
    1. Encode emojis as sentiment tokens
    2. Remove URLs and mentions
    3. Remove diacritics
    4. Normalize Arabic letters
    5. Normalize repeated characters
    6. Clean punctuation
    7. Normalize whitespace
    
    Args:
        text: Raw Darija text
        keep_emojis_as_tokens: If True, convert emojis to text tokens
    
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    
    # Step 1: Encode emojis before removing them
    if keep_emojis_as_tokens:
        text = encode_emojis(text)
    else:
        # Remove emojis entirely
        text = re.sub(r'[^\u0000-\u024F\u0600-\u06FF\s]', '', text)
    
    # Step 2: Remove URLs and mentions
    text = remove_urls(text)
    text = remove_mentions_hashtags(text)
    
    # Step 3: Remove diacritics
    text = remove_diacritics(text)
    
    # Step 4: Normalize Arabic letters
    text = normalize_arabic_letters(text)
    
    # Step 5: Normalize repeated characters
    text = normalize_repeated_chars(text)
    
    # Step 6: Clean punctuation noise
    text = remove_punctuation_noise(text)
    
    # Step 7: Final whitespace cleanup
    text = normalize_whitespace(text)
    
    return text


def preprocess_dataset(input_csv, output_csv):
    """
    Apply preprocessing to the full scraped dataset.
    
    Args:
        input_csv: Path to raw scraped CSV
        output_csv: Path to save cleaned CSV
    """
    print(f"📂 Loading dataset: {input_csv}")
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    
    print(f"📊 Raw dataset: {len(df)} comments")
    
    # Apply cleaning
    print("🧹 Applying Darija preprocessing pipeline...")
    df["text_clean"] = df["text"].apply(clean_darija)
    df["script"] = df["text_clean"].apply(detect_script)
    df["text_length"] = df["text_clean"].str.len()
    df["word_count"] = df["text_clean"].str.split().str.len()
    
    # Filter out very short comments after cleaning
    df = df[df["text_length"] >= 5]
    df = df[df["word_count"] >= 2]
    
    # Remove duplicates on cleaned text
    df = df.drop_duplicates(subset=["text_clean"])
    df = df.reset_index(drop=True)
    
    print(f"✅ Clean dataset: {len(df)} comments")
    print(f"📈 Script distribution:\n{df['script'].value_counts()}")
    print(f"📏 Avg comment length: {df['text_length'].mean():.1f} chars")
    print(f"📝 Avg word count: {df['word_count'].mean():.1f} words")
    
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n💾 Saved: {output_csv}")
    
    return df


# ─── QUICK TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_samples = [
        "مزيان بزاف هاد الخبر 👍😍",
        "هاد الحكومة مكتخدمش والو، سلبي جداً 😡",
        "واش هاد الشي صحيح؟ مفهمتش شي حاجة",
        "c'est vraiment مزيان هاد المشروع ❤️",
        "لا لا لا مقبولش هاد القرار 👎👎👎",
        "شكراً بزاف على هاد المعلومة المفيدة 🙏",
    ]
    
    print("🧪 Testing Darija Preprocessor\n" + "="*40)
    for sample in test_samples:
        cleaned = clean_darija(sample)
        script = detect_script(cleaned)
        print(f"Original : {sample}")
        print(f"Cleaned  : {cleaned}")
        print(f"Script   : {script}")
        print()
