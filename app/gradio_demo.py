"""
Darija Sentiment Analyzer — Live Demo
Uses local TF-IDF model (no HuggingFace required)
Author: Khalid Morjan
"""

import gradio as gr
import pickle
import re

MODEL_LOADED = False
tfidf = None
clf = None

def load_model():
    global tfidf, clf, MODEL_LOADED
    try:
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('models/logistic_regression.pkl', 'rb') as f:
            clf = pickle.load(f)
        MODEL_LOADED = True
        print("✅ Local TF-IDF model loaded!")
    except Exception as e:
        print(f"⚠️ Model not found: {e}")

load_model()

def clean_darija(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[@#]\w+', '', text)
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'[ىي]', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def analyze_sentiment(text):
    if not text or not text.strip():
        return "⚠️ Entrez un texte.", {}, ""
    if not MODEL_LOADED:
        return "⚠️ Modèle non chargé. Lancez Notebook 02 d'abord.", {}, ""
    cleaned = clean_darija(text)
    vec = tfidf.transform([cleaned])
    label = clf.predict(vec)[0]
    proba = clf.predict_proba(vec)[0]
    classes = clf.classes_
    confidence = {c.capitalize(): round(float(p), 4) for c, p in zip(classes, proba)}
    emoji_map = {
        'positive': '😊 POSITIVE — إيجابي',
        'negative': '😞 NEGATIVE — سلبي',
        'neutral':  '😐 NEUTRAL — محايد'
    }
    display = emoji_map.get(label, label.upper())
    top_score = max(proba)
    certainty = "Très confiant (>90%)" if top_score > 0.90 else "Confiant (>75%)" if top_score > 0.75 else "Modéré"
    return display, confidence, certainty

examples = [
    ["مزيان بزاف هاد الخبر شكراً 👍"],
    ["هاد الحكومة مكتخدمش والو 😡"],
    ["واش صحيح هاد الخبر؟"],
    ["c'est vraiment bien ce projet mzyan!"],
    ["هاد المشروع ممتاز ويستحق الدعم"],
    ["لا لا لا غلط بالكامل خسارة"],
]

with gr.Blocks(title="Darija Sentiment Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🇲🇦 Darija Sentiment Analyzer — تحليل المشاعر بالدارجة
    **First open-source Moroccan Arabic sentiment analysis — Accuracy: 90.03%**
    ---
    """)
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="اكتب جملة بالدارجة",
                placeholder="مثال: مزيان بزاف هاد الخبر...",
                lines=4,
            )
            analyze_btn = gr.Button("🔍 Analyser — تحليل", variant="primary", size="lg")
        with gr.Column(scale=1):
            label_output = gr.Textbox(label="Résultat", interactive=False, lines=2)
            confidence_output = gr.Label(label="Scores de confiance", num_top_classes=3)
            certainty_output = gr.Textbox(label="Certitude", interactive=False)

    gr.Examples(examples=examples, inputs=text_input, label="📝 Exemples")

    gr.Markdown("""
    ---
    **Modèle:** TF-IDF + Logistic Regression | **Accuracy:** 90.03% | **Dataset:** 8,619 commentaires Darija
    **Auteur:** Khalid Morjan — Sultan Moulay Slimane University, Morocco
    """)

    analyze_btn.click(fn=analyze_sentiment, inputs=text_input,
                      outputs=[label_output, confidence_output, certainty_output])
    text_input.submit(fn=analyze_sentiment, inputs=text_input,
                      outputs=[label_output, confidence_output, certainty_output])

if __name__ == "__main__":
    demo.launch(share=False)