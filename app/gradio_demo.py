"""
Darija Sentiment Analyzer — Live Demo
======================================
Deploy this on HuggingFace Spaces (free hosting):
  1. Create account at huggingface.co
  2. New Space → SDK: Gradio
  3. Upload this file as app.py
  4. Upload your trained model or use the HuggingFace model ID

Author: Khalid Morjan
"""

import gradio as gr
from transformers import pipeline

# ── Load model ──
# Option A: Use your uploaded HuggingFace model (after pushing)
MODEL_ID = "KHALIDMRJ/darija-sentiment-camelbert"

# Option B: Load from local folder (for testing)
# MODEL_ID = "models/camelbert/final"

print(f"Loading model: {MODEL_ID}")
try:
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        top_k=None  # Return all label scores
    )
    print("✅ Model loaded!")
except Exception as e:
    print(f"⚠️ Model not found: {e}")
    print("Using fallback — train and push your model first (Notebook 03)")
    classifier = None


# ── Inference function ──
def analyze_sentiment(text):
    if not text or not text.strip():
        return "⚠️ Please enter some text.", {}, ""

    if classifier is None:
        return "⚠️ Model not loaded. Run Notebook 03 first.", {}, ""

    results = classifier(text, truncation=True, max_length=128)[0]

    # Sort by score
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    top = results_sorted[0]

    label = top['label']
    score = top['score']

    # Emoji and color
    emoji_map = {
        'positive': '😊 POSITIVE',
        'negative': '😞 NEGATIVE',
        'neutral':  '😐 NEUTRAL'
    }
    display_label = emoji_map.get(label.lower(), label.upper())

    # Confidence scores for all labels
    confidence = {r['label'].capitalize(): round(r['score'], 4) for r in results_sorted}

    # Interpretation
    if score > 0.90:
        certainty = "Very confident"
    elif score > 0.75:
        certainty = "Confident"
    elif score > 0.60:
        certainty = "Moderately confident"
    else:
        certainty = "Uncertain"

    detail = f"{certainty} ({score:.1%})"

    return display_label, confidence, detail


# ── Example sentences ──
examples = [
    ["مزيان بزاف هاد الخبر شكراً على المعلومة 👍"],
    ["هاد الحكومة مكتخدمش والو، مقبولش هاد القرار 😡"],
    ["واش صحيح هاد الخبر؟ مفهمتش شي حاجة"],
    ["c'est vraiment bien ce projet, bravo mzyan!"],
    ["هاد المشروع ممتاز ويستحق الدعم والتشجيع"],
    ["لا لا لا، هاد الشي غلط بالكامل، خسارة"],
]

# ── Gradio Interface ──
with gr.Blocks(
    title="Darija Sentiment Analyzer — تحليل المشاعر بالدارجة",
    theme=gr.themes.Soft(primary_hue="blue")
) as demo:

    gr.Markdown("""
    # 🇲🇦 Darija Sentiment Analyzer — تحليل المشاعر بالدارجة
    
    **First open-source sentiment analysis system for Moroccan Arabic (Darija)**
    
    Supports: Arabic script · Arabizi (Franco-Arabic) · Mixed Arabic/French
    
    ---
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter Darija text — اكتب جملة بالدارجة",
                placeholder="مثال: مزيان بزاف هاد الخبر... / c'est vraiment bien...",
                lines=4,
                rtl=True
            )
            analyze_btn = gr.Button("🔍 Analyze Sentiment — تحليل المشاعر",
                                    variant="primary", size="lg")

        with gr.Column(scale=1):
            label_output = gr.Label(label="Sentiment Prediction")
            confidence_output = gr.Label(label="Confidence Scores", num_top_classes=3)
            detail_output = gr.Textbox(label="Certainty Level", interactive=False)

    gr.Examples(
        examples=examples,
        inputs=text_input,
        label="📝 Example sentences — أمثلة"
    )

    gr.Markdown("""
    ---
    ### 📊 About This Model
    
    | | |
    |---|---|
    | **Architecture** | CAMeL-BERT (fine-tuned) |
    | **Dataset** | 8,619 Darija comments (Kaggle + HuggingFace + Hespress) |
    | **Labels** | Positive · Negative · Neutral |
    | **Author** | Khalid Morjan — Sultan Moulay Slimane University |
    | **GitHub** | [KHALIDMRJ/Darija-Sentiment-Analysis](https://github.com/KHALIDMRJ) |
    """)

    analyze_btn.click(
        fn=analyze_sentiment,
        inputs=text_input,
        outputs=[label_output, confidence_output, detail_output]
    )
    text_input.submit(
        fn=analyze_sentiment,
        inputs=text_input,
        outputs=[label_output, confidence_output, detail_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
