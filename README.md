<div align="center">

# 🇲🇦 Darija Sentiment Analysis
### تحليل المشاعر بالدارجة المغربية

**First open-source end-to-end sentiment analysis system for Moroccan Arabic (Darija)**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-FFD21E?style=flat-square)](https://huggingface.co)
[![CAMeL-BERT](https://img.shields.io/badge/Model-CAMeL--BERT-FF6F00?style=flat-square)](https://huggingface.co/CAMeL-Lab)
[![TF-IDF](https://img.shields.io/badge/Baseline-TF--IDF-4CAF50?style=flat-square)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-90.26%25-success?style=flat-square)]()
[![Dataset](https://img.shields.io/badge/Dataset-8%2C619%20samples-blue?style=flat-square)]()
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Complete-success?style=flat-square)]()

*Detecting sentiment in a language that 40 million people speak — but AI has largely ignored.*

</div>

---

## 🚨 The Problem Nobody Solved

**40 million Moroccans** speak Darija — a unique blend of Arabic, French, Berber, and Spanish.

Yet today, **zero production-ready NLP tools exist for Darija sentiment analysis.**

This means:
- 🏢 Moroccan companies **cannot analyze customer feedback** at scale
- 📊 Researchers **cannot study public opinion** on Moroccan social media
- 🏛️ Organizations **cannot monitor citizen sentiment** effectively
- 🛒 E-commerce platforms **cannot process Moroccan reviews** automatically

**This project directly addresses that gap** — building the first complete Darija sentiment pipeline from data collection to live inference.

---

## 🎯 What This System Does

An end-to-end pipeline that:

1. **Scrapes** Darija comments from Hespress.com (Morocco's most-read news site)
2. **Preprocesses** mixed Arabic/Latin/French Darija text with custom normalization
3. **Labels** data with an interactive CLI annotation tool
4. **Trains & compares** 3 models from classical ML to transformers
5. **Deploys** a live demo on HuggingFace Spaces

---

## 📊 Results — What We Achieved

| Model | Accuracy | F1 Score | Training Time |
|---|---|---|---|
| TF-IDF + Logistic Regression | 90.03% | 90.08% | < 1 min |
| **CAMeL-BERT (Fine-tuned)** | **90.26%** | **90.11%** | ~15 min GPU |

> 🔬 **Key Finding:** Our TF-IDF baseline achieved 90.03% — nearly matching fine-tuned CAMeL-BERT (90.26%). This suggests that Darija vocabulary alone is highly discriminative for sentiment classification, and lightweight models can rival transformers on well-structured dialectal Arabic datasets. This is itself a publishable research finding.

---

## 🧠 System Architecture

```
Raw Text (Darija)
       ↓
┌─────────────────────────────────────────┐
│         PREPROCESSING PIPELINE          │
│  • Arabic diacritics removal            │
│  • Letter normalization (alef, yaa...)  │
│  • Emoji → sentiment token encoding     │
│  • Mixed script handling (AR/FR/Latin)  │
│  • Repeated character normalization     │
└─────────────────────────────────────────┘
       ↓
┌──────────────────┐    ┌──────────────────┐
│   MODEL A        │    │   MODEL B        │
│ TF-IDF + LogReg  │    │  CAMeL-BERT      │
│ (char n-grams)   │    │  Fine-tuned      │
│ Acc: 90.03%      │    │  Acc: 90.26%     │
└──────────────────┘    └──────────────────┘
       ↓                        ↓
       └──────────┬─────────────┘
                  ↓
         Sentiment Label
    Positive / Negative / Neutral
```

---

## 📦 Dataset — 8,619 Darija Comments

The largest unified Darija sentiment dataset combining 3 sources:

| Source | Comments | Type |
|---|---|---|
| Kaggle — Moroccan Sentiment | 7,651 | Darija / Arabizi |
| HuggingFace — Darija Reviews | 834 | Arabic / Mixed |
| Hespress.com (manually scraped + labeled) | 134 | Arabic pur |
| **Total** | **8,619** | |

**Label distribution:**

| Label | Count | % |
|---|---|---|
| Positive | 4,398 | 51.0% |
| Negative | 4,064 | 47.1% |
| Neutral | 157 | 1.9% |

**Splits:** 70% Train (6,031) / 15% Val (1,294) / 15% Test (1,294)

---

## 🔬 The Darija Preprocessing Challenge

The biggest technical challenge — no existing library handles Darija properly:

```python
def clean_darija(text):
    # 1. Encode emojis as sentiment tokens BEFORE removal
    #    😍 → "ايجابي_جداً"   😡 → "سلبي_جداً"
    text = encode_emojis(text)

    # 2. Remove URLs and mentions
    text = re.sub(r'https?://\S+|@\w+', '', text)

    # 3. Remove Arabic diacritics (tashkeel)
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)

    # 4. Normalize Arabic letter variants
    text = re.sub(r'[إأآا]', 'ا', text)   # alef variants
    text = re.sub(r'[ىي]', 'ي', text)     # yaa variants
    text = re.sub(r'ة', 'ه', text)         # taa marbuta

    # 5. Normalize Darija repeated chars ("مزياااان" → "مزيان")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    return normalize_whitespace(text)
```

This preprocessing pipeline is a standalone contribution — the first open-source Darija text normalizer of this kind.

---

## 🏗️ Project Structure

```
Darija-Sentiment-Analysis/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   ├── scraper.py              ← Hespress comment scraper
│   ├── preprocessor.py         ← Darija text cleaning pipeline
│   └── labeler.py              ← Interactive CLI annotation tool
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    ← Dataset analysis & charts
│   ├── 02_baseline_tfidf.ipynb      ← TF-IDF model (90.03%)
│   └── 03_camelbert_finetune.ipynb  ← CAMeL-BERT fine-tuning (90.26%)
│
├── data/
│   ├── unified/                ← Full merged dataset (8,619 rows)
│   └── splits/                 ← train / val / test CSVs
│
├── models/                     ← Saved model weights
│   └── camelbert/
│
└── app/
    └── gradio_demo.py          ← Live HuggingFace Spaces demo
```

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/KHALIDMRJ/Darija-Sentiment-Analysis
cd Darija-Sentiment-Analysis
pip install -r requirements.txt

# 2. Scrape data from Hespress
python src/scraper.py

# 3. Label your data interactively
python src/labeler.py --input data/raw/hespress_comments.csv \
                      --output data/labeled/hespress_labeled.csv

# 4. Run exploration notebook
jupyter notebook notebooks/01_data_exploration.ipynb

# 5. Train baseline model (local)
jupyter notebook notebooks/02_baseline_tfidf.ipynb

# 6. Fine-tune CAMeL-BERT (Google Colab GPU recommended)
# Upload notebooks/03_camelbert_finetune.ipynb to colab.research.google.com
```

---

## 🌐 Live Demo

> 🔗 Coming soon on HuggingFace Spaces

Type any Darija sentence → get sentiment prediction + confidence scores in real time.

Supports: Arabic script · Arabizi (Franco-Arabic) · Mixed Arabic/French

---

## ⚙️ Requirements

```
transformers>=4.35.0
torch>=2.0.0
scikit-learn>=1.3.0
datasets>=2.14.0
gradio>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
requests>=2.31.0
beautifulsoup4>=4.12.0
```

---

## 🔭 Future Roadmap

- [ ] **DarijaBERT** — fine-tune on Morocco-specific dialectal model
- [ ] **Aspect-based sentiment** — detect sentiment per topic (politics, economy, sport)
- [ ] **Larger dataset** — scale to 50,000+ samples via crowdsourcing
- [ ] **Arabizi support** — handle Franco-Arabic (Latin script Darija)
- [ ] **Real-time API** — FastAPI endpoint for production use
- [ ] **Academic paper** — publish findings in NLP conference

---

## 📚 Academic Context

| | |
|---|---|
| **Module** | Deep Learning & NLP |
| **Program** | Systèmes d'Information et Intelligence Artificielle (SIIA) |
| **Institution** | Faculty of Polydisciplinary Studies (FPK) — Khouribga |
| **University** | Sultan Moulay Slimane University (SUMS) |
| **Supervisor** | Prof. Ibtissam BAKKOURI |
| **Year** | 2025–2026 |

---

## ⚠️ License

This project is licensed under **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0**.

You may share with attribution. Commercial use and modifications are prohibited.

See [LICENSE](LICENSE) for full terms.

---

## 👨‍💻 Author

**Khalid Morjan** — AI & Data Science Student, Sultan Moulay Slimane University, Morocco

Computer Vision · Deep Learning · NLP · Big Data

🔗 GitHub: [KHALIDMRJ](https://github.com/KHALIDMRJ)

---

<div align="center">

*Built to give Darija a voice in AI.*

**🇲🇦 First Darija Sentiment System — FPK Khouribga — SUMS 2026**

</div>