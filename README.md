# InstaScout AI — Instagram Profile Discovery & Filtering Assistant

An AI-powered creator intelligence engine and interactive Streamlit dashboard built in Python. InstaScout helps brand marketers, agencies, and researchers discover, analyze, and filter public Instagram creator profiles based on niche taxonomy, geographic location, and intelligent semantic keyword expansion.

---

## 🌟 Key Features

1. **Intelligent Semantic Keyword Expansion (`processor.py`)**:
   - Bridges the creator vocabulary gap using domain-specific NLP ontologies and synonym graphs (e.g. mapping `"Fitness"` to `"gym, workout, trainer, wellness, health, bodybuilding, calisthenics, nutrition, hiit"`).
   - Stopword filtering, emoji normalization, and alphanumeric tokenization.
   - Vector-space cosine similarity using `scikit-learn` `TfidfVectorizer` (sublinear TF scaling).

2. **Multi-Factor Weighted Relevance Engine (`analyzer.py`)**:
   - Computes a transparent, composite match percentage ($0 - 100\%$) across 4 weighted pillars:
     - 🎯 **Niche Match ($35\%$)**: Exact category matches, ontology ancestors, and bio keyword presence.
     - 🔑 **Keyword Coverage & Frequency ($30\%$)**: Original keyword matches plus semantic synonym bonuses.
     - 📍 **Location Match ($20\%$)**: Exact city, regional/country, and tokenized substring matching.
     - 🧠 **Bio & Content Semantic Similarity ($15\%$)**: TF-IDF cosine similarity across bios, tags, and creator engagement credibility.

3. **Dynamic Filtering & Tiers (`filter.py`)**:
   - Filter by minimum relevance percentage threshold ($0\% - 100\%$).
   - Follower tier filtering: Nano ($<10\text{k}$), Micro ($10\text{k}-50\text{k}$), Mid-Tier ($50\text{k}-200\text{k}$), Macro ($200\text{k}-1\text{M}$), Mega ($1\text{M}+$).
   - Minimum engagement rate filter & verified account filter.
   - Dynamic sorting by relevance, followers, or engagement rate.

4. **Production-Grade Streamlit Dashboard (`app.py`)**:
   - Glassmorphic UI with responsive card grids and visual score badges.
   - 1-click preset scenarios (LA Fitness Coaches, European Food Creators, AI Developers, etc.).
   - Detailed per-creator score audit expanders with progress bars.
   - Interactive structured table with one-click **CSV** and **JSON** exports.
   - Real-time NLP query expansion inspector and knowledge graph preview.

---

## 🏗️ Architecture (B.L.A.S.T. Framework)

```
AI_Insta/
├── app.py                      # Interactive Streamlit Web Dashboard
├── data/
│   ├── sample_profiles.json    # Mock public profiles dataset (20 profiles)
│   └── sample_profiles.csv     # CSV representation for data interoperability
├── modules/
│   ├── __init__.py             # Clean package exports
│   ├── processor.py            # Text cleaning, tokenization & semantic NLP expander
│   ├── analyzer.py             # 4-component weighted relevance scoring engine
│   └── filter.py               # Filtering pipeline, follower tiers & DataFrame export
├── tests/
│   └── test_analyzer.py        # 18 automated Pytest unit and integration tests
├── pytest.ini                  # Pytest environment configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation & setup guide
```

---

## 🧮 Weighted Scoring Formula

$$\text{Relevance Score} = (0.35 \times \text{Niche}) + (0.30 \times \text{Keyword}) + (0.20 \times \text{Location}) + (0.15 \times \text{BioSim})$$

| Component | Weight | Scoring Logic |
| :--- | :---: | :--- |
| **Niche Alignment** | **35%** | 100% for exact category match; 92% for direct substring; 85% for ontology relationship; 60% for bio mention. |
| **Keyword Coverage** | **30%** | Ratio of matched original user search terms (up to 75 pts) + bonus for matched semantic synonyms (up to 25 pts). |
| **Location Match** | **20%** | 100% for exact city match; 85% for country match; 60–85% for partial region token overlap; 0% if mismatched. |
| **Bio & Content Sim**| **15%** | Sublinear TF-IDF cosine similarity between search query and creator bio/tags, boosted by verified status and high engagement. |

---

## 🚀 Running Locally

### 1. Prerequisites
- Python 3.10+ (Python 3.13 supported)
- Git / Terminal

### 2. Setup Virtual Environment & Install Dependencies
```bash
# Clone or navigate to the repository
cd AI_Insta

# Create a virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Then open your browser to **[http://localhost:8501](http://localhost:8501)**.

---

## 🧪 Running Automated Tests

Run the complete Pytest suite (18 unit and integration tests):
```bash
pytest -v
```

All 18 tests cover:
- Text normalization, URL stripping, and stopword filtering.
- Domain ontology and synonym keyword expansions.
- Exact, partial, and mismatched niche and location evaluations.
- Mathematical precision of the weighted score formula.
- Follower tier and verified creator threshold filtering.
