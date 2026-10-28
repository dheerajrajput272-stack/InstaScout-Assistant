"""
processor.py
Text cleaning, tokenization, semantic keyword expansion, and vector similarity.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Set, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Comprehensive standard English stopwords list (zero external downloads needed)
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "dm", "link", "new", "every", "free", "best", "daily", "based"
}

# Domain ontology mapping for popular creator niches
NICHE_ONTOLOGY: Dict[str, List[str]] = {
    "fitness": [
        "gym", "workout", "trainer", "wellness", "health", "bodybuilding", "calisthenics",
        "nutrition", "hiit", "crossfit", "strength", "physique", "athlete", "exercise",
        "pilates", "mobility", "coach", "conditioning", "hypertrophy", "core", "fit"
    ],
    "food creator": [
        "chef", "recipes", "foodie", "baking", "culinary", "restaurant", "cooking",
        "gastronomy", "pastry", "mealprep", "vegan", "gourmet", "tasting", "baker",
        "sourdough", "pasta", "kitchen", "food", "delicious", "artisan"
    ],
    "food": [
        "chef", "recipes", "foodie", "baking", "culinary", "restaurant", "cooking",
        "gastronomy", "pastry", "mealprep", "vegan", "gourmet", "tasting", "kitchen"
    ],
    "tech": [
        "software", "ai", "coding", "developer", "machinelearning", "startup", "python",
        "gadgets", "web3", "engineering", "hardware", "datascience", "saas", "frontend",
        "tech", "programming", "cloud", "code", "engineer"
    ],
    "technology": [
        "software", "ai", "coding", "developer", "machinelearning", "startup", "python",
        "gadgets", "engineering", "hardware", "saas", "tech", "programming"
    ],
    "fashion": [
        "style", "outfit", "ootd", "wardrobe", "runway", "model", "streetwear",
        "couture", "apparel", "vintage", "sustainablefashion", "styling", "chic",
        "lookbook", "fashion", "luxury", "attire", "clothing"
    ],
    "travel": [
        "wanderlust", "nomad", "adventure", "backpacking", "destinations", "explore",
        "photography", "landscape", "hotel", "travelblogger", "mountains", "hiking",
        "travel", "outdoors", "journey", "expedition", "scenic"
    ],
    "photography": [
        "portrait", "cinematography", "camera", "lens", "visuals", "photographer",
        "lightroom", "editorial", "filmphotography", "streetphotography", "35mm",
        "photography", "photo", "film", "shutter"
    ],
    "wellness": [
        "mindfulness", "yoga", "meditation", "selfcare", "holistic", "skincare",
        "health", "breathwork", "cleanbeauty", "wellness", "balance", "healing"
    ]
}

# Granular synonym expansion graph for common discovery keywords
KEYWORD_SYNONYMS: Dict[str, List[str]] = {
    # Fitness & Health
    "gym": ["workout", "fitness", "strength", "training", "exercise", "bodybuilding"],
    "workout": ["training", "gym", "exercise", "fitness", "hiit", "splits", "routines"],
    "trainer": ["coach", "instructor", "mentor", "educator", "specialist"],
    "coach": ["trainer", "instructor", "mentor", "advisor"],
    "strength": ["conditioning", "powerlifting", "hypertrophy", "muscle", "lifting"],
    "pilates": ["mobility", "core", "reformer", "mat", "posture"],
    "calisthenics": ["bodyweight", "handstand", "streetworkout", "gymnastics"],
    "nutrition": ["diet", "mealprep", "supplements", "protein", "healthyfood", "macros"],
    "yoga": ["breathwork", "mindfulness", "asana", "meditation", "vinyasa"],

    # Food & Culinary
    "chef": ["cook", "culinary", "gastronomy", "gourmet", "kitchen"],
    "cooking": ["recipes", "culinary", "baking", "foodprep", "kitchen"],
    "recipes": ["cooking", "baking", "culinary", "meals", "dishes", "foodie"],
    "foodie": ["restaurant", "tasting", "dining", "eats", "culinary"],
    "vegan": ["plantbased", "vegetarian", "dairyfree", "sustainable"],
    "baking": ["pastry", "sourdough", "bread", "dessert", "bakery"],
    "pasta": ["italian", "handmade", "ravioli", "culinary"],

    # Tech & Software
    "coding": ["programming", "developer", "software", "code", "engineering"],
    "developer": ["engineer", "coder", "programmer", "architect", "builder"],
    "ai": ["machinelearning", "artificial intelligence", "llm", "deeplearning", "data"],
    "machinelearning": ["ai", "datascience", "neural", "algorithms", "python"],
    "python": ["coding", "backend", "developer", "datascience", "programming"],
    "startup": ["saas", "founder", "entrepreneur", "bootstrap", "venture"],
    "gadgets": ["hardware", "electronics", "devices", "gear", "tech"],
    "frontend": ["ui", "ux", "webdev", "react", "javascript", "design"],

    # Fashion & Style
    "fashion": ["style", "wardrobe", "couture", "apparel", "clothing"],
    "style": ["fashion", "outfit", "aesthetic", "lookbook", "styling"],
    "streetwear": ["sneakers", "urban", "vintage", "hoodies", "thrift"],
    "ootd": ["outfit", "lookbook", "style", "wardrobe", "attire"],
    "sustainable": ["slowfashion", "ethical", "eco-friendly", "minimalism", "organic"],
    "couture": ["runway", "luxury", "editorial", "designer", "highfashion"],

    # Travel & Photography
    "travel": ["wanderlust", "adventure", "explore", "destinations", "nomad"],
    "nomad": ["remotework", "traveler", "expat", "lifestyle", "backpacking"],
    "photography": ["camera", "visuals", "photo", "cinematography", "portrait"],
    "camera": ["lens", "35mm", "dslr", "gear", "photographer"],
    "mountains": ["hiking", "outdoor", "alpine", "trails", "wilderness"]
}


class TextProcessor:
    """Utility class for cleaning, tokenizing, and normalizing profile text."""

    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        """
        Clean and normalize raw text:
        - Normalize unicode characters (NFKD)
        - Lowercase
        - Remove URLs (http/https/www)
        - Replace punctuation and emojis with spaces while preserving alphanumerics & hashtags
        - Collapse consecutive whitespace
        """
        if not text:
            return ""

        # Normalize unicode
        text = unicodedata.normalize("NFKD", text)

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Replace non-alphanumeric chars (excluding hashtag characters and spaces) with spaces
        text = re.sub(r"[^\w\s#]", " ", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def extract_hashtags(text: Optional[str]) -> List[str]:
        """Extract hashtag words without the '#' prefix."""
        if not text:
            return []
        matches = re.findall(r"#(\w+)", text.lower())
        return [m for m in matches if m]

    @classmethod
    def tokenize(cls, text: Optional[str], remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize cleaned text into words, optionally filtering out stopwords
        and short single-character tokens.
        """
        cleaned = cls.clean_text(text)
        # Extract word sequences of 2 or more letters/digits
        tokens = re.findall(r"[a-z0-9_]{2,}", cleaned)

        if remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]

        return tokens


class SemanticExpander:
    """
    Intelligent semantic keyword expander combining domain ontology,
    synonym maps, and Scikit-Learn TF-IDF vector similarity.
    """

    def __init__(self, corpus: Optional[List[str]] = None):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english"
        )
        self.fitted = False
        if corpus:
            self.fit_corpus(corpus)

    def fit_corpus(self, corpus: List[str]) -> None:
        """Fit the TF-IDF vectorizer over a text corpus (e.g. all profile bios + tags)."""
        clean_corpus = [TextProcessor.clean_text(doc) for doc in corpus if doc]
        if clean_corpus:
            self.vectorizer.fit(clean_corpus)
            self.fitted = True

    def compute_similarity(self, query: str, document: str) -> float:
        """
        Compute TF-IDF cosine similarity between query and document.
        Uses fitted vectorizer if available, otherwise fits a pairwise model dynamically.
        Returns a float between 0.0 and 1.0.
        """
        clean_q = TextProcessor.clean_text(query)
        clean_d = TextProcessor.clean_text(document)

        if not clean_q or not clean_d:
            return 0.0

        try:
            if self.fitted:
                vecs = self.vectorizer.transform([clean_q, clean_d])
                sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
            else:
                # Pairwise fallback
                pair_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
                vecs = pair_vectorizer.fit_transform([clean_q, clean_d])
                sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
            return max(0.0, min(1.0, float(sim)))
        except Exception:
            return 0.0

    @staticmethod
    def normalize_term(term: str) -> str:
        """Strip punctuation and lowercase a single search term."""
        return re.sub(r"[^\w]", "", term.lower().strip())

    def expand_keywords(
        self,
        keywords: List[str] | str,
        niche: Optional[str] = None,
        max_expansions_per_keyword: int = 5
    ) -> Dict[str, object]:
        """
        Expand search keywords using:
        1. Exact synonym mappings
        2. Niche domain ontology keywords
        3. Cross-term expansion

        Returns:
            {
                "original_keywords": [...],
                "expanded_keywords": [...],
                "expansion_map": { kw: [synonyms...] },
                "niche_context_keywords": [...]
            }
        """
        if isinstance(keywords, str):
            # Split on comma or whitespace
            if "," in keywords:
                raw_list = [k.strip() for k in keywords.split(",") if k.strip()]
            else:
                raw_list = [k.strip() for k in keywords.split() if k.strip()]
        else:
            raw_list = list(keywords)

        original_cleaned: List[str] = []
        for kw in raw_list:
            norm = self.normalize_term(kw)
            if norm and norm not in STOPWORDS:
                original_cleaned.append(norm)

        expansion_map: Dict[str, List[str]] = {}
        all_expanded_set: Set[str] = set(original_cleaned)

        for kw in original_cleaned:
            synonyms: List[str] = []
            if kw in KEYWORD_SYNONYMS:
                synonyms = KEYWORD_SYNONYMS[kw][:max_expansions_per_keyword]
            else:
                # Check for substring matches or reverse matches
                for key, syns in KEYWORD_SYNONYMS.items():
                    if kw in key or key in kw:
                        synonyms.extend(syns[:3])
                        break
            expansion_map[kw] = synonyms
            all_expanded_set.update(synonyms)

        # Enrich with niche ontology if niche specified
        niche_context: List[str] = []
        if niche:
            clean_niche = self.normalize_term(niche)
            matched_ontology_key = None
            for key in NICHE_ONTOLOGY:
                if key in clean_niche or clean_niche in key:
                    matched_ontology_key = key
                    break

            if matched_ontology_key:
                niche_context = NICHE_ONTOLOGY[matched_ontology_key]
                all_expanded_set.update(niche_context[:6])

        return {
            "original_keywords": original_cleaned,
            "expanded_keywords": sorted(list(all_expanded_set)),
            "expansion_map": expansion_map,
            "niche_context_keywords": niche_context
        }
