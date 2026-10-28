"""
analyzer.py
Relevance scoring engine for Instagram creator profile discovery.
Calculates weighted composite scores based on niche match, keyword overlap,
location alignment, and semantic text similarity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from .processor import TextProcessor, SemanticExpander, NICHE_ONTOLOGY


@dataclass
class RelevanceScore:
    """Structured representation of a profile's relevance score and audit trail."""
    total_score: float
    percentage_str: str
    niche_score: float
    keyword_score: float
    location_score: float
    bio_relevance_score: float
    matched_keywords: List[str]
    matched_niche: bool
    location_match_type: str
    summary_explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProfileAnalyzer:
    """
    Scoring engine calculating a weighted relevance score based on:
    - Niche Exact/Semantic Match (Weight: 35%)
    - Keyword Coverage / Frequency (Weight: 30%)
    - Location Match (Weight: 20%)
    - Bio / Category Semantic Relevance (Weight: 15%)
    """

    WEIGHT_NICHE: float = 0.35
    WEIGHT_KEYWORD: float = 0.30
    WEIGHT_LOCATION: float = 0.20
    WEIGHT_BIO_RELEVANCE: float = 0.15

    def __init__(self, expander: Optional[SemanticExpander] = None):
        self.expander = expander or SemanticExpander()

    def evaluate_niche_match(
        self,
        target_niche: Optional[str],
        profile_category: str,
        profile_bio: str
    ) -> tuple[float, bool]:
        """
        Evaluate niche match (0 - 100).
        Weight: 35%
        """
        if not target_niche or target_niche.strip().lower() in ["", "all", "any"]:
            # Neutral / no penalty if not filtering by niche
            return 100.0, True

        target_norm = TextProcessor.clean_text(target_niche)
        category_norm = TextProcessor.clean_text(profile_category)
        bio_norm = TextProcessor.clean_text(profile_bio)

        # 1. Exact match
        if target_norm == category_norm:
            return 100.0, True

        # 2. Direct substring match (e.g. "food" in "food creator")
        if target_norm in category_norm or category_norm in target_norm:
            return 92.0, True

        # 3. Check domain ontology mapping
        matched_in_ontology = False
        target_key = None
        for key in NICHE_ONTOLOGY:
            if key in target_norm or target_norm in key:
                target_key = key
                break

        if target_key:
            # Check if profile category appears in ontology
            category_tokens = TextProcessor.tokenize(category_norm)
            ontology_terms = NICHE_ONTOLOGY[target_key]
            if any(tok in ontology_terms for tok in category_tokens):
                return 85.0, True

            # Check if bio heavily features target ontology terms
            bio_tokens = set(TextProcessor.tokenize(bio_norm))
            bio_overlap = [tok for tok in ontology_terms if tok in bio_tokens]
            if len(bio_overlap) >= 3:
                return 75.0, True
            elif len(bio_overlap) >= 1:
                return 55.0, True

        # 4. Check if target niche word appears directly in bio
        if target_norm in bio_norm:
            return 60.0, True

        return 0.0, False

    def evaluate_keyword_coverage(
        self,
        target_keywords: Optional[List[str] | str],
        profile: Dict[str, Any],
        target_niche: Optional[str] = None
    ) -> tuple[float, List[str]]:
        """
        Evaluate keyword coverage and frequency (0 - 100).
        Weight: 30%
        """
        if not target_keywords:
            return 100.0, []

        expansion_result = self.expander.expand_keywords(target_keywords, niche=target_niche)
        original_kws = expansion_result["original_keywords"]
        expanded_kws = expansion_result["expanded_keywords"]

        if not original_kws and not expanded_kws:
            return 100.0, []

        # Construct profile searchable corpus tokens
        bio_tokens = set(TextProcessor.tokenize(profile.get("bio", "")))
        category_tokens = set(TextProcessor.tokenize(profile.get("category", "")))
        name_tokens = set(TextProcessor.tokenize(profile.get("name", "")))

        # Handle recent post tags
        tags = profile.get("recent_posts_tags", [])
        if isinstance(tags, str):
            tag_tokens = set(t.strip().lower() for t in tags.split(",") if t.strip())
        else:
            tag_tokens = set(str(t).lower() for t in tags)

        profile_all_tokens = bio_tokens | category_tokens | name_tokens | tag_tokens

        matched_original = []
        matched_expanded = []

        # Check original keywords (higher weight)
        for kw in original_kws:
            # Check single word or multi-word phrase
            if kw in profile_all_tokens or kw in TextProcessor.clean_text(profile.get("bio", "")):
                matched_original.append(kw)

        # Check expanded keywords
        for kw in expanded_kws:
            if kw not in matched_original:
                if kw in profile_all_tokens or kw in TextProcessor.clean_text(profile.get("bio", "")):
                    matched_expanded.append(kw)

        all_matched = sorted(list(set(matched_original + matched_expanded)))

        # Base scoring:
        # Original keyword match ratio (up to 75 points)
        orig_ratio = (len(matched_original) / len(original_kws)) if original_kws else 1.0
        score_orig = orig_ratio * 75.0

        # Semantic/expanded keyword bonus (up to 25 points)
        num_expanded_bonus = len(matched_expanded)
        score_expanded = min(25.0, num_expanded_bonus * 5.0)

        total_kw_score = min(100.0, score_orig + score_expanded)

        return round(total_kw_score, 1), all_matched

    def evaluate_location_match(
        self,
        target_location: Optional[str],
        profile_location: str,
        profile_city: str = "",
        profile_country: str = ""
    ) -> tuple[float, str]:
        """
        Evaluate location match (0 - 100).
        Weight: 20%
        """
        if not target_location or target_location.strip().lower() in ["", "all", "global", "any"]:
            return 100.0, "Global / Any"

        t_loc = TextProcessor.clean_text(target_location)
        p_loc = TextProcessor.clean_text(profile_location)
        p_city = TextProcessor.clean_text(profile_city)
        p_country = TextProcessor.clean_text(profile_country)

        # 1. Exact or City match
        if p_city and (t_loc == p_city or t_loc in p_city or p_city in t_loc):
            return 100.0, f"Exact City Match ({profile_city})"

        if t_loc == p_loc or t_loc in p_loc:
            return 100.0, "Exact Location String Match"

        # 2. Country match
        if p_country and (t_loc == p_country or t_loc in p_country or p_country in t_loc):
            return 85.0, f"Country/Region Match ({profile_country})"

        # 3. Substring tokens match
        target_tokens = TextProcessor.tokenize(t_loc, remove_stopwords=False)
        profile_tokens = TextProcessor.tokenize(f"{p_loc} {p_city} {p_country}", remove_stopwords=False)

        overlap = [tok for tok in target_tokens if tok in profile_tokens]
        if overlap:
            overlap_ratio = len(overlap) / len(target_tokens)
            score = 60.0 + (overlap_ratio * 25.0)
            return round(score, 1), f"Partial Match ({', '.join(overlap)})"

        return 0.0, "No Location Match"

    def evaluate_bio_semantic_relevance(
        self,
        query_text: str,
        profile: Dict[str, Any]
    ) -> float:
        """
        Evaluate semantic text similarity between entire query and profile text (0 - 100).
        Weight: 15%
        """
        if not query_text.strip():
            return 100.0

        tags_str = " ".join(profile.get("recent_posts_tags", [])) if isinstance(profile.get("recent_posts_tags"), list) else str(profile.get("recent_posts_tags", ""))
        profile_doc = f"{profile.get('name', '')} {profile.get('category', '')} {profile.get('bio', '')} {tags_str}"

        raw_sim = self.expander.compute_similarity(query_text, profile_doc)

        # Non-linear boost for TF-IDF cosine similarity which naturally peaks around 0.15 - 0.45 in short texts
        # Map 0.0 -> 0.0, 0.15 -> 50.0, 0.35 -> 85.0, 0.5+ -> 100.0
        scaled = min(100.0, raw_sim * 220.0)

        # Add modest engagement & verification signal boost (up to 10 points)
        bonus = 0.0
        if profile.get("verified", False):
            bonus += 5.0
        eng_rate = float(profile.get("engagement_rate", 0))
        if eng_rate >= 4.5:
            bonus += 5.0

        return round(min(100.0, scaled + bonus), 1)

    def analyze_profile(
        self,
        profile: Dict[str, Any],
        target_niche: Optional[str] = None,
        target_location: Optional[str] = None,
        target_keywords: Optional[List[str] | str] = None
    ) -> Dict[str, Any]:
        """
        Analyze an individual profile and calculate weighted relevance score.
        Returns the original profile data augmented with score metrics.
        """
        # 1. Niche match (35%)
        niche_score, matched_niche = self.evaluate_niche_match(
            target_niche=target_niche,
            profile_category=profile.get("category", ""),
            profile_bio=profile.get("bio", "")
        )

        # 2. Keyword coverage (30%)
        keyword_score, matched_kws = self.evaluate_keyword_coverage(
            target_keywords=target_keywords,
            profile=profile,
            target_niche=target_niche
        )

        # 3. Location match (20%)
        location_score, loc_match_type = self.evaluate_location_match(
            target_location=target_location,
            profile_location=profile.get("location", ""),
            profile_city=profile.get("city", ""),
            profile_country=profile.get("country", "")
        )

        # 4. Semantic similarity (15%)
        # Combine search inputs into cohesive query
        query_parts = []
        if target_niche and target_niche.strip().lower() not in ["", "all", "any"]:
            query_parts.append(target_niche)
        if target_keywords:
            if isinstance(target_keywords, list):
                query_parts.extend(target_keywords)
            else:
                query_parts.append(str(target_keywords))
        if target_location and target_location.strip().lower() not in ["", "all", "global", "any"]:
            query_parts.append(target_location)

        query_str = " ".join(query_parts)
        bio_score = self.evaluate_bio_semantic_relevance(query_str, profile)

        # Weighted calculation
        total = (
            (self.WEIGHT_NICHE * niche_score) +
            (self.WEIGHT_KEYWORD * keyword_score) +
            (self.WEIGHT_LOCATION * location_score) +
            (self.WEIGHT_BIO_RELEVANCE * bio_score)
        )
        total = round(total, 1)

        # Summary explanation
        summary_reasons = []
        if matched_niche:
            summary_reasons.append(f"Niche matched ({profile.get('category', '')})")
        if matched_kws:
            summary_reasons.append(f"{len(matched_kws)} keywords matched")
        if "Match" in loc_match_type:
            summary_reasons.append(loc_match_type)

        explanation = "; ".join(summary_reasons) if summary_reasons else "Low alignment with search parameters"

        score_obj = RelevanceScore(
            total_score=total,
            percentage_str=f"{int(round(total))}%",
            niche_score=niche_score,
            keyword_score=keyword_score,
            location_score=location_score,
            bio_relevance_score=bio_score,
            matched_keywords=matched_kws,
            matched_niche=matched_niche,
            location_match_type=loc_match_type,
            summary_explanation=explanation
        )

        result = dict(profile)
        result["relevance"] = score_obj.to_dict()
        result["relevance_score"] = total
        result["relevance_percentage"] = f"{int(round(total))}%"
        result["matched_keywords"] = matched_kws

        return result

    def batch_analyze(
        self,
        profiles: List[Dict[str, Any]],
        target_niche: Optional[str] = None,
        target_location: Optional[str] = None,
        target_keywords: Optional[List[str] | str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze a collection of profiles and sort by total relevance score descending.
        """
        # Fit expander corpus on profile bios for enhanced semantic vectors
        corpus = [
            f"{p.get('name', '')} {p.get('category', '')} {p.get('bio', '')}"
            for p in profiles
        ]
        if not self.expander.fitted:
            self.expander.fit_corpus(corpus)

        analyzed = [
            self.analyze_profile(
                profile=p,
                target_niche=target_niche,
                target_location=target_location,
                target_keywords=target_keywords
            )
            for p in profiles
        ]

        # Sort descending by relevance score
        analyzed.sort(key=lambda x: x["relevance_score"], reverse=True)
        return analyzed
