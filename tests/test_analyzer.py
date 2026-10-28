"""
test_analyzer.py
Pytest suite verifying the relevance scoring engine, keyword matching logic,
text processing, semantic expansion, and filtering rules.
"""

import pytest
from pathlib import Path
from modules.processor import TextProcessor, SemanticExpander, NICHE_ONTOLOGY
from modules.analyzer import ProfileAnalyzer, RelevanceScore
from modules.filter import ProfileFilter


# Sample mock profiles for isolated deterministic testing
MOCK_FITNESS_PROFILE = {
    "id": "test_01",
    "username": "fit_alex",
    "name": "Alex Fitness Coach",
    "category": "Fitness",
    "location": "Los Angeles, CA, USA",
    "city": "Los Angeles",
    "country": "USA",
    "follower_count": 85000,
    "following_count": 500,
    "posts_count": 300,
    "engagement_rate": 4.5,
    "verified": True,
    "bio": "Certified fitness coach & strength trainer. Daily gym workouts and sports nutrition tips.",
    "recent_posts_tags": ["gym", "workout", "fitness", "trainer", "nutrition"],
    "external_link": "https://alexfit.com"
}

MOCK_FOOD_PROFILE = {
    "id": "test_02",
    "username": "chef_marco",
    "name": "Marco Rossi Chef",
    "category": "Food Creator",
    "location": "Rome, Italy",
    "city": "Rome",
    "country": "Italy",
    "follower_count": 120000,
    "following_count": 400,
    "posts_count": 450,
    "engagement_rate": 3.8,
    "verified": False,
    "bio": "Artisan handmade pasta & authentic Italian recipes. Culinary reviews and kitchen secrets.",
    "recent_posts_tags": ["pasta", "recipes", "culinary", "chef", "italianfood"],
    "external_link": "https://marcopasta.it"
}

MOCK_TECH_PROFILE = {
    "id": "test_03",
    "username": "code_sarah",
    "name": "Sarah Chen AI Engineer",
    "category": "Tech",
    "location": "Seattle, WA, USA",
    "city": "Seattle",
    "country": "USA",
    "follower_count": 45000,
    "following_count": 250,
    "posts_count": 200,
    "engagement_rate": 5.2,
    "verified": True,
    "bio": "Python developer and AI researcher. Building machine learning agents and coding tutorials.",
    "recent_posts_tags": ["python", "ai", "coding", "developer", "machinelearning"],
    "external_link": "https://sarahchen.dev"
}


class TestTextProcessor:
    """Test text cleaning and tokenization."""

    def test_clean_text_removes_urls_and_punctuation(self):
        raw = "Check out https://example.com/fit & my new #workout routine! 🏋️‍♂️"
        cleaned = TextProcessor.clean_text(raw)
        assert "http" not in cleaned
        assert "example" not in cleaned
        assert "workout" in cleaned
        assert "routine" in cleaned

    def test_tokenize_filters_stopwords(self):
        text = "This is a great gym workout for the new athlete"
        tokens = TextProcessor.tokenize(text, remove_stopwords=True)
        assert "this" not in tokens
        assert "is" not in tokens
        assert "for" not in tokens
        assert "gym" in tokens
        assert "workout" in tokens
        assert "athlete" in tokens

    def test_extract_hashtags(self):
        text = "Morning routine #fitness #gym_life #health2026"
        tags = TextProcessor.extract_hashtags(text)
        assert "fitness" in tags
        assert "gym_life" in tags
        assert "health2026" in tags


class TestSemanticExpander:
    """Test semantic keyword expansion and ontology."""

    def test_expand_keywords_fitness_synonyms(self):
        expander = SemanticExpander()
        res = expander.expand_keywords("gym, workout", niche="Fitness")
        assert "gym" in res["original_keywords"]
        assert "workout" in res["original_keywords"]
        # Synonyms like strength, fitness, exercise should be included
        expanded = res["expanded_keywords"]
        assert any(term in expanded for term in ["fitness", "strength", "training", "exercise"])

    def test_expand_keywords_tech_synonyms(self):
        expander = SemanticExpander()
        res = expander.expand_keywords("coding", niche="Tech")
        expanded = res["expanded_keywords"]
        assert "coding" in res["original_keywords"]
        assert any(term in expanded for term in ["programming", "developer", "software", "python"])


class TestProfileAnalyzer:
    """Test the 4-component weighted scoring algorithm."""

    def setup_method(self):
        self.analyzer = ProfileAnalyzer()

    def test_niche_exact_match(self):
        score, matched = self.analyzer.evaluate_niche_match("Fitness", "Fitness", "Workout bio")
        assert score == 100.0
        assert matched is True

    def test_niche_mismatch(self):
        score, matched = self.analyzer.evaluate_niche_match("Tech", "Fitness", "Workout bio")
        assert score == 0.0
        assert matched is False

    def test_niche_unspecified_neutral(self):
        score, matched = self.analyzer.evaluate_niche_match(None, "Fitness", "Workout bio")
        assert score == 100.0
        assert matched is True

    def test_location_match_city(self):
        score, match_type = self.analyzer.evaluate_location_match(
            "Los Angeles",
            "Los Angeles, CA, USA",
            profile_city="Los Angeles",
            profile_country="USA"
        )
        assert score == 100.0
        assert "Exact City Match" in match_type

    def test_location_match_country(self):
        score, match_type = self.analyzer.evaluate_location_match(
            "USA",
            "Los Angeles, CA, USA",
            profile_city="Los Angeles",
            profile_country="USA"
        )
        assert score >= 85.0

    def test_location_mismatch(self):
        score, match_type = self.analyzer.evaluate_location_match(
            "Tokyo",
            "Los Angeles, CA, USA",
            profile_city="Los Angeles",
            profile_country="USA"
        )
        assert score == 0.0
        assert "No Location Match" in match_type

    def test_keyword_coverage_and_matched_list(self):
        score, matched_kws = self.analyzer.evaluate_keyword_coverage(
            target_keywords=["gym", "nutrition", "workout"],
            profile=MOCK_FITNESS_PROFILE,
            target_niche="Fitness"
        )
        assert score > 75.0
        assert "gym" in matched_kws
        assert "nutrition" in matched_kws
        assert "workout" in matched_kws

    def test_overall_weighted_score_formula(self):
        """
        Verify exact weighted formula:
        Total = 0.35 * Niche + 0.30 * Keyword + 0.20 * Location + 0.15 * BioSemantic
        """
        result = self.analyzer.analyze_profile(
            profile=MOCK_FITNESS_PROFILE,
            target_niche="Fitness",
            target_location="Los Angeles",
            target_keywords=["gym", "workout"]
        )
        rel = result["relevance"]
        expected = (
            0.35 * rel["niche_score"] +
            0.30 * rel["keyword_score"] +
            0.20 * rel["location_score"] +
            0.15 * rel["bio_relevance_score"]
        )
        assert abs(result["relevance_score"] - round(expected, 1)) < 0.01
        assert result["relevance_score"] >= 80.0  # High match
        assert "gym" in result["matched_keywords"]
        assert "workout" in result["matched_keywords"]

    def test_batch_analyze_orders_by_relevance_descending(self):
        profiles = [MOCK_FOOD_PROFILE, MOCK_FITNESS_PROFILE, MOCK_TECH_PROFILE]
        ranked = self.analyzer.batch_analyze(
            profiles=profiles,
            target_niche="Fitness",
            target_location="Los Angeles",
            target_keywords=["gym", "strength"]
        )
        # Top profile should be MOCK_FITNESS_PROFILE
        assert ranked[0]["username"] == "fit_alex"
        assert ranked[0]["relevance_score"] > ranked[1]["relevance_score"]
        assert ranked[0]["relevance_score"] > ranked[2]["relevance_score"]


class TestProfileFilter:
    """Test filtering by threshold, followers, verification, and dataset loading."""

    def test_filter_by_min_relevance_threshold(self):
        profiles = [MOCK_FITNESS_PROFILE, MOCK_FOOD_PROFILE, MOCK_TECH_PROFILE]
        f = ProfileFilter(profiles)
        results = f.filter_profiles(
            target_niche="Fitness",
            target_location="Los Angeles",
            target_keywords=["gym"],
            min_relevance_pct=70.0
        )
        # Only MOCK_FITNESS_PROFILE should satisfy >= 70%
        assert len(results) == 1
        assert results[0]["username"] == "fit_alex"

    def test_filter_by_follower_tier(self):
        profiles = [MOCK_FITNESS_PROFILE, MOCK_FOOD_PROFILE, MOCK_TECH_PROFILE]
        f = ProfileFilter(profiles)
        # Sarah Chen has 45k (Micro: 10k-50k)
        micro_results = f.filter_profiles(follower_tier="Micro (10K - 50K)")
        usernames = [p["username"] for p in micro_results]
        assert "code_sarah" in usernames
        assert "chef_marco" not in usernames  # 120k is mid-tier

    def test_filter_by_verified_only(self):
        profiles = [MOCK_FITNESS_PROFILE, MOCK_FOOD_PROFILE, MOCK_TECH_PROFILE]
        f = ProfileFilter(profiles)
        verified_results = f.filter_profiles(verified_only=True)
        assert all(p["verified"] is True for p in verified_results)
        assert len(verified_results) == 2

    def test_load_sample_dataset_files(self):
        json_path = Path("data/sample_profiles.json")
        csv_path = Path("data/sample_profiles.csv")
        assert json_path.exists()
        assert csv_path.exists()

        f_json = ProfileFilter.load_from_json(json_path)
        assert len(f_json.profiles) >= 15

        f_csv = ProfileFilter.load_from_csv(csv_path)
        assert len(f_csv.profiles) >= 15
