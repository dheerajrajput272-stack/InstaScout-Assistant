"""
filter.py
Filtering and query orchestration engine for Instagram creator profile discovery.
Applies threshold criteria, follower tiers, verification checks, and sorting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from .analyzer import ProfileAnalyzer


# Standard creator follower tiers
FOLLOWER_TIERS = {
    "All Tiers": (0, 100_000_000),
    "Nano (< 10K)": (0, 10_000),
    "Micro (10K - 50K)": (10_000, 50_000),
    "Mid-Tier (50K - 200K)": (50_000, 200_000),
    "Macro (200K - 1M)": (200_000, 1_000_000),
    "Mega (1M+)": (1_000_000, 100_000_000)
}


class ProfileFilter:
    """
    Filtering and query orchestration engine.
    Filters profiles dynamically based on relevance thresholds,
    location constraints, follower counts, and verification badges.
    """

    def __init__(self, profiles: Optional[List[Dict[str, Any]]] = None):
        self.profiles: List[Dict[str, Any]] = profiles or []
        self.analyzer = ProfileAnalyzer()

    @classmethod
    def load_from_json(cls, filepath: str | Path) -> "ProfileFilter":
        """Load profiles from a JSON dataset."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Profiles dataset not found at {filepath}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(profiles=data)

    @classmethod
    def load_from_csv(cls, filepath: str | Path) -> "ProfileFilter":
        """Load profiles from a CSV dataset."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Profiles dataset not found at {filepath}")
        df = pd.read_csv(path)
        records = df.to_dict(orient="records")
        # Parse recent_posts_tags if comma-separated
        for r in records:
            if isinstance(r.get("recent_posts_tags"), str):
                r["recent_posts_tags"] = [t.strip() for t in r["recent_posts_tags"].split(",") if t.strip()]
        return cls(profiles=records)

    def filter_profiles(
        self,
        target_niche: Optional[str] = None,
        target_location: Optional[str] = None,
        target_keywords: Optional[List[str] | str] = None,
        min_relevance_pct: float = 0.0,
        follower_tier: str = "All Tiers",
        verified_only: bool = False,
        min_engagement: float = 0.0,
        sort_by: str = "Relevance (High to Low)"
    ) -> List[Dict[str, Any]]:
        """
        Execute full discovery pipeline:
        1. Batch analyze profiles with relevance scoring engine
        2. Apply minimum relevance threshold
        3. Filter by follower count tier
        4. Filter by verification badge if requested
        5. Filter by engagement rate
        6. Sort according to specified criteria
        """
        # Run analyzer
        scored_profiles = self.analyzer.batch_analyze(
            profiles=self.profiles,
            target_niche=target_niche,
            target_location=target_location,
            target_keywords=target_keywords
        )

        min_followers, max_followers = FOLLOWER_TIERS.get(follower_tier, (0, 100_000_000))

        filtered: List[Dict[str, Any]] = []
        for p in scored_profiles:
            # Min relevance check
            if p["relevance_score"] < min_relevance_pct:
                continue

            # Follower tier check
            followers = p.get("follower_count", 0)
            if not (min_followers <= followers <= max_followers):
                continue

            # Verified check
            if verified_only and not p.get("verified", False):
                continue

            # Engagement check
            eng = float(p.get("engagement_rate", 0))
            if eng < min_engagement:
                continue

            filtered.append(p)

        # Apply sorting
        if sort_by == "Relevance (High to Low)":
            filtered.sort(key=lambda x: x["relevance_score"], reverse=True)
        elif sort_by == "Followers (High to Low)":
            filtered.sort(key=lambda x: x.get("follower_count", 0), reverse=True)
        elif sort_by == "Followers (Low to High)":
            filtered.sort(key=lambda x: x.get("follower_count", 0))
        elif sort_by == "Engagement Rate (High to Low)":
            filtered.sort(key=lambda x: x.get("engagement_rate", 0), reverse=True)

        return filtered

    @staticmethod
    def to_dataframe(profiles: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert filtered profile results into a clean, display-ready DataFrame.
        """
        rows = []
        for p in profiles:
            rel = p.get("relevance", {})
            rows.append({
                "Avatar": p.get("avatar_emoji", "👤"),
                "Username": f"@{p.get('username')}",
                "Name": p.get("name"),
                "Category": p.get("category"),
                "Location": p.get("location"),
                "Followers": p.get("follower_count", 0),
                "Engagement": f"{p.get('engagement_rate', 0)}%",
                "Verified": "Verified" if p.get("verified") else "Standard",
                "Relevance Score": f"{p.get('relevance_score', 0):.1f}%",
                "Score Float": p.get("relevance_score", 0),
                "Matched Keywords": ", ".join(p.get("matched_keywords", [])) or "None",
                "Niche Score": f"{rel.get('niche_score', 0):.0f}%",
                "Keyword Score": f"{rel.get('keyword_score', 0):.0f}%",
                "Location Score": f"{rel.get('location_score', 0):.0f}%",
                "Bio Score": f"{rel.get('bio_relevance_score', 0):.0f}%",
                "Bio": p.get("bio", ""),
                "External Link": p.get("external_link", "")
            })
        return pd.DataFrame(rows)
