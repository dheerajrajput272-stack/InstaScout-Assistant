"""
app.py
AI-Powered Instagram Profile Discovery & Filtering Assistant
Interactive Streamlit Dashboard
"""

from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
import pandas as pd
from modules.processor import TextProcessor, SemanticExpander, NICHE_ONTOLOGY, KEYWORD_SYNONYMS
from modules.analyzer import ProfileAnalyzer
from modules.filter import ProfileFilter, FOLLOWER_TIERS


# Page configuration
st.set_page_config(
    page_title="InstaScout AI | Instagram Profile Discovery",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphic, Modern Gradient, Instagram Aesthetic)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1380px;
    }

    /* Header Gradient Title */
    .app-header {
        background: linear-gradient(135deg, rgba(131, 58, 180, 0.15) 0%, rgba(253, 29, 29, 0.12) 50%, rgba(252, 176, 69, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 26px 32px;
        margin-bottom: 25px;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.25);
    }

    .title-gradient {
        background: linear-gradient(90deg, #FA709A 0%, #FEE140 100%, #833AB4 50%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.25rem;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }

    .subtitle-text {
        font-size: 1rem;
        opacity: 0.85;
        font-weight: 500;
        margin-top: 4px;
    }

    /* Metric Card Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 14px;
        padding: 16px 20px;
        display: flex;
        flex-direction: column;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(250, 112, 154, 0.4);
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.7;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        margin-top: 4px;
    }

    /* Profile Cards Grid */
    .profile-card {
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    .profile-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(250, 112, 154, 0.35);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.3);
        transform: translateY(-3px);
    }

    /* Score Badge */
    .badge-score-high {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        font-weight: 800;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.95rem;
        display: inline-flex;
        align-items: center;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
    }
    .badge-score-mid {
        background: linear-gradient(135deg, #3B82F6, #2563EB);
        color: white;
        font-weight: 800;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.95rem;
        display: inline-flex;
        align-items: center;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.35);
    }
    .badge-score-low {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        font-weight: 800;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.95rem;
        display: inline-flex;
        align-items: center;
    }

    /* Tag Pills */
    .tag-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .tag-matched {
        background: rgba(250, 112, 154, 0.18);
        border: 1px solid rgba(250, 112, 154, 0.5);
        color: #FFB3C6;
    }
    .tag-niche {
        background: rgba(131, 58, 180, 0.2);
        border: 1px solid rgba(131, 58, 180, 0.45);
        color: #E0C3FC;
    }

    /* Small code pills */
    .code-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        padding: 2px 7px;
        border-radius: 5px;
        background: rgba(0, 0, 0, 0.25);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_dataset() -> list[dict]:
    """Load sample profiles dataset."""
    json_path = Path("data/sample_profiles.json")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    csv_path = Path("data/sample_profiles.csv")
    if csv_path.exists():
        filter_inst = ProfileFilter.load_from_csv(csv_path)
        return filter_inst.profiles
    return []


# Initialize data and analyzer
raw_profiles = load_dataset()
profile_filter = ProfileFilter(raw_profiles)
semantic_expander = SemanticExpander()


# Preset Discovery Scenarios
PRESETS = {
    "Select a Quick Preset...": {
        "niche": "All",
        "location": "",
        "keywords": "",
        "min_score": 30
    },
    "🏋️‍♂️ Los Angeles Fitness Trainers": {
        "niche": "Fitness",
        "location": "Los Angeles",
        "keywords": "gym, workout, strength, trainer",
        "min_score": 60
    },
    "🍝 European Food & Culinary Creators": {
        "niche": "Food Creator",
        "location": "Rome",
        "keywords": "recipes, chef, gourmet, pasta",
        "min_score": 50
    },
    "💻 AI & Python Tech Developers": {
        "niche": "Tech",
        "location": "",
        "keywords": "ai, python, coding, software",
        "min_score": 55
    },
    "👗 Fashion & Sustainable Streetwear": {
        "niche": "Fashion",
        "location": "New York",
        "keywords": "streetwear, vintage, sustainable, style",
        "min_score": 55
    },
    "🏔️ Alpine & Travel Photographers": {
        "niche": "Travel",
        "location": "",
        "keywords": "photography, adventure, mountains, landscape",
        "min_score": 50
    }
}


# --- SIDEBAR: SEARCH & DISCOVERY CONTROLS ---
with st.sidebar:
    st.markdown("### 🔍 Search & Filters")
    st.markdown("<small style='opacity:0.75;'>Configure discovery parameters</small>", unsafe_allow_html=True)

    # Preset selector
    selected_preset_key = st.selectbox(
        "⚡ Quick Scenarios",
        list(PRESETS.keys()),
        index=0
    )
    preset_data = PRESETS[selected_preset_key]

    # Category / Niche
    available_niches = ["All", "Fitness", "Food Creator", "Tech", "Fashion", "Travel", "Photography", "Wellness"]
    default_niche_idx = available_niches.index(preset_data["niche"]) if preset_data["niche"] in available_niches else 0

    selected_niche = st.selectbox(
        "🎯 Target Niche",
        available_niches,
        index=default_niche_idx,
        help="Primary category of the creator. Exact and semantic niche synonyms are evaluated."
    )

    # Location
    input_location = st.text_input(
        "📍 Target Location",
        value=preset_data["location"],
        placeholder="e.g. Los Angeles, London, Rome, USA",
        help="Target city, state, or country. Supports exact city, regional, and partial matches."
    )

    # Intelligent Keywords
    input_keywords = st.text_input(
        "🔑 Semantic Keywords",
        value=preset_data["keywords"],
        placeholder="e.g. gym, workout, trainer, nutrition",
        help="Comma or space-separated terms. Automatically expanded using domain NLP ontology."
    )

    # Relevance Score Threshold Slider
    st.markdown("---")
    min_relevance = st.slider(
        "📊 Minimum Relevance Score",
        min_value=0,
        max_value=100,
        value=preset_data["min_score"],
        step=5,
        format="%d%%",
        help="Filter out profiles scoring below this weighted relevance threshold."
    )

    # Advanced Filters Expander
    with st.expander("⚙️ Advanced Filters & Sorting", expanded=False):
        follower_tier = st.selectbox(
            "👥 Follower Tier",
            list(FOLLOWER_TIERS.keys()),
            index=0
        )

        min_engagement = st.slider(
            "📈 Min Engagement Rate (%)",
            min_value=0.0,
            max_value=8.0,
            value=0.0,
            step=0.5,
            format="%.1f%%"
        )

        verified_only = st.checkbox(" Verified Creators Only", value=False)

        sort_by = st.selectbox(
            "🔃 Sort Results By",
            [
                "Relevance (High to Low)",
                "Followers (High to Low)",
                "Followers (Low to High)",
                "Engagement Rate (High to Low)"
            ],
            index=0
        )

    # Scoring Engine Weight Configurator
    with st.expander("⚖️ Customize Scoring Weights", expanded=False):
        st.markdown("<small style='opacity:0.8;'>Default: Niche (35%), Keywords (30%), Location (20%), Semantic Bio (15%)</small>", unsafe_allow_html=True)
        w_niche = st.slider("Niche Weight", 0.0, 1.0, 0.35, 0.05)
        w_keyword = st.slider("Keyword Weight", 0.0, 1.0, 0.30, 0.05)
        w_location = st.slider("Location Weight", 0.0, 1.0, 0.20, 0.05)
        w_bio = st.slider("Semantic Bio Weight", 0.0, 1.0, 0.15, 0.05)
        total_w = w_niche + w_keyword + w_location + w_bio
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total_w:.2f} (normalized to 1.0 during scoring)")

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size:0.75rem; opacity:0.6; text-align:center;'>
            Powered by <b>Scikit-Learn NLP</b> & <b>InstaScout Engine</b><br>
            B.L.A.S.T. Architecture v1.0
        </div>
        """,
        unsafe_allow_html=True
    )


# Apply custom weights if modified
profile_filter.analyzer.WEIGHT_NICHE = w_niche / total_w
profile_filter.analyzer.WEIGHT_KEYWORD = w_keyword / total_w
profile_filter.analyzer.WEIGHT_LOCATION = w_location / total_w
profile_filter.analyzer.WEIGHT_BIO_RELEVANCE = w_bio / total_w


# --- EXECUTE DISCOVERY PIPELINE ---
filtered_profiles = profile_filter.filter_profiles(
    target_niche=selected_niche if selected_niche != "All" else None,
    target_location=input_location if input_location.strip() else None,
    target_keywords=input_keywords if input_keywords.strip() else None,
    min_relevance_pct=float(min_relevance),
    follower_tier=follower_tier,
    verified_only=verified_only,
    min_engagement=min_engagement,
    sort_by=sort_by
)

# Semantic expansion inspection for feedback
expansion_info = semantic_expander.expand_keywords(
    keywords=input_keywords,
    niche=selected_niche if selected_niche != "All" else None
)


# --- MAIN INTERFACE: HERO HEADER ---
st.markdown(
    """
    <div class="app-header">
        <div class="title-gradient">InstaScout AI Assistant</div>
        <div class="subtitle-text">
            Autonomous creator discovery, semantic keyword expansion, and multi-factor relevance scoring for Instagram.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# --- METRIC KPI BAR ---
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

total_count = len(raw_profiles)
matched_count = len(filtered_profiles)
avg_score = (
    sum(p["relevance_score"] for p in filtered_profiles) / matched_count
    if matched_count > 0 else 0.0
)
top_creator = filtered_profiles[0]["username"] if matched_count > 0 else "None"
top_score = f"{filtered_profiles[0]['relevance_score']:.1f}%" if matched_count > 0 else "0%"

with col_m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Profiles Scanned</span>
            <span class="metric-value">{total_count}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Qualified Matches</span>
            <span class="metric-value" style="color: #10B981;">{matched_count}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Avg Relevance</span>
            <span class="metric-value" style="color: #3B82F6;">{avg_score:.1f}%</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">Top Match</span>
            <span class="metric-value" style="font-size: 1.35rem; color: #FA709A;">@{top_creator} ({top_score})</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")


# --- ACTIVE QUERY SUMMARY PILLS ---
active_terms = []
if selected_niche != "All":
    active_terms.append(f"<b>Niche:</b> {selected_niche}")
if input_location.strip():
    active_terms.append(f"<b>Location:</b> {input_location}")
if input_keywords.strip():
    active_terms.append(f"<b>Keywords:</b> {input_keywords}")
if min_relevance > 0:
    active_terms.append(f"<b>Min Score:</b> {min_relevance}%")

if active_terms:
    st.markdown(
        f"<div style='margin-bottom: 20px; font-size: 0.88rem; opacity: 0.9;'>🎯 Active Query: {' &bull; '.join(active_terms)}</div>",
        unsafe_allow_html=True
    )


# --- MAIN TABS ---
tab_grid, tab_table, tab_nlp = st.tabs([
    "📇 Discovery Cards Grid",
    "📊 Structured Table & Export",
    "🧠 Semantic NLP & Scoring Audit"
])


# ==============================================================================
# TAB 1: DISCOVERY CARDS GRID (FIXED CLEAN STREAMLIT RENDERING)
# ==============================================================================
with tab_grid:
    if not filtered_profiles:
        st.warning("⚠️ No profiles matched the specified criteria. Try lowering the Minimum Relevance Score or clearing specific keyword constraints.")
    else:
        # Render cards in a 2-column responsive layout using clean Streamlit components
        for i in range(0, len(filtered_profiles), 2):
            col1, col2 = st.columns(2)
            pair = filtered_profiles[i:i+2]

            for idx, p in enumerate(pair):
                target_col = col1 if idx == 0 else col2
                rel = p.get("relevance", {})
                score_num = p.get("relevance_score", 0)

                # Badge class based on score
                if score_num >= 80:
                    badge_label = f"🔥 {p.get('relevance_percentage')}"
                elif score_num >= 55:
                    badge_label = f"✨ {p.get('relevance_percentage')}"
                else:
                    badge_label = f"⚡ {p.get('relevance_percentage')}"

                matched_kws = p.get("matched_keywords", [])
                verified_str = " ✓" if p.get("verified") else ""

                with target_col:
                    with st.container(border=True):
                        # Top row: Avatar, Username, and Score Badge
                        h_col1, h_col2 = st.columns([3, 1])
                        with h_col1:
                            st.markdown(f"### {p.get('avatar_emoji', '👤')} @{p.get('username')}{verified_str}")
                            st.caption(f"{p.get('name')} • **{p.get('category')}**")
                        with h_col2:
                            st.markdown(f"### {badge_label}")

                        # Bio
                        st.markdown(f"**Bio:** {p.get('bio')}")

                        # Metadata
                        st.markdown(
                            f"📍 **{p.get('location')}** | "
                            f"👥 **{p.get('follower_count'):,}** followers | "
                            f"📈 **{p.get('engagement_rate')}%** engagement | "
                            f"📸 **{p.get('posts_count')}** posts"
                        )

                        # Matched Keywords Tags
                        if matched_kws:
                            kw_markdown = " ".join([f"`✓ {k}`" for k in matched_kws])
                            st.markdown(f"**Matched Keywords:** {kw_markdown}")
                        else:
                            st.markdown("**Matched Keywords:** None")

                        # Recent Post Tags
                        recent_tags = p.get("recent_posts_tags", [])[:5]
                        if recent_tags:
                            tag_markdown = " ".join([f"#{t}" for t in recent_tags])
                            st.markdown(f"**Recent Post Tags:** {tag_markdown}")

                        # Expandable score breakdown
                        with st.expander(f"🔎 Score Breakdown for @{p.get('username')}", expanded=False):
                            sc_col1, sc_col2 = st.columns(2)
                            with sc_col1:
                                st.write(f"**Niche Match ({int(profile_filter.analyzer.WEIGHT_NICHE*100)}%):** {rel.get('niche_score', 0):.0f}%")
                                st.progress(min(1.0, rel.get('niche_score', 0) / 100.0))
                                st.write(f"**Location Match ({int(profile_filter.analyzer.WEIGHT_LOCATION*100)}%):** {rel.get('location_score', 0):.0f}%")
                                st.caption(rel.get("location_match_type", ""))
                                st.progress(min(1.0, rel.get('location_score', 0) / 100.0))
                            with sc_col2:
                                st.write(f"**Keyword Coverage ({int(profile_filter.analyzer.WEIGHT_KEYWORD*100)}%):** {rel.get('keyword_score', 0):.0f}%")
                                st.progress(min(1.0, rel.get('keyword_score', 0) / 100.0))
                                st.write(f"**Semantic Bio Relevance ({int(profile_filter.analyzer.WEIGHT_BIO_RELEVANCE*100)}%):** {rel.get('bio_relevance_score', 0):.0f}%")
                                st.progress(min(1.0, rel.get('bio_relevance_score', 0) / 100.0))

                            st.info(f"💡 **Audit Trail:** {rel.get('summary_explanation', 'N/A')}")
                            if p.get("external_link"):
                                st.link_button("🌐 Visit Creator Website", p.get("external_link"), use_container_width=True)


# ==============================================================================
# TAB 2: STRUCTURED TABLE & EXPORT
# ==============================================================================
with tab_table:
    st.markdown("### 📋 Discovery Results Matrix")
    st.markdown("<small style='opacity:0.75;'>Structured view ready for export to CRM or campaign workflows</small>", unsafe_allow_html=True)

    if filtered_profiles:
        df_display = ProfileFilter.to_dataframe(filtered_profiles)

        def color_score(val):
            try:
                score = float(str(val).replace("%", ""))
                if score >= 80:
                    return "background-color: rgba(16, 185, 129, 0.25); color: #6EE7B7; font-weight: bold;"
                elif score >= 55:
                    return "background-color: rgba(59, 130, 246, 0.25); color: #93C5FD; font-weight: bold;"
                else:
                    return "background-color: rgba(245, 158, 11, 0.2); color: #FCD34D;"
            except Exception:
                return ""

        styled_df = df_display[[
            "Avatar", "Username", "Name", "Category", "Location", "Followers",
            "Engagement", "Verified", "Relevance Score", "Matched Keywords",
            "Niche Score", "Keyword Score", "Location Score", "Bio Score"
        ]].style.map(color_score, subset=["Relevance Score"])

        st.dataframe(styled_df, use_container_width=True, height=450)

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            csv_data = df_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name="instascout_discovered_profiles.csv",
                mime="text/csv",
                use_container_width=True
            )
        with dcol2:
            json_export = json.dumps(filtered_profiles, indent=2).encode("utf-8")
            st.download_button(
                label="📥 Download Results as JSON",
                data=json_export,
                file_name="instascout_discovered_profiles.json",
                mime="application/json",
                use_container_width=True
            )
    else:
        st.info("No matching profiles to export.")


# ==============================================================================
# TAB 3: SEMANTIC NLP & SCORING AUDIT
# ==============================================================================
with tab_nlp:
    st.markdown("### 🧠 Semantic NLP & Ontology Intelligence")
    st.markdown(
        """
        The **InstaScout AI Engine** automatically bridges the vocabulary gap between creator bios and search queries.
        Instead of relying on brittle exact text matching, queries are semantically tokenized, stemmed, and expanded.
        """
    )

    nlp_col1, nlp_col2 = st.columns(2)

    with nlp_col1:
        st.markdown("#### 1. Real-Time Query Expansion Breakdown")
        st.write("**Original Cleaned Keywords:**")
        if expansion_info["original_keywords"]:
            st.code(", ".join(expansion_info["original_keywords"]))
        else:
            st.write("*(No keywords provided)*")

        st.write("**Discovered Semantic Synonyms & Expansions:**")
        if expansion_info["expanded_keywords"]:
            exp_pills = " ".join([f"<span class='tag-pill tag-matched'>{kw}</span>" for kw in expansion_info["expanded_keywords"]])
            st.markdown(exp_pills, unsafe_allow_html=True)
        else:
            st.write("*(None)*")

        if expansion_info.get("niche_context_keywords"):
            st.write(f"**Domain Ontology Context for Niche ('{selected_niche}'):**")
            niche_pills = " ".join([f"<span class='tag-pill tag-niche'>{kw}</span>" for kw in expansion_info["niche_context_keywords"]])
            st.markdown(niche_pills, unsafe_allow_html=True)

    with nlp_col2:
        st.markdown("#### 2. Weighted Relevance Formula")
        st.latex(
            r"\text{Score} = (w_1 \cdot \text{Niche}) + (w_2 \cdot \text{Keyword}) + (w_3 \cdot \text{Location}) + (w_4 \cdot \text{BioSim})"
        )
        st.markdown(
            f"""
- **Niche Alignment ($w_1 = {profile_filter.analyzer.WEIGHT_NICHE:.2f}$)**: Exact category match ($100\\%$), ontology ancestor ($85\\%$), bio keyword match ($60\\%$).
- **Keyword Coverage ($w_2 = {profile_filter.analyzer.WEIGHT_KEYWORD:.2f}$)**: Evaluates presence of original search terms ($75\\%$ weight) plus bonus for semantic expanded synonyms ($25\\%$ weight).
- **Location Alignment ($w_3 = {profile_filter.analyzer.WEIGHT_LOCATION:.2f}$)**: City match ($100\\%$), Country match ($85\\%$), Partial token overlap ($60-85\\%$).
- **Semantic Bio Relevance ($w_4 = {profile_filter.analyzer.WEIGHT_BIO_RELEVANCE:.2f}$)**: TF-IDF vector cosine similarity across bios and hashtags, with creator credibility boost.
            """
        )

    st.markdown("---")
    st.markdown("#### 3. Domain Ontology Knowledge Graph Preview")
    onto_tabs = st.tabs(list(NICHE_ONTOLOGY.keys()))
    for idx, (n_name, n_terms) in enumerate(NICHE_ONTOLOGY.items()):
        with onto_tabs[idx]:
            pills = " ".join([f"<span class='tag-pill tag-niche'>{term}</span>" for term in n_terms])
            st.markdown(f"**Associated concepts for `{n_name}`:**", unsafe_allow_html=True)
            st.markdown(pills, unsafe_allow_html=True)