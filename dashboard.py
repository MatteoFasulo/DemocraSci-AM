"""Collaborator dashboard for staged parliamentary evidence search."""

from __future__ import annotations

import os
import re
import unicodedata
from urllib.parse import urlencode

import pandas as pd
import requests
import streamlit as st

from const import N_TOP_BILLS


DEFAULT_SERVICE_URL = os.getenv("DEMOCRASCI_SERVICE_URL", "http://127.0.0.1:8000")
DEFAULT_CLAIM = "Switzerland should pursue a free-trade agreement with the US."
DEFAULT_LANG = "en"


@st.cache_data(ttl=30, show_spinner=False)
def load_health(service_url: str) -> dict:
    response = requests.get(f"{service_url.rstrip('/')}/health", timeout=5)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=3600, show_spinner="Loading speaker names…")
def load_speaker_facets(service_url: str) -> dict:
    response = requests.get(f"{service_url.rstrip('/')}/facets", timeout=120)
    response.raise_for_status()
    return response.json()


def parliament_profile_url(name: str, speaker_id: object) -> str | None:
    if speaker_id in (None, ""):
        return None
    if isinstance(speaker_id, float) and speaker_id.is_integer():
        speaker_id = int(speaker_id)
    normalized_name = unicodedata.normalize("NFKD", name)
    normalized_name = normalized_name.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", normalized_name.lower()).strip("-")
    if not slug:
        return None
    return f"https://www.parlament.ch/{DEFAULT_LANG}/biografie/{slug}/{speaker_id}"


def google_translate_url(text: str, language: object) -> str:
    aliases = {
        "german": "de",
        "deutsch": "de",
        "french": "fr",
        "français": "fr",
        "francais": "fr",
        "italian": "it",
        "italiano": "it",
        "english": "en",
    }
    source = str(language or "").strip().lower()
    source = aliases.get(source, source)
    if source not in {"de", "fr", "it", "en"}:
        source = "auto"
    query = urlencode({"sl": source, "tl": "en", "text": text, "op": "translate"})
    return f"https://translate.google.com/?{query}"


def filtered_passages(records: list[dict], threshold: float) -> pd.DataFrame:
    passages = pd.DataFrame(records)
    if passages.empty:
        return passages
    passages["rerank_score"] = pd.to_numeric(passages["rerank_score"], errors="coerce")
    passages = passages[passages["rerank_score"] >= threshold]
    return passages.sort_values("rerank_score", ascending=False).reset_index(drop=True)


def show_bill_table(records: list[dict]) -> None:
    bills = pd.DataFrame(records)
    if bills.empty:
        st.info("No bills matched this claim and speaker condition.")
        return
    display_rows = []
    for index, row in bills.iterrows():
        affair_id = str(row.get("Affair_ID", "")).strip()
        title = row.get("bill_title_de")
        if pd.isna(title) or not str(title).strip():
            title = row.get("bill_title_fr")
        if pd.isna(title) or not str(title).strip():
            title = row.get("bill_labels")
        title = "Bill title unavailable" if pd.isna(title) or not str(title).strip() else str(title)

        raw_score = pd.to_numeric(row.get("bill_embedding_score"), errors="coerce")
        if pd.notna(raw_score):
            score = float(raw_score)
            filled = max(0, min(10, round(score / 10)))
            score_display = f":green[{('▰' * filled) + ('▱' * (10 - filled))}] **{score / 100:.2f}**"
        else:
            score_display = "—"

        if "speaker_paragraph_count" in bills.columns:
            paragraph_count = pd.to_numeric(row.get("speaker_paragraph_count"), errors="coerce")
            count_display = f"🎙️ **{int(paragraph_count):,}**" if pd.notna(paragraph_count) else "—"
            count_label = "Speaker paragraphs"
        else:
            paragraph_count = pd.to_numeric(row.get("n_paragraphs"), errors="coerce")
            count_display = f"**{int(paragraph_count):,}**" if pd.notna(paragraph_count) else "—"
            count_label = "Paragraphs"

        affair_url = (
            "https://www.parlament.ch/it/ratsbetrieb/suche-curia-vista/geschaeft?"
            f"{urlencode({'AffairId': affair_id})}"
        )
        affair_display = f"[↗ {affair_id}]({affair_url})" if affair_id else "—"
        display_rows.append(
            {
                "#": f":blue-badge[#{index + 1:02d}]",
                "Affair": affair_display,
                "Bill": title,
                "Cosine similarity": score_display,
                count_label: count_display,
            }
        )

    st.table(
        pd.DataFrame(display_rows),
        border="horizontal",
        hide_index=True,
        alt="Ranked bills with Parliament links, relevance scores, and paragraph counts",
    )


def show_paragraph_cards(passages: pd.DataFrame) -> None:
    for rank, (_, row) in enumerate(passages.iterrows(), start=1):
        score = float(row["rerank_score"])
        def present(value: object, fallback: str) -> str:
            return fallback if pd.isna(value) or not str(value).strip() else str(value)

        title = present(
            row.get("bill_title_de"),
            present(row.get("bill_title_fr"), present(row.get("bill_number"), "Parliamentary paragraph")),
        )
        speaker = present(row.get("speaker_full_name"), "Speaker unknown")
        date = present(row.get("meeting_date"), "Date unknown")
        language = present(row.get("language"), "Language unknown")
        text = present(row.get("text"), "")
        links = []
        profile_url = parliament_profile_url(speaker, row.get("speaker_id"))
        if text:
            links.append(
                f"[Translate to English ↗]({google_translate_url(text, row.get('language'))})"
            )

        with st.container(border=True):
            score_col, content_col = st.columns([0.12, 0.88], vertical_alignment="top")
            with score_col:
                st.metric("Rank", f"#{rank}")
                st.metric("Rerank", f"{score:.2f}")
            with content_col:
                st.markdown(f"**{title}**")
                speaker_display = f"[{speaker}]({profile_url})" if profile_url else speaker
                affair_id = present(row.get("Affair_ID"), "—")
                affair_display = (
                    f"[Affair {affair_id}](https://www.parlament.ch/{DEFAULT_LANG}/ratsbetrieb/"
                    f"suche-curia-vista/geschaeft?{urlencode({'AffairId': affair_id})})"
                    if affair_id != "—"
                    else "Affair —"
                )
                st.markdown(
                    f"{speaker_display} · {date} · {language} · {affair_display}",
                    help="Open this speaker's Swiss Parliament biography" if profile_url else None,
                )
                st.markdown(text)
                if links:
                    st.markdown(" · ".join(links))


st.set_page_config(
    page_title="DemocraSci | Parliamentary evidence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown("### Search settings")
    service_url = st.text_input("Model service URL", value=DEFAULT_SERVICE_URL).rstrip("/")
    try:
        health = load_health(service_url)
        st.success(
            f"Service ready · {health.get('bills', 0):,} bills · "
            f"{health.get('paragraphs', 0):,} paragraphs",
            icon="✅",
        )
    except requests.RequestException as exc:
        st.error(f"Model service unavailable: {exc}")

speaker_facets: dict = {}
speaker_error: str | None = None
try:
    speaker_facets = load_speaker_facets(service_url)
except requests.RequestException as exc:
    speaker_error = str(exc)
speakers = speaker_facets.get("speakers", [])
speaker_ids = speaker_facets.get("speaker_ids", {})

heading_col, workflow_col = st.columns([0.76, 0.24])
with heading_col:
    st.markdown('<div class="eyebrow">DemocraSci · Evidence explorer</div>', unsafe_allow_html=True)
    st.title("Parliamentary evidence")
    st.write("Find relevant bills, then inspect the strongest matching speech paragraphs.")
with workflow_col:
    st.markdown("**Workflow**")
    st.caption("① Retrieve bills  →  ② Rerank evidence")

st.markdown("## 1 · Find candidate bills")
with st.container(border=True):
    with st.form("bill_search_form"):
        claim = st.text_input(
            "Claim",
            value=DEFAULT_CLAIM,
            placeholder="Write a policy claim to search for…",
            help="This claim drives both bill retrieval and paragraph reranking.",
        )
        if speaker_error:
            st.warning(
                f"Speaker list unavailable; you can still search without a speaker filter. {speaker_error}"
            )
        speaker_col, bill_limit_col = st.columns([0.62, 0.38])
        with speaker_col:
            selected_speaker = st.selectbox(
                "Speaker (optional)",
                options=speakers,
                index=None,
                placeholder="Search all speakers",
                help="When selected, only bills where this speaker participated are eligible.",
            )
        with bill_limit_col:
            bill_limit = st.number_input(
                "Bills to retrieve",
                min_value=1,
                max_value=1000,
                value=N_TOP_BILLS,
                help="Maximum number of relevant bills returned after speaker preselection.",
            )
        search_bills = st.form_submit_button(
            "Retrieve bills", type="primary", disabled=not claim.strip()
        )

if selected_speaker:
    profile_url = parliament_profile_url(selected_speaker, speaker_ids.get(selected_speaker))
    if profile_url:
        st.caption(f"Selected speaker: {selected_speaker} · [Parliament profile ↗]({profile_url})")

if search_bills:
    try:
        with st.spinner("Finding bills that match the claim…"):
            payload = {
                "retrieval_query": claim.strip(),
                "top_k_bills": int(bill_limit),
            }
            if selected_speaker:
                payload["speaker_name"] = selected_speaker
                payload["min_speaker_paragraphs"] = 1
            response = requests.post(f"{service_url}/bills", json=payload, timeout=3600)
            response.raise_for_status()
        st.session_state["candidate_bills"] = response.json().get("candidate_bills", [])
        st.session_state["search_claim"] = claim.strip()
        st.session_state["search_speaker_name"] = selected_speaker
        st.session_state["paragraph_pool"] = None
        st.session_state["bill_search_error"] = None
    except requests.RequestException as exc:
        st.session_state["bill_search_error"] = str(exc)

if st.session_state.get("bill_search_error"):
    st.error(f"Bill retrieval failed: {st.session_state['bill_search_error']}")

candidate_bills = st.session_state.get("candidate_bills")
if candidate_bills is not None:
    st.markdown("## 2 · Review bills and retrieve paragraphs")
    claim_used = st.session_state.get("search_claim", "")
    selected_name = st.session_state.get("search_speaker_name")
    st.caption(
        f"Claim: “{claim_used}”"
        + (f" · Speaker: {selected_name}" if selected_name else " · All speakers")
    )
    bills_df = pd.DataFrame(candidate_bills)
    metric_cols = st.columns(3)
    metric_cols[0].metric("Candidate bills", f"{len(bills_df):,}")
    if not bills_df.empty:
        top_score = pd.to_numeric(
            bills_df.get("bill_embedding_score", pd.Series(dtype=float)), errors="coerce"
        ).max()
        metric_cols[1].metric(
            "Top bill similarity",
            f"{top_score / 100:.2f}" if pd.notna(top_score) else "—",
            help=(
                "Cosine similarity between the claim and the bill's embedded text. "
                "This is a ranking signal, not a probability or exact-match percentage."
            ),
        )
        speaker_counts = pd.to_numeric(
            bills_df.get("speaker_paragraph_count", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0)
        all_paragraph_counts = pd.to_numeric(
            bills_df.get("n_paragraphs", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0)
        metric_cols[2].metric(
            "Speaker paragraphs in candidates" if selected_name else "Paragraphs across candidates",
            f"{int(speaker_counts.sum() if selected_name else all_paragraph_counts.sum()):,}",
        )
    else:
        metric_cols[1].metric("Top bill similarity", "—")
        metric_cols[2].metric("Speaker filter", "No matching bills" if selected_name else "No results")

    with st.expander("Candidate bill list", expanded=True):
        show_bill_table(candidate_bills)

    run_reranking = False
    if candidate_bills:
        affair_ids = [str(bill.get("Affair_ID", "")) for bill in candidate_bills]
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Step 2</div>', unsafe_allow_html=True)
            st.subheader("Rerank paragraph evidence")
            st.write("Retrieve the strongest matching paragraphs from these bills.")
            with st.form("paragraph_search_form"):
                dense_pool = st.number_input(
                    "Paragraph reranking pool",
                    min_value=1,
                    max_value=1000,
                    value=100,
                    help="How many paragraph candidates to retrieve and rerank.",
                )
                run_reranking = st.form_submit_button(
                    "Retrieve and rerank paragraphs", type="primary"
                )
        if run_reranking:
            payload = {
                "retrieval_query": st.session_state["search_claim"],
                "stance_claim": st.session_state["search_claim"],
                "affair_ids": affair_ids,
                "paragraph_dense_pool": int(dense_pool),
            }
            speaker_name = st.session_state.get("search_speaker_name")
            if speaker_name:
                payload["speaker_name"] = speaker_name
            try:
                with st.spinner("Retrieving paragraphs and applying the reranker…"):
                    response = requests.post(
                        f"{service_url}/paragraphs", json=payload, timeout=3600
                    )
                    response.raise_for_status()
                st.session_state["paragraph_pool"] = response.json().get(
                    "candidate_passages_before_threshold", []
                )
                st.session_state["paragraph_search_error"] = None
            except requests.RequestException as exc:
                st.session_state["paragraph_search_error"] = str(exc)
        if st.session_state.get("paragraph_search_error"):
            st.error(f"Paragraph retrieval failed: {st.session_state['paragraph_search_error']}")

    paragraph_pool = st.session_state.get("paragraph_pool")
    if paragraph_pool is not None:
        st.markdown("## 3 · Explore reranked paragraphs")
        all_passages = filtered_passages(paragraph_pool, 0.0)
        with st.container(border=True):
            control_col, summary_col = st.columns([0.7, 0.3])
            with control_col:
                threshold = st.slider(
                    "Minimum rerank score",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.40,
                    step=0.01,
                    help="Move the cutoff to show more or fewer matching paragraphs.",
                )
            matches = filtered_passages(paragraph_pool, threshold)
            with summary_col:
                st.metric(
                    "Paragraphs above cutoff",
                    f"{len(matches):,}",
                    delta=f"of {len(all_passages):,} reranked",
                )
        if matches.empty:
            st.info("No paragraphs meet this score cutoff. Lower the slider to include more results.")
        else:
            show_paragraph_cards(matches)
