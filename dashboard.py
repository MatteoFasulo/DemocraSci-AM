"""Collaborator dashboard for staged parliamentary evidence search."""

from __future__ import annotations

import os
import re
import threading
import unicodedata
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from chromadb.api.models.Collection import Collection
from huggingface_hub import sync_bucket
from sentence_transformers import CrossEncoder

from const import N_TOP_BILLS, RERANK_THRESHOLD, RERANKER_MODEL
from scripts.chroma_store import (
    BILLS_COLLECTION,
    PARAGRAPHS_COLLECTION,
    open_chroma_collections,
)

DEFAULT_CLAIM = "Switzerland should pursue a free-trade agreement with the US."
DEFAULT_LANG = "en"
CHROMA_PATH = Path(os.environ.get("CHROMA_PATH", "hf-bucket"))
HF_BUCKET_SOURCE = os.environ.get(
    "HF_BUCKET_SOURCE",
    "hf://buckets/MatteoFasulo/DemocraSci-ChromaDB/DemocraSci-ChromaDB/",
)


@st.cache_resource(show_spinner="Downloading the Chroma index if needed and loading search…")
def load_dashboard_state() -> DashboardState:
    """Download the public index if needed, then open Chroma and the reranker."""
    if not (CHROMA_PATH / "chroma.sqlite3").is_file():
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        try:
            sync_bucket(HF_BUCKET_SOURCE, str(CHROMA_PATH), token=False)
        except Exception as exc:
            raise RuntimeError(
                "Could not download the Chroma index anonymously from the Hugging Face Bucket. "
                "Confirm the bucket is public and that HF_BUCKET_SOURCE points to the folder "
                f"containing chroma.sqlite3. Source: {HF_BUCKET_SOURCE}. Details: {exc}"
            ) from exc
    if not (CHROMA_PATH / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            f"The Hugging Face Bucket download completed, but chroma.sqlite3 was not found in {CHROMA_PATH}. "
            f"Set HF_BUCKET_SOURCE to the bucket folder containing chroma.sqlite3 (currently {HF_BUCKET_SOURCE})."
        )
    return DashboardState(CHROMA_PATH)


def load_dashboard_reranker(device: torch.device | str | None = None) -> CrossEncoder:
    """Load the cross-encoder used to rerank retrieved speech paragraphs."""
    return CrossEncoder(
        RERANKER_MODEL,
        device=device,
        activation_fn=torch.nn.Sigmoid(),
        prompts={"classification": "Classify whether the document matches the query policy claim"},
        default_prompt_name="classification",
    )


def query_bills(
    retrieval_query: str,
    bill_collection: Collection,
    top_k_bills: int,
    affair_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Query the bill collection and return bills ranked by cosine similarity."""
    if affair_ids is not None and not affair_ids:
        return pd.DataFrame()
    query_args = {
        "query_texts": [retrieval_query],
        "n_results": top_k_bills,
        "include": ["documents", "metadatas", "distances"],
    }
    if affair_ids is not None:
        query_args["where"] = {"Affair_ID": {"$in": affair_ids}}
    result = bill_collection.query(**query_args)
    ids = result.get("ids", [[]])[0]
    if not ids:
        return pd.DataFrame()
    rows = []
    for rank, (bill_id, document, metadata, distance) in enumerate(
        zip(ids, result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True),
        start=1,
    ):
        row = dict(metadata or {})
        row["Affair_ID"] = str(row.get("Affair_ID", bill_id))
        row["bill_search_text"] = document
        row["bill_embedding_score"] = (1.0 - float(distance)) * 100
        row["bill_embedding_rank"] = rank
        row["bill_rank"] = rank
        rows.append(row)
    return pd.DataFrame(rows)


def _reranker_documents(passages: pd.DataFrame, contextual_reranking: bool) -> list[str]:
    if not contextual_reranking:
        return passages["text"].astype(str).tolist()
    documents = []
    for row in passages.itertuples(index=False):
        values = row._asdict()
        language = str(values.get("language", "")).upper()
        title = str(values.get("bill_title_de" if language == "DE" else "bill_title_fr", "") or "").strip()
        paragraph = str(values.get("text", ""))
        documents.append(f"Bill title: {title}\n\nParagraph: {paragraph}" if title else paragraph)
    return documents


def query_and_rerank_paragraphs(
    retrieval_query: str,
    stance_claim: str,
    paragraph_collection: Collection,
    candidate_bills: pd.DataFrame,
    reranker: CrossEncoder,
    dense_pool_size: int,
    batch_size: int,
    contextual_reranking: bool = False,
    speaker_names: list[str] | None = None,
) -> pd.DataFrame:
    """Query paragraphs within candidate bills, rerank them, and return the full pool."""
    if candidate_bills.empty:
        return pd.DataFrame()
    affair_ids = candidate_bills["Affair_ID"].astype(str).tolist()
    where: dict = {"Affair_ID": {"$in": affair_ids}}
    selected_speakers = [name.strip() for name in (speaker_names or []) if name.strip()]
    if selected_speakers:
        where = {"$and": [where, {"speaker_full_name": {"$in": selected_speakers}}]}
    result = paragraph_collection.query(
        query_texts=[retrieval_query],
        n_results=dense_pool_size,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = result.get("ids", [[]])[0]
    if not ids:
        return pd.DataFrame()
    passages = []
    for index, (paragraph_id, document, metadata, distance) in enumerate(
        zip(ids, result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True)
    ):
        row = dict(metadata or {})
        row.update({"chroma_id": paragraph_id, "text": document, "corpus_id": index,
                    "dense_score": (1.0 - float(distance)) * 100})
        passages.append(row)
    passage_df = pd.DataFrame(passages)
    ranked = reranker.rank(
        query=stance_claim,
        documents=_reranker_documents(passage_df, contextual_reranking),
        top_k=len(passage_df),
        return_documents=False,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    pool = pd.DataFrame(ranked).merge(passage_df, on="corpus_id", how="left")
    if "score" in pool.columns:
        pool = pool.rename(columns={"score": "rerank_score"})
    elif "rerank_score" not in pool.columns:
        raise ValueError("Reranker output does not expose a score field")
    pool = pool.sort_values("rerank_score", ascending=False).reset_index(drop=True)
    pool["retrieval_query"] = retrieval_query
    pool["stance_claim"] = stance_claim
    pool["topic"] = ""
    return pool


class DashboardState:
    def __init__(self, db_path: Path) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bill_store, self.paragraph_store = open_chroma_collections(
            db_path=db_path,
            bills_name=BILLS_COLLECTION,
            paragraphs_name=PARAGRAPHS_COLLECTION,
            device=device,
        )
        self.reranker = load_dashboard_reranker(device)
        self.lock = threading.Lock()
        self.facets_lock = threading.Lock()
        self._speaker_names: list[str] | None = None
        self._speaker_affair_counts: dict[str, dict[str, int]] | None = None
        self._speaker_ids: dict[str, str] | None = None

    def _load_speaker_facets(self) -> None:
        if self._speaker_affair_counts is not None:
            return
        counts: dict[str, dict[str, int]] = {}
        speaker_ids: dict[str, str] = {}
        page_size = 10_000
        offset = 0
        while True:
            page = self.paragraph_store.get(
                include=["metadatas"], limit=page_size, offset=offset
            )
            metadatas = page.get("metadatas", []) or []
            for metadata in metadatas:
                values = metadata or {}
                name = str(values.get("speaker_full_name", "")).strip()
                affair_id = str(values.get("Affair_ID", "")).strip()
                raw_speaker_id = values.get("speaker_id")
                if name and raw_speaker_id not in (None, "") and name not in speaker_ids:
                    if isinstance(raw_speaker_id, float) and raw_speaker_id.is_integer():
                        speaker_ids[name] = str(int(raw_speaker_id))
                    else:
                        speaker_ids[name] = str(raw_speaker_id).strip()
                if name and affair_id:
                    speaker_counts = counts.setdefault(name, {})
                    speaker_counts[affair_id] = speaker_counts.get(affair_id, 0) + 1
            offset += len(metadatas)
            if len(metadatas) < page_size:
                break
        self._speaker_affair_counts = counts
        self._speaker_names = sorted(counts, key=str.casefold)
        self._speaker_ids = speaker_ids

    def speaker_names(self) -> list[str]:
        """Return cached distinct speaker names from paragraph metadata."""
        with self.facets_lock:
            self._load_speaker_facets()
            return self._speaker_names

    def speaker_ids(self) -> dict[str, str]:
        """Return the parliamentary member ID associated with each speaker."""
        with self.facets_lock:
            self._load_speaker_facets()
            return self._speaker_ids

    def speaker_bill_counts(self, speaker_names: list[str]) -> dict[str, int]:
        """Return each bill's maximum paragraph count among selected speakers."""
        with self.facets_lock:
            self._load_speaker_facets()
            affair_counts: dict[str, int] = {}
            for speaker_name in speaker_names:
                for affair_id, count in self._speaker_affair_counts.get(speaker_name, {}).items():
                    affair_counts[affair_id] = max(affair_counts.get(affair_id, 0), count)
            return affair_counts




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
            f"https://www.parlament.ch/{DEFAULT_LANG}/ratsbetrieb/suche-curia-vista/geschaeft?"
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
    try:
        state = load_dashboard_state()
        st.success(
            f"Index ready · {state.bill_store.count():,} bills · "
            f"{state.paragraph_store.count():,} paragraphs",
            icon="✅",
        )
    except Exception as exc:
        st.error(f"Local index is not ready: {exc}")
        st.caption(
            "The public Hugging Face Bucket is downloaded to the app's local filesystem on startup. "
            "Check that the bucket is public and the configured source folder contains chroma.sqlite3."
        )
        st.stop()

speaker_facets: dict = {"speakers": state.speaker_names(), "speaker_ids": state.speaker_ids()}
speakers = speaker_facets.get("speakers", [])
speaker_ids = speaker_facets.get("speaker_ids", {})

st.markdown('<div class="eyebrow">DemocraSci · Evidence explorer</div>', unsafe_allow_html=True)
st.title("Parliamentary evidence")
st.write("Find relevant bills, then inspect the strongest matching speech paragraphs.")

st.markdown("## 1 · Find candidate bills")
with st.container(border=True), st.form("bill_search_form"):
    claim = st.text_input(
        "Claim",
        value=DEFAULT_CLAIM,
        placeholder="Write a policy claim to search for…",
        help="This claim drives both bill retrieval and paragraph reranking.",
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
            speaker_names = [selected_speaker] if selected_speaker else []
            speaker_bill_counts = state.speaker_bill_counts(speaker_names) if speaker_names else None
            affair_ids = None
            if speaker_bill_counts is not None:
                affair_ids = [
                    affair_id
                    for affair_id, count in speaker_bill_counts.items()
                    if count >= 1
                ]
            with state.lock:
                candidate_bill_df = query_bills(
                    payload["retrieval_query"], state.bill_store, payload["top_k_bills"], affair_ids
                )
            if speaker_bill_counts is not None and not candidate_bill_df.empty:
                candidate_bill_df["speaker_paragraph_count"] = candidate_bill_df["Affair_ID"].map(
                    speaker_bill_counts
                )
        st.session_state["candidate_bills"] = candidate_bill_df.to_dict(orient="records")
        st.session_state["search_claim"] = claim.strip()
        st.session_state["search_speaker_name"] = selected_speaker
        st.session_state["paragraph_pool"] = None
        st.session_state["bill_search_error"] = None
    except Exception as exc:
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
                contextual_reranking = st.checkbox(
                    "Include bill title in reranker context",
                    value=False,
                    help="Adds each paragraph's bill title to the reranker input when available.",
                )
                reranker_batch_size = st.number_input(
                    "Reranker batch size",
                    min_value=1,
                    max_value=512,
                    value=128,
                    help="Larger batches may use more memory, especially on CPU.",
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
                candidate_bill_frame = pd.DataFrame({"Affair_ID": affair_ids})
                with st.spinner("Retrieving paragraphs and applying the reranker…"), state.lock:
                    before_threshold = query_and_rerank_paragraphs(
                        retrieval_query=payload["retrieval_query"],
                        stance_claim=payload["stance_claim"],
                        paragraph_collection=state.paragraph_store,
                        candidate_bills=candidate_bill_frame,
                        reranker=state.reranker,
                        dense_pool_size=payload["paragraph_dense_pool"],
                        batch_size=int(reranker_batch_size),
                        contextual_reranking=contextual_reranking,
                        speaker_names=[speaker_name] if speaker_name else [],
                    )
                st.session_state["paragraph_pool"] = before_threshold.to_dict(orient="records")
                st.session_state["paragraph_search_error"] = None
            except Exception as exc:
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
                    value=RERANK_THRESHOLD,
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
        st.plotly_chart(
            go.Figure(
                go.Histogram(
                    x=all_passages["rerank_score"],
                    nbinsx=20,
                )
            ).update_layout(
                title="Rerank score distribution",
                xaxis_title="Rerank score",
                yaxis_title="Paragraph count",
                margin=dict(l=0, r=0, t=30, b=0),
            ),
            width="content",
        )
        if matches.empty:
            st.info("No paragraphs meet this score cutoff. Lower the slider to include more results.")
        else:
            st.download_button(
                "Download paragraphs as JSON",
                data=matches.to_json(orient="records", force_ascii=False, indent=2),
                file_name="parliamentary_paragraphs.json",
                mime="application/json",
                help="Exports the paragraphs currently shown above the score cutoff.",
            )
            show_paragraph_cards(matches)
