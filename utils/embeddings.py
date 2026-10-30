import numpy as np
import streamlit as st

from utils.config import EMBEDDING_MODEL


# Cache the model in memory.
@st.cache_resource(
    show_spinner="Loading embedding model..."
)
def get_embedding_model(model_name=EMBEDDING_MODEL):

    # Heavy dependency is imported only when required.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name
    )

    return model


def split_into_passages(
    text,
    chunk_size=850,
    overlap=120
):

    text = text.strip()

    if not text:
        return []

    passages = []
    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        if end < len(text):

            boundary = text.rfind(
                " ",
                start,
                end
            )

            if boundary > start + chunk_size // 2:
                end = boundary

        passage = text[start:end].strip()

        if passage:
            passages.append(passage)

        if end >= len(text):
            break

        start = max(
            end - overlap,
            start + 1
        )

    return passages


def generate_embeddings(texts):

    if not texts:

        return np.empty((0, 0))

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32
    )

    return embeddings


def calculate_cosine_similarity(
    requirement_embeddings,
    resume_embeddings
):

    if (
        requirement_embeddings.size == 0
        or resume_embeddings.size == 0
    ):

        raise ValueError(
            "Cannot calculate similarity with empty embeddings."
        )

    return np.matmul(
        requirement_embeddings,
        resume_embeddings.T
    )