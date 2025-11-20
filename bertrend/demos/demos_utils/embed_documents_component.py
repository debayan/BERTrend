#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import streamlit as st

from bertrend.demos.demos_utils.icons import SUCCESS_ICON
from bertrend.demos.demos_utils.messages import EMBEDDINGS_CALCULATED_MESSAGE
from bertrend.demos.demos_utils.state_utils import SessionStateManager
from bertrend.services.embedding_service import EmbeddingService
from bertrend.utils.data_loading import TEXT_COLUMN


def display_embed_documents_component():
    """
    Common Streamlit UI component for embedding documents.
    """
    # Embed documents
    if st.button("Embed Documents"):
        with st.spinner("Embedding documents..."):
            embedding_dtype = SessionStateManager.get("embedding_dtype")
            embedding_model_name = SessionStateManager.get("embedding_model_name")
            # CHANGED: Default embedding service type fallback changed from "remote" to "local"
            # This affects backend behavior when determining which embedding service to use
            # if embedding_service_type is not set in session state.
            # This ensures consistency with the new default embedding service setting in the UI.
            if SessionStateManager.get("embedding_service_type", "local") == "local":
                embedding_service = EmbeddingService(
                    local=True,
                    model_name=embedding_model_name,
                    embedding_dtype=embedding_dtype,
                )
            else:
                embedding_service = EmbeddingService(
                    local=False,
                    url=SessionStateManager.get("embedding_service_url"),
                )

            texts = SessionStateManager.get_dataframe("time_filtered_df")[
                TEXT_COLUMN
            ].tolist()

            embeddings, token_strings, token_embeddings = embedding_service.embed(
                texts=texts,
            )

            SessionStateManager.set(
                "embedding_model", embedding_service.embedding_model
            )
            SessionStateManager.set("embeddings", embeddings)
            SessionStateManager.set("token_strings", token_strings)
            SessionStateManager.set("token_embeddings", token_embeddings)

            SessionStateManager.set("data_embedded", True)

            st.success(EMBEDDINGS_CALCULATED_MESSAGE, icon=SUCCESS_ICON)
