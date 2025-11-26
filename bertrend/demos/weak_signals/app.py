#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import torch

# workaround with streamlit to avoid errors Examining the path of torch.classes raised: Tried to instantiate class 'path.path’, but it does not exist! Ensure that it is registered via torch::class
torch.classes.__path__ = []

import pickle
import shutil
from typing import Literal

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from bertrend import ZEROSHOT_TOPICS_DATA_DIR, CACHE_PATH
from bertrend.BERTrend import BERTrend
from bertrend.demos.demos_utils import is_admin_mode
from bertrend.demos.demos_utils.data_loading_component import (
    display_data_loading_component,
)
from bertrend.demos.demos_utils.embed_documents_component import (
    display_embed_documents_component,
)
from bertrend.demos.demos_utils.icons import (
    WARNING_ICON,
    SUCCESS_ICON,
    SETTINGS_ICON,
    TOPIC_ICON,
    TREND_ICON,
    ERROR_ICON,
    ANALYSIS_ICON,
    MODEL_TRAINING_ICON,
    DATA_LOADING_ICON,
    EMBEDDING_ICON,
)
from bertrend.demos.demos_utils.messages import (
    NO_EMBEDDINGS_WARNING_MESSAGE,
    MODEL_TRAINING_COMPLETE_MESSAGE,
)
from bertrend.demos.demos_utils.parameters_component import (
    display_bertopic_hyperparameters,
    display_bertrend_hyperparameters,
    display_embedding_hyperparameters,
)
from bertrend.BERTopicModel import BERTopicModel
from bertrend.llm_utils.llama3_client import Llama3Client
from bertrend import LLM_CONFIG
from bertrend.demos.weak_signals.messages import (
    MODEL_MERGING_COMPLETE_MESSAGE,
    NO_CACHE_WARNING,
    CACHE_PURGED_MESSAGE,
    MODELS_RESTORED_MESSAGE,
    NO_DATA_WARNING,
    STATE_SAVED_MESSAGE,
    STATE_RESTORED_MESSAGE,
    MODELS_SAVED_MESSAGE,
    NO_MODELS_WARNING,
    NO_STATE_WARNING,
    EMBED_WARNING,
    EMBED_TRAIN_WARNING,
    TRAIN_WARNING,
    MERGE_WARNING,
)
from bertrend.trend_analysis.weak_signals import detect_weak_signals_zeroshot

from bertrend.utils.data_loading import (
    group_by_days,
    TEXT_COLUMN,
)
from bertrend.config.parameters import *
from bertrend.demos.demos_utils.state_utils import SessionStateManager
from bertrend.trend_analysis.visualizations import (
    plot_size_outliers,
    plot_num_topics,
)
from bertrend.demos.weak_signals.visualizations_utils import (
    PLOTLY_BUTTON_SAVE_CONFIG,
    display_sankey_diagram,
    display_topics_per_timestamp,
    display_newly_emerged_topics,
    display_popularity_evolution,
    save_signal_evolution,
    display_signal_analysis,
    retrieve_topic_counts,
    display_signal_types,
)

# UI Settings
PAGE_TITLE = "BERTrend - Retrospective Trend Analysis demo"
LAYOUT: Literal["centered", "wide"] = "wide"


# TODO: handle uploaded files
def save_state():
    """Save the application state"""
    state_file = CACHE_PATH / STATE_FILE
    embeddings_file = CACHE_PATH / EMBEDDINGS_FILE

    # Save the selected files (list of filenames)
    selected_files = SessionStateManager.get("selected_files", [])

    state = SessionStateManager.get_multiple(
        "selected_files",
        "min_chars",
        "split_by_paragraph",
        "timeframe_slider",
        "language",
        "embedding_model_name",
        "embedding_model",
        "sample_size",
        "min_similarity",
        "zeroshot_min_similarity",
        "embedding_dtype",
        "data_embedded",
    )

    state["selected_files"] = selected_files

    with open(state_file, "wb") as f:
        pickle.dump(state, f)

    np.save(embeddings_file, SessionStateManager.get_embeddings())
    st.success(STATE_SAVED_MESSAGE, icon=SUCCESS_ICON)


# TODO: handle uploaded files
def restore_state():
    """Restore the application state"""
    state_file = CACHE_PATH / STATE_FILE
    embeddings_file = CACHE_PATH / EMBEDDINGS_FILE

    if state_file.exists() and embeddings_file.exists():
        with open(state_file, "rb") as f:
            state = pickle.load(f)

        # Restore the selected files
        selected_files = state.get("selected_files", [])
        SessionStateManager.set("selected_files", selected_files)

        # Restore other states
        SessionStateManager.set_multiple(**state)
        SessionStateManager.set("embeddings", np.load(embeddings_file))
        st.success(STATE_RESTORED_MESSAGE, icon=SUCCESS_ICON)

        # Update the multiselect widget with restored selected files
        st.session_state["selected_files"] = selected_files
    else:
        st.warning(NO_STATE_WARNING, icon=WARNING_ICON)


def purge_cache():
    """Purge cache data"""
    if CACHE_PATH.exists():
        shutil.rmtree(CACHE_PATH)
        st.success(CACHE_PURGED_MESSAGE, icon=SUCCESS_ICON)
    else:
        st.warning(NO_CACHE_WARNING, icon=WARNING_ICON)


def load_data_page():
    st.header("Data Loading and Preprocessing")

    display_data_loading_component()

    if "time_filtered_df" in st.session_state:
        try:
            display_embed_documents_component()
            if SessionStateManager.get("data_embedded", False):
                save_state()
        except Exception as e:
            logger.error(f"An error occurred while embedding documents: {e}")
            st.error(
                f"An error occurred while embedding documents: {e}",
                icon=ERROR_ICON,
            )


def training_page():
    st.header("Model Training")

    if not SessionStateManager.get("data_embedded"):
        st.warning(NO_EMBEDDINGS_WARNING_MESSAGE, icon=WARNING_ICON)
        st.stop()

    # Show documents per grouped timestamp
    with st.expander("Documents per Timestamp", expanded=True):
        st.write(f"Granularity: {st.session_state['granularity']}")
        grouped_data = group_by_days(
            SessionStateManager.get_dataframe("time_filtered_df"),
            day_granularity=st.session_state["granularity"],
        )
        non_empty_timestamps = [
            timestamp for timestamp, group in grouped_data.items() if not group.empty
        ]
        if non_empty_timestamps:
            selected_timestamp = st.select_slider(
                "Select Timestamp",
                options=non_empty_timestamps,
                key="timestamp_slider",
            )
            selected_docs = grouped_data[selected_timestamp]
            st.dataframe(
                selected_docs[
                    ["timestamp", TEXT_COLUMN, "document_id", "source", "url"]
                ],
                use_container_width=True,
            )
        else:
            st.warning(NO_DATA_WARNING, icon=WARNING_ICON)

    if not SessionStateManager.get("data_embedded", False):
        st.warning(
            EMBED_WARNING,
            icon=WARNING_ICON,
        )
        st.stop()
    else:
        # Zero-shot topic definition
        zeroshot_topic_list = st.text_input(
            "Enter zero-shot topics (separated by /)", value=""
        )
        zeroshot_topic_list = [
            topic.strip() for topic in zeroshot_topic_list.split("/") if topic.strip()
        ]
        SessionStateManager.set("zeroshot_topic_list", zeroshot_topic_list)

        if st.button("Train Models", type="primary"):
            with st.spinner("Training models..."):
                # FIXME: called twice (see above)
                grouped_data = group_by_days(
                    SessionStateManager.get_dataframe("time_filtered_df"),
                    day_granularity=st.session_state["granularity"],
                )

                # Initialize topic model
                topic_model = BERTopicModel(st.session_state["bertopic_config"])

                # Created BERTrend object
                bertrend = BERTrend(
                    config_file=st.session_state["bertrend_config"],
                    topic_model=topic_model,
                )
                # Train topic models on data
                bertrend.train_topic_models(
                    grouped_data=grouped_data,
                    embedding_model=SessionStateManager.get("embedding_model"),
                    embeddings=SessionStateManager.get_embeddings(),
                )
                st.success(MODEL_TRAINING_COMPLETE_MESSAGE, icon=SUCCESS_ICON)

                # Save trained models
                bertrend.save_model()
                st.success(MODELS_SAVED_MESSAGE, icon=SUCCESS_ICON)

                # Compute signal popularity
                bertrend.calculate_signal_popularity()
                SessionStateManager.set("popularity_computed", True)

                # Store bertrend object
                SessionStateManager.set("bertrend", bertrend)

                st.success(MODEL_MERGING_COMPLETE_MESSAGE, icon=SUCCESS_ICON)


def analysis_page():
    st.header("Results Analysis")

    if not SessionStateManager.get("data_embedded"):
        st.warning(
            EMBED_TRAIN_WARNING,
            icon=WARNING_ICON,
        )
        st.stop()

    elif (
        not SessionStateManager.get("bertrend")
        or not SessionStateManager.get("bertrend")._is_fitted
    ):
        st.warning(
            TRAIN_WARNING,
            icon=WARNING_ICON,
        )
        st.stop()

    else:
        topic_models = SessionStateManager.get("bertrend").restore_topic_models()
        with st.expander("Topic Overview", expanded=False):
            # Number of Topics Detected for each topic model
            st.plotly_chart(
                plot_num_topics(topic_models),
                config=PLOTLY_BUTTON_SAVE_CONFIG,
                use_container_width=True,
            )
            # Size of Outlier Topic for each topic model
            st.plotly_chart(
                plot_size_outliers(topic_models),
                config=PLOTLY_BUTTON_SAVE_CONFIG,
                use_container_width=True,
            )

        display_topics_per_timestamp(topic_models)

        # Display zeroshot signal trend
        zeroshot_topic_list = SessionStateManager.get("zeroshot_topic_list", None)
        if zeroshot_topic_list:
            st.subheader("Zero-shot Weak Signal Trends")
            weak_signal_trends = detect_weak_signals_zeroshot(
                topic_models,
                zeroshot_topic_list,
                st.session_state["granularity"],
            )
            with st.expander("Zero-shot Weak Signal Trends", expanded=False):
                fig_trend = go.Figure()
                for topic, weak_signal_trend in weak_signal_trends.items():
                    timestamps = list(weak_signal_trend.keys())
                    popularity = [
                        weak_signal_trend[timestamp]["Document_Count"]
                        for timestamp in timestamps
                    ]
                    hovertext = [
                        f"Topic: {topic}<br>Timestamp: {timestamp}<br>Popularity: {weak_signal_trend[timestamp]['Document_Count']}<br>Representation: {weak_signal_trend[timestamp]['Representation']}"
                        for timestamp in timestamps
                    ]
                    fig_trend.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=popularity,
                            mode="lines+markers",
                            name=topic,
                            hovertext=hovertext,
                            hoverinfo="text",
                        )
                    )
                fig_trend.update_layout(
                    title="Popularity of Zero-Shot Topics",
                    xaxis_title="Timestamp",
                    yaxis_title="Popularity",
                )
                st.plotly_chart(
                    fig_trend,
                    config=PLOTLY_BUTTON_SAVE_CONFIG,
                    use_container_width=True,
                )

                # Display the dataframe with zeroshot topics information
                zeroshot_topics_data = [
                    {
                        "Topic": topic,
                        "Timestamp": timestamp,
                        "Representation": data["Representation"],
                        "Representative_Docs": data["Representative_Docs"],
                        "Count": data["Count"],
                        "Document_Count": data["Document_Count"],
                    }
                    for topic, weak_signal_trend in weak_signal_trends.items()
                    for timestamp, data in weak_signal_trend.items()
                ]
                zeroshot_topics_df = pd.DataFrame(zeroshot_topics_data)
                st.dataframe(zeroshot_topics_df, use_container_width=True)

                # Save the zeroshot topics data to a JSON file
                json_file_path = ZEROSHOT_TOPICS_DATA_DIR
                json_file_path.mkdir(parents=True, exist_ok=True)

                zeroshot_topics_df.to_json(
                    json_file_path / ZEROSHOT_TOPICS_DATA_FILE,
                    orient="records",
                    date_format="iso",
                    indent=4,
                )
                st.success(
                    f"Zeroshot topics data saved to {json_file_path}",
                    icon=SUCCESS_ICON,
                )

        if not SessionStateManager.get("popularity_computed", False):
            st.warning(
                MERGE_WARNING,
                icon=WARNING_ICON,
            )
            st.stop()

        else:
            # Display merged signal trend
            with st.expander("Topic Size Evolution", expanded=False):
                st.dataframe(
                    SessionStateManager.get("bertrend").all_merge_histories_df[
                        [
                            "Timestamp",
                            "Topic1",
                            "Topic2",
                            "Representation1",
                            "Representation2",
                            "Document_Count1",
                            "Document_Count2",
                        ]
                    ]
                )

            # Display topic popularity evolution
            with st.expander("Topic Popularity Evolution", expanded=True):
                display_popularity_evolution()
                # Save Signal Evolution Data to investigate later on in a separate notebook
                save_signal_evolution()

            # Show weak/strong signals
            display_signal_types()

            # Analyze signal
            with st.expander("Signal Analysis", expanded=True):
                st.subheader("Signal Analysis")
                topic_number = st.number_input(
                    "Enter a topic number to take a closer look:", min_value=0, step=1
                )
                if st.button("Analyze signal", type="primary"):
                    try:
                        display_signal_analysis(topic_number)
                    except Exception as e:
                        from bertrend.demos.weak_signals.messages import (
                            SIGNAL_ANALYSIS_ERROR,
                        )

                        logger.error(
                            f"Error in signal analysis for topic {topic_number}: {str(e)}"
                        )
                        st.error(
                            f"{SIGNAL_ANALYSIS_ERROR} Error: {str(e)}",
                            icon=ERROR_ICON,
                        )

            # Chat
            with st.expander("Chat", expanded=True):
                st.subheader("Chat")
                chat_topic_number = st.number_input(
                    "Enter a topic number to chat about:", min_value=0, step=1, key="chat_topic_number"
                )

                if f"chat_history_{chat_topic_number}" not in st.session_state:
                    st.session_state[f"chat_history_{chat_topic_number}"] = []

                for message in st.session_state[f"chat_history_{chat_topic_number}"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("What is up?"):
                    st.session_state[f"chat_history_{chat_topic_number}"].append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        full_response = ""
                        
                        # Get context for the topic
                        bertrend = SessionStateManager.get("bertrend")
                        topic_merge_rows = bertrend.all_merge_histories_df[
                            bertrend.all_merge_histories_df["Topic1"] == chat_topic_number
                        ].sort_values("Timestamp")
                        
                        content_summary = "\n".join(
                            [
                                f"Timestamp: {row.Timestamp.strftime('%Y-%m-%d')}\n"
                                f"Topic representation: {row.Representation1}\n"
                                f"{' '.join(f'- {doc}' for doc in row.Documents1 if isinstance(doc, str))}\n"
                                f"Timestamp: {(row.Timestamp + pd.Timedelta(days=bertrend.config['granularity'])).strftime('%Y-%m-%d')}\n"
                                f"Topic representation: {row.Representation2}\n"
                                f"{' '.join(f'- {doc}' for doc in row.Documents2 if isinstance(doc, str))}\n"
                                for row in topic_merge_rows.itertuples()
                            ]
                        )

                        # Truncate context to avoid errors
                        max_content_length = 8000
                        if len(content_summary) > max_content_length:
                            content_summary = content_summary[:max_content_length] + "\n\n[Context truncated due to length]"
                        
                        system_prompt = f"You are a helpful assistant. The user wants to chat about topic number {chat_topic_number}. Here is some context about the topic:\n{content_summary}"

                        messages=[
                                {"role": "system", "content": system_prompt}
                            ] + st.session_state[f"chat_history_{chat_topic_number}"]
                        
                        try:
                            llama3_client = Llama3Client(
                                api_key=LLM_CONFIG["api_key"],
                                endpoint=LLM_CONFIG["endpoint"],
                                model=LLM_CONFIG["model"],
                            )
                            
                            response = llama3_client.generate_from_history(
                                messages,
                                temperature=LLM_CONFIG["temperature"],
                                max_output_tokens=LLM_CONFIG["max_output_tokens"],
                            )
                            full_response = response
                            
                        except Exception as e:
                            full_response = f"Sorry, I encountered an error: {e}"
                        
                        message_placeholder.markdown(full_response)
                    st.session_state[f"chat_history_{chat_topic_number}"].append({"role": "assistant", "content": full_response})

            # Global Chat
            with st.expander("Global Chat (All Clusters)", expanded=False):
                st.subheader("Global Chat")

                def generate_global_context(weak_df, strong_df):
                    context = "Here is a summary of the latest trends across all topics:\n\n"
                    
                    if not strong_df.empty:
                        context += "**Strong Signals (Established Trends):**\n"
                        for _, row in strong_df.head(5).iterrows(): # Top 5
                            context += f"- Topic {row['Topic']}: {row['Representation']}. (Popularity: {row['Latest_Popularity']:.2f})\n"
                        context += "\n"

                    if not weak_df.empty:
                        context += "**Weak Signals (Emerging Trends):**\n"
                        for _, row in weak_df.head(5).iterrows(): # Top 5
                            context += f"- Topic {row['Topic']}: {row['Representation']}. (Popularity: {row['Latest_Popularity']:.2f})\n"
                        context += "\n"
                    
                    if strong_df.empty and weak_df.empty:
                        return "There are currently no significant trends detected."
                        
                    return context

                if "global_chat_history" not in st.session_state:
                    st.session_state.global_chat_history = []

                for message in st.session_state.global_chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Ask about emerging trends..."):
                    st.session_state.global_chat_history.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        
                        # Get context
                        bertrend = SessionStateManager.get("bertrend")
                        window_end = st.session_state.get("window_end")
                        window_start = st.session_state.get("window_start")
                        q1 = st.session_state.get("q1")
                        q3 = st.session_state.get("q3")
                        noise_topics_df, weak_signal_topics_df, strong_signal_topics_df = (
                            bertrend._classify_signals(window_start, window_end, q1, q3)
                        )
                        
                        global_context = generate_global_context(weak_signal_topics_df, strong_signal_topics_df)
                        
                        system_prompt = f"You are a helpful assistant that can answer questions about emerging trends based on the provided summary.\n\n{global_context}"

                        messages = [
                            {"role": "system", "content": system_prompt}
                        ] + st.session_state.global_chat_history
                        
                        try:
                            llama3_client = Llama3Client(
                                api_key=LLM_CONFIG["api_key"],
                                endpoint=LLM_CONFIG["endpoint"],
                                model=LLM_CONFIG["model"],
                            )
                            
                            response = llama3_client.generate_from_history(
                                messages,
                                temperature=LLM_CONFIG["temperature"],
                                max_output_tokens=LLM_CONFIG["max_output_tokens"],
                            )
                            full_response = response
                            
                        except Exception as e:
                            full_response = f"Sorry, I encountered an error: {e}"
                        
                        message_placeholder.markdown(full_response)
                    st.session_state.global_chat_history.append({"role": "assistant", "content": full_response})

            # Create the Sankey Diagram
            st.subheader("Topic Evolution")
            display_sankey_diagram(
                SessionStateManager.get("bertrend").all_merge_histories_df
            )

            # Newly emerged topics
            if SessionStateManager.get("bertrend").all_new_topics_df is not None:
                st.subheader("Newly Emerged Topics")
                display_newly_emerged_topics(
                    SessionStateManager.get("bertrend").all_new_topics_df
                )

            if st.button("Retrieve Topic Counts"):
                with st.spinner("Retrieving topic counts..."):
                    # Number of topics per individual topic model
                    retrieve_topic_counts(topic_models)


def main():
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout=LAYOUT,
        initial_sidebar_state="expanded" if is_admin_mode() else "collapsed",
        page_icon=":part_alternation_mark:",
    )

    st.title(":part_alternation_mark: " + PAGE_TITLE)

    # Set the main flags
    SessionStateManager.get_or_set("data_embedded", False)
    SessionStateManager.get_or_set("popularity_computed", False)

    # Sidebar
    with st.sidebar:
        st.header(SETTINGS_ICON + " Settings and Controls")

        # State Management
        st.subheader("State Management")

        if st.button("Restore Previous Run", use_container_width=True):
            restore_state()
            try:
                SessionStateManager.set("bertrend", BERTrend.restore_model())
                st.success(MODELS_RESTORED_MESSAGE, icon=SUCCESS_ICON)
            except Exception as e:
                st.warning(NO_MODELS_WARNING, icon=WARNING_ICON)

        if st.button("Purge Cache", use_container_width=True):
            purge_cache()

        if st.button("Clear session state", use_container_width=True):
            SessionStateManager.clear()

        # BERTopic Hyperparameters
        st.subheader(EMBEDDING_ICON + " Embedding Hyperparameters")
        display_embedding_hyperparameters()
        st.subheader(TOPIC_ICON + " BERTopic Hyperparameters")
        display_bertopic_hyperparameters()
        st.subheader(TREND_ICON + " BERTrend Hyperparameters")
        display_bertrend_hyperparameters()

    # Main content
    tab1, tab2, tab3 = st.tabs(
        [
            DATA_LOADING_ICON + " Data Loading",
            MODEL_TRAINING_ICON + " Model Training",
            ANALYSIS_ICON + " Results Analysis",
        ]
    )

    with tab1:
        load_data_page()

    with tab2:
        training_page()

    with tab3:
        analysis_page()


if __name__ == "__main__":
    main()
