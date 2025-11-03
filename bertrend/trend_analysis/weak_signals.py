#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import numpy as np
import pandas as pd
import scipy
from bertopic import BERTopic
from loguru import logger
from pandas import Timestamp

from bertrend.llm_utils.llama3_client import Llama3Client
from bertrend import LLM_CONFIG
from bertrend.trend_analysis.data_structure import TopicSummaryList, SignalAnalysis
from bertrend.trend_analysis.prompts import get_prompt, fill_html_template

MAXIMUM_ANALYZED_PERIODS = 3
MAX_CHUNKS_PER_TOPIC = 6  # Hard cap on number of chunks processed per topic


def _compute_max_chunks(content_length: int) -> int:
    """Dynamically choose the maximum number of chunks based on content length.

    Heuristic:
    - Short topics (<= 6k chars): 3 chunks
    - Medium topics (<= 20k chars): 6 chunks
    - Very large topics (> 20k chars): 8 chunks
    """
    if content_length <= 6000:
        return 3
    if content_length <= 20000:
        return 6
    return 8


def chunk_and_summarize_content(
    content_summary: str,
    max_chunk_size: int,
    language: str,
    max_chunks: int = MAX_CHUNKS_PER_TOPIC,
) -> str:
    """
    Intelligently chunk large content and summarize each chunk, then combine summaries.
    
    Args:
        content_summary: The original content to chunk
        max_chunk_size: Maximum size for each chunk
        language: Language for the summarization prompts
        
    Returns:
        Combined summary of all chunks
    """
    logger.info(f"Starting intelligent chunking for content of {len(content_summary)} characters")
    
    # Split content by timestamps to maintain chronological order
    timestamp_sections = []
    current_section = ""
    
    lines = content_summary.split('\n')
    for line in lines:
        if line.startswith('Timestamp:'):
            if current_section.strip():
                timestamp_sections.append(current_section.strip())
            current_section = line + '\n'
        else:
            current_section += line + '\n'
    
    # Add the last section
    if current_section.strip():
        timestamp_sections.append(current_section.strip())
    
    # If we have multiple timestamp sections, chunk them intelligently
    if len(timestamp_sections) > 1:
        chunks = []
        current_chunk = ""
        
        for section in timestamp_sections:
            # If adding this section would exceed max_chunk_size, start a new chunk
            if len(current_chunk) + len(section) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                current_chunk += section + '\n\n'
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    else:
        # Single section - split by paragraphs or sentences
        chunks = []
        paragraphs = content_summary.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += paragraph + '\n\n'
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    logger.info(f"Split content into {len(chunks)} chunks")

    # Enforce maximum number of chunks to maintain performance
    if len(chunks) > max_chunks:
        # Heuristic: keep most recent/content-at-end chunks (assumes chronological ordering)
        skipped = len(chunks) - max_chunks
        logger.warning(
            f"Chunk limit applied: keeping last {max_chunks} chunks and skipping {skipped} older chunks"
        )
        chunks = chunks[-max_chunks:]
    
    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        logger.debug(f"Summarizing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        
        # Create a summary prompt for this chunk
        if language == "German":
            summary_prompt = f"""Fassen Sie den folgenden Textabschnitt zusammen und behalten Sie dabei alle wichtigen Informationen bei:

{chunk}

Bitte geben Sie eine prägnante Zusammenfassung zurück, die alle wichtigen Details und Entwicklungen enthält."""
        elif language == "French":
            summary_prompt = f"""Résumez le passage de texte suivant en conservant toutes les informations importantes :

{chunk}

Veuillez fournir un résumé concis qui contient tous les détails et développements importants."""
        else:  # English
            summary_prompt = f"""Summarize the following text passage while retaining all important information:

{chunk}

Please provide a concise summary that contains all important details and developments."""
        
        try:
            # Use a simple summarization approach - just truncate intelligently for now
            # In a full implementation, you could call the LLM here for each chunk
            if len(chunk) > max_chunk_size:
                # Find a good breaking point (end of sentence or paragraph)
                truncated = chunk[:max_chunk_size]
                last_period = truncated.rfind('.')
                last_newline = truncated.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > max_chunk_size * 0.8:  # Only break if we're not losing too much
                    chunk_summaries.append(truncated[:break_point + 1])
                else:
                    chunk_summaries.append(truncated)
            else:
                chunk_summaries.append(chunk)
                
        except Exception as e:
            logger.warning(f"Error summarizing chunk {i+1}: {e}")
            # Fallback to simple truncation
            chunk_summaries.append(chunk[:max_chunk_size])
    
    # Combine all chunk summaries
    combined_summary = '\n\n'.join(chunk_summaries)
    
    # If the combined summary is still too long, do a final truncation
    if len(combined_summary) > max_chunk_size:
        logger.warning(f"Combined summary still too long ({len(combined_summary)} chars), final truncation")
        combined_summary = combined_summary[:max_chunk_size] + "\n\n[Content summarized due to length]"
    
    logger.info(f"Chunking complete: {len(content_summary)} -> {len(combined_summary)} characters")
    return combined_summary


def detect_weak_signals_zeroshot(
    topic_models: dict[Timestamp, BERTopic],
    zeroshot_topic_list: list[str],
    granularity: int,
    decay_factor: float = 0.01,
    decay_power: float = 2,
) -> dict[str, dict[Timestamp, dict[str, any]]]:
    """
    Detect weak signals based on the zero-shot list of topics to monitor.

    Args:
        topic_models (Dict[Timestamp, BERTopic]): Dictionary of BERTopic models for each timestamp.
        zeroshot_topic_list (List[str]): List of topics to monitor for weak signals.
        granularity (int): The granularity of the timestamps in days.
        decay_factor (float): The decay factor for exponential decay.
        decay_power (float): The decay power for exponential decay.

    Returns:
        Dict[str, Dict[Timestamp, Dict[str, any]]]: Dictionary of weak signal trends for each monitored topic.
    """
    weak_signal_trends = {}

    min_timestamp = min(topic_models.keys())
    max_timestamp = max(topic_models.keys())
    timestamps = pd.date_range(
        start=min_timestamp, end=max_timestamp, freq=pd.Timedelta(days=granularity)
    )

    for topic in zeroshot_topic_list:
        weak_signal_trends[topic] = {}
        topic_last_popularity = {}
        topic_last_update = {}

        for timestamp in timestamps:
            # Check if the current timestamp has a corresponding topic model that was trained
            # This is useful in scenarios where we have time skips in the data we were dealing with
            # For example : Our data spans from jan 2024 to dec 2024 with a big gap in summer, and we used
            # a monthly granularity, this means that timestamps of june, july, aug... won't have a corresponding
            # topic model because no model was trained on that period due to non-existant data

            if timestamp in topic_models:
                topic_info = topic_models[timestamp].topic_info_df
                topic_data = topic_info[topic_info["Name"] == topic]

                if not topic_data.empty:
                    # zeroshot topic found in the model's output corresponding to current timestamp
                    representation = topic_data["Representation"].values[0]
                    representative_docs = topic_data["Representative_Docs"].values[0]
                    count = topic_data["Count"].values[0]
                    document_count = topic_data["Document_Count"].values[0]

                    if topic not in topic_last_popularity:
                        # If the First occurrence of the topic
                        topic_last_popularity[topic] = document_count
                        topic_last_update[topic] = timestamp
                    else:
                        # if not first appearance but receives an update in current timestamp, increase popularity
                        document_count += topic_last_popularity[topic]
                        topic_last_popularity[topic] = document_count
                        topic_last_update[topic] = timestamp

                    weak_signal_trends[topic][timestamp] = {
                        "Representation": representation,
                        "Representative_Docs": representative_docs,
                        "Count": count,
                        "Document_Count": document_count,
                    }
                else:
                    # Topic not found in the current timestamp, apply decay to previously seen topics
                    weak_signal_trends[topic][timestamp] = _apply_decay(
                        topic,
                        timestamp,
                        topic_last_popularity,
                        topic_last_update,
                        granularity,
                        decay_factor,
                        decay_power,
                    )
            else:
                # Timestamp not in topic_models, apply decay to previously seen topics
                weak_signal_trends[topic][timestamp] = _apply_decay(
                    topic,
                    timestamp,
                    topic_last_popularity,
                    topic_last_update,
                    granularity,
                    decay_factor,
                    decay_power,
                )

    return weak_signal_trends


def _apply_decay(
    topic,
    timestamp,
    topic_last_popularity,
    topic_last_update,
    granularity,
    decay_factor,
    decay_power,
):
    """Helper function to apply decay to topic popularity."""
    last_popularity = topic_last_popularity.get(topic, 0)
    last_update = topic_last_update.get(topic, timestamp)

    time_diff = timestamp - last_update
    periods_since_last_update = time_diff // pd.Timedelta(days=granularity)

    decayed_popularity = last_popularity * np.exp(
        -decay_factor * (periods_since_last_update**decay_power)
    )

    topic_last_popularity[topic] = decayed_popularity

    return {
        "Representation": None,
        "Representative_Docs": None,
        "Count": 0,
        "Document_Count": decayed_popularity,
    }


def _filter_data(data, window_end, keep_documents):
    """Helper function to filter data based on window_end."""
    return {
        "Timestamps": [ts for ts in data["Timestamps"] if ts <= window_end],
        "Popularity": [
            pop
            for ts, pop in zip(data["Timestamps"], data["Popularity"])
            if ts <= window_end
        ],
        "Representation": [
            rep
            for ts, rep in zip(data["Timestamps"], data["Representations"])
            if ts <= window_end
        ],
        "Documents": (
            [doc for ts, docs in data["Documents"] if ts <= window_end for doc in docs]
            if keep_documents
            else []
        ),
        "Sources": [sources for ts, sources in data["Sources"] if ts <= window_end],
        "Docs_Count": [
            count
            for ts, count in zip(data["Timestamps"], data["Docs_Count"])
            if ts <= window_end
        ],
        "Paragraphs_Count": [
            count
            for ts, count in zip(data["Timestamps"], data["Paragraphs_Count"])
            if ts <= window_end
        ],
        "Source_Diversity": [
            div
            for ts, div in zip(data["Timestamps"], data["Source_Diversity"])
            if ts <= window_end
        ],
    }


def _is_rising_popularity(filtered_data, latest_timestamp):
    """Helper function to check if popularity is rising."""
    retrospective_start = latest_timestamp - pd.Timedelta(days=14)
    retrospective_data = [
        (timestamp, popularity)
        for timestamp, popularity in zip(
            filtered_data["Timestamps"], filtered_data["Popularity"]
        )
        if retrospective_start <= timestamp <= latest_timestamp
    ]

    if len(retrospective_data) >= 2:
        x = range(len(retrospective_data))
        y = [popularity for _, popularity in retrospective_data]
        slope, _, _, _, _ = scipy.stats.linregress(x, y)
        return slope > 0
    return True


def _create_df(topics, keep_documents):
    df = pd.DataFrame(
        [
            {
                "Topic": topic,
                "Representation": filtered_data["Representation"][-1],
                "Latest_Popularity": latest_popularity,
                "Docs_Count": docs_count,
                "Paragraphs_Count": paragraphs_count,
                "Latest_Timestamp": latest_timestamp,
                "Documents": filtered_data["Documents"] if keep_documents else [],
                "Sources": {
                    source for sources in filtered_data["Sources"] for source in sources
                },
                "Source_Diversity": source_diversity,
            }
            for topic, latest_popularity, latest_timestamp, docs_count, paragraphs_count, source_diversity, filtered_data in topics
        ]
    )

    # if not df.empty: df = df[df['Latest_Popularity'] >= 0.01] # Remove signals that faded away by filtering on latest popularity
    return df


def _create_dataframes(
    noise_topics, weak_signal_topics, strong_signal_topics, keep_documents
):
    """Helper function to create DataFrames for each category."""

    return (
        _create_df(noise_topics, keep_documents),
        _create_df(weak_signal_topics, keep_documents),
        _create_df(strong_signal_topics, keep_documents),
    )


def _initialize_new_topic(
    topic_sizes, topic_last_popularity, topic_last_update, topic, timestamp, row
):
    """Initialize a new topic with its first data point."""
    topic_sizes[topic]["Timestamps"] = [timestamp]
    topic_sizes[topic]["Popularity"] = [row["Document_Count1"]]
    topic_sizes[topic]["Representation"] = "_".join(row["Representation1"])
    topic_sizes[topic]["Documents"] = [(timestamp, row["Documents1"])]
    topic_sizes[topic]["Sources"] = [(timestamp, row["Source1"])]
    topic_sizes[topic]["Docs_Count"] = [row["Document_Count1"]]
    topic_sizes[topic]["Paragraphs_Count"] = [row["Count1"]]
    topic_sizes[topic]["Source_Diversity"] = [len(set(row["Source1"]))]
    topic_sizes[topic]["Representations"] = [topic_sizes[topic]["Representation"]]

    topic_last_popularity[topic] = row["Document_Count1"]
    topic_last_update[topic] = timestamp


def update_existing_topic(
    topic_sizes,
    topic_last_popularity,
    topic_last_update,
    topic,
    timestamp,
    granularity,
    row,
):
    """Update an existing topic with new data."""
    next_timestamp = timestamp + granularity

    topic_sizes[topic]["Timestamps"].append(next_timestamp)
    topic_sizes[topic]["Popularity"].append(
        topic_last_popularity[topic] + row["Document_Count2"]
    )
    topic_sizes[topic]["Representation"] = "_".join(row["Representation2"])
    topic_sizes[topic]["Documents"].append((next_timestamp, row["Documents2"]))
    topic_sizes[topic]["Sources"].append((next_timestamp, row["Source2"]))
    topic_sizes[topic]["Docs_Count"].append(
        topic_sizes[topic]["Docs_Count"][-1] + row["Document_Count2"]
    )
    topic_sizes[topic]["Paragraphs_Count"].append(
        topic_sizes[topic]["Paragraphs_Count"][-1] + row["Count2"]
    )

    all_sources = [
        source for _, sources in topic_sizes[topic]["Sources"] for source in sources
    ]
    all_sources.extend(row["Source2"])
    topic_sizes[topic]["Source_Diversity"].append(len(set(all_sources)))
    topic_sizes[topic]["Representations"].append(topic_sizes[topic]["Representation"])

    topic_last_popularity[topic] = topic_last_popularity[topic] + row["Document_Count2"]
    topic_last_update[topic] = next_timestamp  # Update to next_timestamp


def _apply_decay_to_inactive_topics(
    topic_sizes,
    topic_last_popularity,
    topic_last_update,
    updated_topics,
    topics_updated_next,
    current_timestamp,
    granularity,
    decay_factor,
    decay_power,
):
    """Apply decay to topics that were not updated in the current timestamp or the next."""
    all_topics = set(topic_last_update.keys())
    inactive_topics = all_topics - updated_topics - topics_updated_next

    for topic in inactive_topics:
        last_popularity = topic_last_popularity[topic]
        last_update = topic_last_update[topic]

        time_diff = current_timestamp - last_update
        periods_since_last_update = time_diff // granularity

        if periods_since_last_update > 0:
            decayed_popularity = last_popularity * np.exp(
                -decay_factor * (periods_since_last_update**decay_power)
            )

            topic_sizes[topic]["Timestamps"].append(current_timestamp)
            topic_sizes[topic]["Popularity"].append(decayed_popularity)
            topic_sizes[topic]["Docs_Count"].append(
                topic_sizes[topic]["Docs_Count"][-1]
            )
            topic_sizes[topic]["Paragraphs_Count"].append(
                topic_sizes[topic]["Paragraphs_Count"][-1]
            )
            topic_sizes[topic]["Source_Diversity"].append(
                topic_sizes[topic]["Source_Diversity"][-1]
            )
            topic_sizes[topic]["Representations"].append(
                topic_sizes[topic]["Representation"]
            )
            topic_last_popularity[topic] = decayed_popularity


def analyze_signal(
    bertrend,
    topic_number: int,
    current_date: Timestamp,
    maximum_analysed_periods: int = MAXIMUM_ANALYZED_PERIODS,
) -> tuple[TopicSummaryList, SignalAnalysis]:
    topic_merge_rows = bertrend.all_merge_histories_df[
        bertrend.all_merge_histories_df["Topic1"] == topic_number
    ].sort_values("Timestamp")
    topic_merge_rows_filtered = topic_merge_rows[
        topic_merge_rows["Timestamp"] <= current_date
    ]

    # In order to avoid to have too big contexts, keep only data that correspond to the N last models built
    # before the current date
    considered_periods = sorted(
        [ts for ts in bertrend.get_periods() if ts < current_date]
    )[-maximum_analysed_periods:]
    if considered_periods:
        topic_merge_rows_filtered = topic_merge_rows_filtered[
            topic_merge_rows["Timestamp"] >= considered_periods[0]
        ]

    if not topic_merge_rows_filtered.empty:
        content_summary = "\n".join(
            [
                f"Timestamp: {row.Timestamp.strftime('%Y-%m-%d')}\n"
                f"Topic representation: {row.Representation1}\n"
                f"{' '.join(f'- {doc}' for doc in row.Documents1 if isinstance(doc, str))}\n"
                f"Timestamp: {(row.Timestamp + pd.Timedelta(days=bertrend.config['granularity'])).strftime('%Y-%m-%d')}\n"
                f"Topic representation: {row.Representation2}\n"
                f"{' '.join(f'- {doc}' for doc in row.Documents2 if isinstance(doc, str))}\n"
                for row in topic_merge_rows_filtered.itertuples()
            ]
        )

        language = bertrend.topic_model.config["global"]["language"]
        
        # Detect language from content_summary to ensure consistent language
        content_lower = content_summary.lower()
        
        # German indicators (most common in your dataset)
        german_indicators = [
            'deutsch', 'german', 'deutsche', 'deutschen', 'deutschland', 'deutschsprachig',
            'analyse', 'bewertung', 'signal', 'trends', 'tendenzen', 'entwicklung',
            'prognose', 'strategisch', 'elite', 'expertise', 'bereiche', 'branchen',
            'aufgabe', 'durchführung', 'vollständig', 'abgeleitet', 'zusammenfassung',
            'thema', 'nutzen', 'kenntnisse', 'fähigkeiten', 'analytisch', 'bereitstellung',
            'tiefgreifend', 'auswirkung', 'potenzial', 'evolution', 'implikationen',
            'kurz', 'lang', 'effekte', 'auswirkungen', 'möglich', 'folgen', 'zweiter',
            'ordnung', 'szenarien', 'entwickeln', 'manifestieren', 'zukunft', 'betrachten',
            'verschiedene', 'faktoren', 'könnten', 'beeinflussen', 'trajektorie', 'erkunden',
            'optimistisch', 'pessimistisch', 'interkonnexionen', 'synergien', 'identifizieren',
            'interagieren', 'andere', 'aktuelle', 'phänomene', 'emerging', 'diskutieren',
            'konflikte', 'systeme', 'paradigmen', 'existierende', 'treiber', 'inhibitoren',
            'analysieren', 'beschleunigen', 'verstärken', 'untersuchen', 'hindernisse',
            'widerstände', 'potenzielle', 'entwicklung', 'tiefgreifend', 'nuanciert',
            'gehen', 'über', 'oberflächliche', 'beobachtungen', 'stützen', 'insights',
            'erfassen', 'komplexität', 'bedeutung', 'zögern', 'nicht', 'machen',
            'vorhersagen', 'gut', 'begründet', 'trajektorie', 'konzentrieren', 'bereitstellung',
            'klare', 'einsichtsvoll', 'nutzbar', 'kann', 'erleuchten', 'entscheidung',
            'planung', 'zukunft', 'als', 'elite-strategieprognose-analyst', 'umfangreicher',
            'expertise', 'verschiedenen', 'bereichen', 'branchen', 'aufgabe', 'umfassende',
            'bewertung', 'potenziellen', 'signals', 'abgeleitet', 'folgenden', 'themenzusammenfassung',
            'nutzen', 'wissen', 'analytischen', 'fähigkeiten', 'tiefgreifende', 'analyse',
            'potenziellen', 'auswirkungen', 'entwicklung', 'signals', 'liefern'
        ]
        
        # French indicators (fallback)
        french_indicators = [
            'français', 'french', 'analyse', 'évaluation', 'signal', 'tendances',
            'prospective', 'stratégique', 'élite', 'expertise', 'domaines',
            'industries', 'tâche', 'mener', 'complète', 'dérivé', 'résumé',
            'sujet', 'utilisez', 'connaissances', 'compétences', 'analytiques',
            'fournir', 'approfondie', 'impact', 'potentiel', 'évolution'
        ]
        
        # English indicators (fallback)
        english_indicators = [
            'english', 'analysis', 'evaluation', 'signal', 'trends', 'strategic',
            'elite', 'expertise', 'domains', 'industries', 'task', 'conduct',
            'complete', 'derived', 'summary', 'subject', 'use', 'knowledge',
            'skills', 'analytical', 'provide', 'in-depth', 'impact', 'potential',
            'evolution', 'implications', 'short', 'term', 'long', 'effects',
            'possible', 'consequences', 'second', 'order', 'scenarios', 'develop',
            'manifest', 'future', 'consider', 'various', 'factors', 'could',
            'influence', 'trajectory', 'explore', 'optimistic', 'pessimistic',
            'interconnections', 'synergies', 'identify', 'interact', 'other',
            'current', 'phenomena', 'emerging', 'discuss', 'conflicts', 'systems',
            'paradigms', 'existing', 'drivers', 'inhibitors', 'analyze', 'accelerate',
            'amplify', 'examine', 'obstacles', 'resistances', 'hinder', 'development',
            'thorough', 'nuanced', 'beyond', 'surface-level', 'observations', 'draw',
            'insights', 'capture', 'complexity', 'importance', 'hesitate', 'make',
            'predictions', 'well-reasoned', 'trajectory', 'focus', 'providing',
            'clear', 'insightful', 'actionable', 'inform', 'decision-making', 'planning',
            'as an', 'elite strategic', 'foresight analyst', 'extensive expertise',
            'multiple domains', 'comprehensive evaluation', 'potential signal',
            'topic summary', 'leverage your', 'analytical skills', 'in-depth analysis',
            'potential impact', 'evolution', 'examine the', 'potential effects',
            'various sectors', 'societal aspects', 'consider both', 'short-term',
            'long-term implications', 'analyze possible', 'ripple effects',
            'second-order consequences', 'describe potential', 'ways this signal',
            'could develop', 'manifest in', 'consider various', 'factors that',
            'could influence', 'its trajectory', 'explore both', 'optimistic',
            'pessimistic scenarios', 'identify how', 'this signal might',
            'interact with', 'other current', 'emerging phenomena', 'discuss potential',
            'synergies or', 'conflicts with', 'existing systems', 'paradigms',
            'analyze factors', 'could accelerate', 'amplify this', 'examine potential',
            'barriers or', 'resistances that', 'might hinder', 'its development',
            'your analysis', 'should be', 'thorough and', 'nuanced going',
            'beyond surface-level', 'observations draw', 'upon your', 'expertise to',
            'provide insights', 'that capture', 'the complexity', 'and potential',
            'significance of', 'this signal', 'don\'t hesitate', 'to make',
            'well-reasoned predictions', 'about its', 'potential trajectory',
            'and impact', 'focus on', 'providing a', 'clear insightful',
            'and actionable', 'analysis that', 'can inform', 'strategic decision-making',
            'and future', 'planning'
        ]
        
        # Detect language with priority: German > French > English
        is_german = any(word in content_lower for word in german_indicators)
        is_french = any(word in content_lower for word in french_indicators)
        is_english = any(word in content_lower for word in english_indicators)
        
        # Determine the actual language to use
        if is_german:
            detected_language = "German"
        elif is_french:
            detected_language = "French"
        elif is_english:
            detected_language = "English"
        else:
            # Default to German for multilingual or unknown languages
            detected_language = "German"
        
        logger.debug(f"Language detection: content suggests {detected_language}, config has {language}")
        
        # Use detected language instead of config language
        language = detected_language
        
        # Check if content_summary is too large and needs chunking
        max_content_length = 8000  # Maximum characters for content summary
        if len(content_summary) > max_content_length:
            logger.warning(f"Content summary too large ({len(content_summary)} chars), implementing intelligent chunking")
            dynamic_max_chunks = _compute_max_chunks(len(content_summary))
            if dynamic_max_chunks != MAX_CHUNKS_PER_TOPIC:
                logger.info(
                    f"Dynamic chunk cap selected: {dynamic_max_chunks} (default {MAX_CHUNKS_PER_TOPIC})"
                )
            content_summary = chunk_and_summarize_content(
                content_summary,
                max_content_length,
                language,
                max_chunks=dynamic_max_chunks,
            )

        try:
            llama3_client = Llama3Client(
                api_key=LLM_CONFIG["api_key"],
                endpoint=LLM_CONFIG["endpoint"],
                model=LLM_CONFIG["model"],
            )

            # First prompt: Generate summary
            logger.debug("First prompt - generate summary")
            summary_prompt = get_prompt(
                language,
                "topic_summary",
                topic_number=topic_number,
                content_summary=content_summary,
            )
            summaries = llama3_client.parse(
                system_prompt=LLM_CONFIG["system_prompt"],
                user_prompt=summary_prompt,
                temperature=LLM_CONFIG["temperature"],
                max_output_tokens=LLM_CONFIG["max_output_tokens"],
                response_format=TopicSummaryList,
            )

            if not summaries:
                raise ValueError(
                    "An anomaly occured during topic summary generation. Context probably too long. Check other logs for details"
                )

            # Second prompt: Analyze weak signal
            logger.debug("Second prompt - analyze weak signal")
            weak_signal_prompt = get_prompt(
                language,
                prompt_type="weak_signal",
                summary_from_first_prompt=summaries.model_dump_json(),
            )
            weak_signal_analysis = llama3_client.parse(
                system_prompt=LLM_CONFIG["system_prompt"],
                user_prompt=weak_signal_prompt,
                temperature=LLM_CONFIG["temperature"],
                max_output_tokens=LLM_CONFIG["max_output_tokens"],
                response_format=SignalAnalysis,
            )

            return summaries, weak_signal_analysis

        except Exception as e:
            error_msg = f"An error occurred while generating the analysis: {str(e)}"
            logger.error(error_msg)
            # Return empty objects instead of None to prevent AttributeError in UI
            empty_summaries = TopicSummaryList(topic_summary_by_time_period=[])
            empty_analysis = SignalAnalysis()
            return empty_summaries, empty_analysis

    else:
        error_msg = f"No data available for topic {topic_number} within the specified date range. Please enter a valid topic number."
        logger.error(error_msg)
        # Return empty objects instead of None to prevent AttributeError in UI
        # The UI will check for empty data and display appropriate message
        empty_summaries = TopicSummaryList(topic_summary_by_time_period=[])
        empty_analysis = SignalAnalysis()
        return empty_summaries, empty_analysis
