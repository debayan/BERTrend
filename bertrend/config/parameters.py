#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import json
from pathlib import Path

import torch

from bertrend import BERTREND_CONFIG, EMBEDDING_CONFIG

stopwords_en_file = Path(__file__).parent.parent / "resources" / "stopwords-en.json"
stopwords_fr_file = Path(__file__).parent.parent / "resources" / "stopwords-fr.json"
stopwords_rte_file = Path(__file__).parent.parent / "resources" / "stopwords-rte.json"
common_ngrams_file = Path(__file__).parent.parent / "resources" / "common_ngrams.json"
with open(stopwords_en_file, "r", encoding="utf-8") as file:
    ENGLISH_STOPWORDS = json.load(file)
with open(stopwords_fr_file, "r", encoding="utf-8") as file:
    FRENCH_STOPWORDS = json.load(file)
with open(stopwords_rte_file, "r", encoding="utf-8") as file:
    STOP_WORDS_RTE = json.load(file)
with open(common_ngrams_file, "r", encoding="utf-8") as file:
    COMMON_NGRAMS = json.load(file)

STOPWORDS = STOP_WORDS_RTE + COMMON_NGRAMS + FRENCH_STOPWORDS

# File names
STATE_FILE = "app_state.pkl"
EMBEDDINGS_FILE = "embeddings.npy"
BERTREND_FILE = "bertrend.dill"
DOC_INFO_DF_FILE = "doc_info_df.pkl"
TOPIC_INFO_DF_FILE = "topic_info_df.pkl"

# Model file names
ZEROSHOT_TOPICS_DATA_FILE = "zeroshot_topics_data.json"
INDIVIDUAL_MODEL_TOPIC_COUNTS_FILE = "individual_topic_counts.json"
CUMULATIVE_MERGED_TOPIC_COUNTS_FILE = "cumulative_topic_counts.json"

# Embedding models
ENGLISH_EMBEDDING_MODELS = [
    "all-mpnet-base-v2",
    "Alibaba-NLP/gte-base-en-v1.5",
    "all-MiniLM-L12-v2",
]
FRENCH_EMBEDDING_MODELS = [
    "OrdalieTech/Solon-embeddings-base-0.1",
    "OrdalieTech/Solon-embeddings-large-0.1",
    "dangvantuan/sentence-camembert-large",
    "antoinelouis/biencoder-distilcamembert-mmarcoFR",
]

MULTILINGUAL_EMBEDDING_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
]

# MODEL REPRESENTATIONS
MMR_REPRESENTATION_MODEL = "MaximalMarginalRelevance"
KEYBERTINSPIRED_REPRESENTATION_MODEL = "KeyBERTInspired"
OPENAI_REPRESENTATION_MODEL = "OpenAI"

# BERTopic Hyperparameters
BERTOPIC_SERIALIZATION = "safetensors"  # or pickle
LANGUAGES = ["French", "English", "Multilingual"]
REPRESENTATION_MODELS = [
    MMR_REPRESENTATION_MODEL,
    KEYBERTINSPIRED_REPRESENTATION_MODEL,
    OPENAI_REPRESENTATION_MODEL,
]

# BERTrend parameters
SIGNAL_CLASSIF_LOWER_BOUND = BERTREND_CONFIG["signal_classif_lower_bound"]
SIGNAL_CLASSIF_UPPER_BOUND = BERTREND_CONFIG["signal_classif_upper_bound"]

# Embedding Settings
EMBEDDING_DTYPES = EMBEDDING_CONFIG["embedding_dtypes"]
EMBEDDING_BATCH_SIZE = EMBEDDING_CONFIG["embedding_batch_size"]
EMBEDDING_MAX_SEQ_LENGTH = EMBEDDING_CONFIG["embedding_max_seq_length"]
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Representation models parameters
KEYBERT_NR_REPR_DOCS = (5,)
KEYBERT_NR_CANDIDATE_WORDS = (40,)
KEYBERT_TOP_N_WORDS = 20
OPENAI_NR_DOCS = 5

# Data Processing
MIN_CHARS_DEFAULT = 1
SAMPLE_SIZE_DEFAULT = 1.0  # Or whatever default you want, None means all documents

# Time Settings
DEFAULT_WINDOW_SIZE = 7  # days
MAX_WINDOW_SIZE = 365  # days

# Data Analysis Settings
POPULARITY_THRESHOLD = 0.1  # for weak signal detection, if applicable
