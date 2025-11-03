#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

# Success Messages
STATE_SAVED_MESSAGE = "Application state saved."
STATE_RESTORED_MESSAGE = "Application state restored."
MODELS_SAVED_MESSAGE = "Models saved."
MODELS_RESTORED_MESSAGE = "Models restored."
MODEL_MERGING_COMPLETE_MESSAGE = "Model merging complete!"
TOPIC_COUNTS_SAVED_MESSAGE = "Topic and signal counts saved to {file_path}"
CACHE_PURGED_MESSAGE = "Cache purged."

PROGRESS_BAR_DESCRIPTION = "Batches processed"

# Error Messages
NO_DATA_WARNING = "No data available for the selected granularity."
NO_STATE_WARNING = "No saved state found."
NO_MODELS_WARNING = "No saved models found."
NO_CACHE_WARNING = "No cache found."
EMBED_WARNING = "Please embed data before proceeding to model training."
EMBED_TRAIN_WARNING = (
    "Please embed data and train models before proceeding to analysis."
)
TRAIN_WARNING = "Please train models before proceeding to analysis."
MERGE_WARNING = "Please merge models to view additional analyses."
TOPIC_NOT_FOUND_WARNING = (
    "Topic {topic_number} not found in the merge histories within the specified window."
)
NO_GRANULARITY_WARNING = "Granularity value not found."
HTML_GENERATION_FAILED_WARNING = "HTML generation failed. Displaying markdown instead."
NO_TOPIC_DATA_WARNING = (
    "No data available for topic {topic_number} within the specified date range. "
    "Please select a different topic or adjust the date range."
)
SIGNAL_ANALYSIS_ERROR = "An error occurred while generating the signal analysis. Please check the logs for details."
