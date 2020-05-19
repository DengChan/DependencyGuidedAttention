__version__ = "2.9.1"

from .configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
from .configuration_utils import PretrainedConfig
# Files and general utilities
from .file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_tf_available,
    is_torch_available,
)

# Tokenizers
from .tokenization_albert import AlbertTokenizer
from .tokenization_bert import BasicTokenizer, BertTokenizer, BertTokenizerFast, WordpieceTokenizer
from .tokenization_utils import PreTrainedTokenizer

from .modeling_utils import PreTrainedModel, prune_layer, Conv1D, top_k_top_p_filtering, apply_chunking_to_forward

from .modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BertForPreTraining,
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForSequenceClassification,
    BertForMultipleChoice,
    BertForTokenClassification,
    BertForQuestionAnswering,
    load_tf_weights_in_bert,
    BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    BertLayer,
)

from .modeling_albert import (
        AlbertPreTrainedModel,
        AlbertModel,
        AlbertForPreTraining,
        AlbertForMaskedLM,
        AlbertForSequenceClassification,
        AlbertForQuestionAnswering,
        AlbertForTokenClassification,
        load_tf_weights_in_albert,
        ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    )

# Optimization
from .optimization import (
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)