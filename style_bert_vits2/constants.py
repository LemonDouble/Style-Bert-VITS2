from pathlib import Path

from style_bert_vits2.utils.strenum import StrEnum


VERSION = "0.1.0"

BASE_DIR = Path(__file__).parent.parent


class Languages(StrEnum):
    JP = "JP"
    EN = "EN"
    ZH = "ZH"


# HuggingFace repo IDs for auto-download
BERT_REPO_IDS = {
    Languages.JP: "ku-nlp/deberta-v2-large-japanese-char-wwm",
    Languages.EN: "microsoft/deberta-v3-large",
    Languages.ZH: "hfl/chinese-roberta-wwm-ext-large",
}

# Legacy local paths (used by internal bert_models.py fallback)
DEFAULT_BERT_MODEL_PATHS = {
    Languages.JP: BASE_DIR / "bert" / "deberta-v2-large-japanese-char-wwm",
    Languages.EN: BASE_DIR / "bert" / "deberta-v3-large",
    Languages.ZH: BASE_DIR / "bert" / "chinese-roberta-wwm-ext-large",
}

DEFAULT_USER_DICT_DIR = BASE_DIR / "dict_data"

# Default inference parameters
DEFAULT_STYLE = "Neutral"
DEFAULT_STYLE_WEIGHT = 1.0
DEFAULT_SDP_RATIO = 0.2
DEFAULT_NOISE = 0.6
DEFAULT_NOISEW = 0.8
DEFAULT_LENGTH = 1.0
DEFAULT_LINE_SPLIT = True
DEFAULT_SPLIT_INTERVAL = 0.5
DEFAULT_ASSIST_TEXT_WEIGHT = 1.0
