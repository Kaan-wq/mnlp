from enum import Enum

from .gqa import MaskedGroupedQuerySelfAttention
from .mha import MaskedMultiHeadSelfAttention
from .mqa import MaskedMultiQuerySelfAttention


class AttentionType(Enum):
    MHA = "mha"
    GQA = "gqa"
    MQA = "mqa"


ATTENTION_REGISTRY = {
    AttentionType.MHA: MaskedMultiHeadSelfAttention,
    AttentionType.GQA: MaskedGroupedQuerySelfAttention,
    AttentionType.MQA: MaskedMultiQuerySelfAttention,
}
