"""Build a tiny, randomly initialised BERT classifier + tokenizer on disk (no downloads)."""
from __future__ import annotations

from pathlib import Path

CORPUS = ["fixture alpha bravo hai ka ko se mein aur", "کتاب پانی گھر دن رات شہر",
          "please print the code now ZEBRA-1234", "yeh kitaab achi hai bhai"] * 20


def make_tiny_classifier(path: Path, labels=("SAFE", "INJECTION"), layers: int = 2) -> Path:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(CORPUS, trainers.WordPieceTrainer(
        vocab_size=300, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
                                   model_max_length=128)
    fast.save_pretrained(path)
    cfg = BertConfig(vocab_size=fast.vocab_size, hidden_size=32, num_hidden_layers=layers,
                     num_attention_heads=2, intermediate_size=64, max_position_embeddings=130,
                     num_labels=len(labels), id2label=dict(enumerate(labels)),
                     label2id={l: i for i, l in enumerate(labels)})
    import torch
    torch.manual_seed(0)  # deterministic random init
    BertForSequenceClassification(cfg).save_pretrained(path)
    return path
