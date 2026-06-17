"""토크나이저 / vocab 단위 테스트."""
from config import BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX
from transformer.data.tokenizer import detokenize, tokenize
from transformer.data.vocab import Vocab


def test_tokenize_splits_punctuation():
    assert tokenize("A man, riding!") == ["a", "man", ",", "riding", "!"]


def test_detokenize_roundtrip_is_readable():
    toks = tokenize("a man, riding a bike!")
    assert detokenize(toks) == "a man, riding a bike!"


def test_vocab_special_token_indices_are_fixed():
    vocab = Vocab.build([["a", "b", "c"]], min_freq=1)
    assert vocab.stoi["<pad>"] == PAD_IDX
    assert vocab.stoi["<bos>"] == BOS_IDX
    assert vocab.stoi["<eos>"] == EOS_IDX
    assert vocab.stoi["<unk>"] == UNK_IDX


def test_vocab_min_freq_drops_rare_words():
    corpus = [["a", "a", "b"], ["a", "b"]]  # a:3, b:2, (c 없음)
    vocab = Vocab.build(corpus, min_freq=2)
    assert "a" in vocab.stoi and "b" in vocab.stoi


def test_encode_adds_bos_eos_and_maps_unknown():
    vocab = Vocab.build([["a", "b"]], min_freq=1)
    ids = vocab.encode(["a", "zzz"])  # zzz는 미등록 -> UNK
    assert ids[0] == BOS_IDX and ids[-1] == EOS_IDX
    assert UNK_IDX in ids


def test_decode_strips_special_tokens():
    vocab = Vocab.build([["a", "b"]], min_freq=1)
    ids = vocab.encode(["a", "b"])
    assert vocab.decode(ids) == ["a", "b"]


def test_vocab_save_load_roundtrip(tmp_path):
    vocab = Vocab.build([["hello", "world"]], min_freq=1)
    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = Vocab.load(path)
    assert loaded.itos == vocab.itos
