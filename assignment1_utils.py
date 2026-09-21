"""Standard-library-only data and measurement helpers for Assignment 1."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

from bpe import normalize

SOURCES = {
    "Shakespeare": {
        "file": "shakespeare_corpus.txt",
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "sha256": "434c0554a8c4c53dc17e56a0abb0f30b88f83cbceb0289cb897db68c25e89eba",
    },
    "English prose": {
        "file": "pride_and_prejudice.txt",
        "url": "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
        "sha256": "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75",
    },
    "Python source": {
        "file": "cpython_test_typing.txt",
        "url": "https://raw.githubusercontent.com/python/cpython/v3.13.0/Lib/test/test_typing.py",
        "sha256": "750a931fa976f8ca7a113f946eddda380a2d68b64f94bc251b8eb338f30ca4d8",
    },
}


def load_corpora():
    texts, metadata = {}, []
    for name, source in SOURCES.items():
        path = Path("data") / source["file"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(source["url"], timeout=60) as response:
                content = response.read()
            path.write_bytes(content)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != source["sha256"]:
            raise ValueError(f"{name}: checksum changed. Verify the source before updating its manifest.")
        if name != "Shakespeare" and len(content) < 200_000:
            raise ValueError(f"{name} must contain at least 200 kB.")
        # Normalized newline representation; keep the complete downloaded text.
        texts[name] = path.read_text(encoding="utf-8")
        metadata.append(dict(corpus=name, bytes=len(content), sha256=digest, url=source["url"]))
    return texts, metadata


def stability(tokenizers, text, strategy):
    """Exact internal-cut agreement over distinct words encodable by both models."""
    first, second = tokenizers
    words = sorted(set(normalize(text, strategy).split()))
    eligible = [word for word in words
                if first.unk_id not in first._encode_word(word)
                and second.unk_id not in second._encode_word(word)]
    matches = sum(first.cut_positions(word) == second.cut_positions(word) for word in eligible)
    return {
        "stability": matches / len(eligible) if eligible else float("nan"),
        "stability eligible types": len(eligible),
        "stability coverage": len(eligible) / len(words) if words else float("nan"),
    }


def roundtrip_accuracy(tokenizer, text):
    lines = text.splitlines()
    matches = sum(tokenizer.decode(tokenizer.encode(line)) == normalize(line, tokenizer.strategy)
                  for line in lines)
    return matches / len(lines) if lines else 1.0


def measure(tokenizer, text, half_models=None):
    normalized = normalize(text, tokenizer.strategy)
    ids = tokenizer.encode(text)
    characters, words, tokens = len(normalized), len(normalized.split()), len(ids)
    unknown = ids.count(tokenizer.unk_id)
    result = {
        "characters": characters, "words": words, "tokens": tokens,
        "compression ratio": characters / tokens if tokens else float("nan"),
        "tokens per word": tokens / words if words else float("nan"),
        "unknown tokens": unknown, "UNK rate": unknown / tokens if tokens else 0.0,
        "round-trip accuracy": roundtrip_accuracy(tokenizer, text),
    }
    if half_models is not None:
        result.update(stability(half_models, text, tokenizer.strategy))
    return result
