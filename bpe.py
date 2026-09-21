"""Shared BPE for Assignments 1 and 3, reusing the group's original helpers.

The supplied bpe.py is an unfinished, non-importable draft; its exact original is
in the source repository. byte_pair_encoder.py is preserved unchanged. Its sorted
vocabulary and counter helpers are reused here. Its corpus-mutating encoder
cannot apply learned merges to held-out text, so that part needs an implementation.
"""
from collections import Counter, defaultdict
import heapq
import json
from pathlib import Path
import re
import unicodedata

from byte_pair_encoder import BytePairEncoder


def normalize(text, strategy="clean"):
    if strategy == "raw":
        return text
    text = text.lower()
    if strategy == "lower":
        return text
    if strategy not in {"clean", "split"}:
        raise ValueError(strategy)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\d", "0", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    if strategy == "split":
        text = "".join(f" {c} " if unicodedata.category(c).startswith("P") else c
                       for c in text)
        text = re.sub(r"[^\S\n]+", " ", text)
    return text


def merge_pair(symbols, pair):
    result, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            result.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return tuple(result)


class BPETokenizer:
    def __init__(self, num_merges=1000, end_of_word="</w>", strategy="clean",
                 start_of_word=None):
        if num_merges < 0:
            raise ValueError("num_merges must be nonnegative")
        normalize("", strategy)  # validate the strategy
        self.num_merges = num_merges
        self.end_of_word = end_of_word
        self.start_of_word = start_of_word
        self.strategy = strategy
        self.marker = start_of_word if start_of_word is not None else end_of_word
        if not self.marker:
            raise ValueError("A nonempty boundary marker is required")
        self.merges = []
        self.vocab = {}
        self._cache = {}

    def train(self, train_text):
        text = normalize(train_text, self.strategy)
        if self.marker in text:
            raise ValueError("The reserved boundary marker occurs in training text")
        frequencies = Counter(text.split())
        words = {w: self._with_boundary(tuple(w)) for w in sorted(frequencies)}
        # Reuse the original deterministic vocabulary helper unchanged.
        alphabet = BytePairEncoder.get_vocabulary(text)
        symbols = ["<bos>", "<eos>", "<unk>", self.marker] + alphabet
        self.vocab = {s: i for i, s in enumerate(dict.fromkeys(symbols))}
        self.merges = []
        counts, members = {}, defaultdict(set)
        for word, pieces in words.items():
            for pair, count in Counter(zip(pieces, pieces[1:])).items():
                BytePairEncoder.add_to_counter(counts, pair, count * frequencies[word])
                members[pair].add(word)
        heap = [(-count, pair) for pair, count in counts.items() if count > 0]
        heapq.heapify(heap)
        for _ in range(self.num_merges):
            while heap:
                negative_count, pair = heapq.heappop(heap)
                if counts.get(pair, 0) == -negative_count:
                    break
            else:
                break
            changed_pairs = set()
            # Sorted iteration and tuple tie-breaking make training deterministic.
            for word in sorted(members[pair].copy()):
                before = Counter(zip(words[word], words[word][1:]))
                words[word] = merge_pair(words[word], pair)
                after = Counter(zip(words[word], words[word][1:]))
                for adjacent in before.keys() | after.keys():
                    difference = (after[adjacent] - before[adjacent]) * frequencies[word]
                    if difference > 0:
                        BytePairEncoder.add_to_counter(counts, adjacent, difference)
                    elif difference < 0:
                        BytePairEncoder.subtract_from_counter(counts, adjacent, -difference)
                    if after[adjacent]:
                        members[adjacent].add(word)
                    else:
                        members[adjacent].discard(word)
                    if difference:
                        changed_pairs.add(adjacent)
            self.merges.append(pair)
            joined = "".join(pair)
            if joined not in self.vocab:
                self.vocab[joined] = len(self.vocab)
            for adjacent in sorted(changed_pairs):
                if counts[adjacent] > 0:
                    heapq.heappush(heap, (-counts[adjacent], adjacent))
        self._prepare()
        return self

    def _with_boundary(self, pieces):
        if self.start_of_word is not None:
            return (self.marker,) + pieces
        return pieces + (self.marker,)

    def _prepare(self):
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._id_to_token = {i: token for token, i in self.vocab.items()}
        self._characters = {token for token in self.vocab
                            if len(token) == 1 and token != self.marker}
        self._cache = {}
        self.bos_id, self.eos_id, self.unk_id = (self.vocab[t] for t in
                                               ("<bos>", "<eos>", "<unk>"))

    def _encode_word(self, word):
        if word not in self._cache:
            pieces = tuple(c if c in self._characters else "<unk>" for c in word)
            # k=0 is the pure character baseline: emit no synthetic marker.
            if self.num_merges != 0:
                pieces = self._with_boundary(pieces)
            while len(pieces) > 1:
                candidates = [p for p in zip(pieces, pieces[1:]) if p in self._ranks]
                if not candidates:
                    break
                pair = min(candidates, key=self._ranks.get)
                pieces = merge_pair(pieces, pair)
            self._cache[word] = [self.vocab[p] for p in pieces]
        return self._cache[word]

    def encode(self, text):
        ids = []
        for chunk in re.findall(r"\s+|\S+", normalize(text, self.strategy)):
            if chunk.isspace():
                ids.extend(self.vocab.get(c, self.unk_id) for c in chunk)
            else:
                ids.extend(self._encode_word(chunk))
        return ids

    def decode(self, ids):
        return "".join(self._without_boundary(self._id_to_token[int(i)]) for i in ids
                       if int(i) not in (self.bos_id, self.eos_id))

    def _without_boundary(self, symbol):
        if self.start_of_word is not None:
            return symbol.removeprefix(self.marker)
        return symbol.removesuffix(self.marker)

    def cut_positions(self, word):
        """Internal character offsets of a normalized word's BPE boundaries."""
        pieces = [self._without_boundary(self._id_to_token[i])
                  for i in self._encode_word(word)]
        position, cuts = 0, []
        for piece in pieces:
            position += len(piece)
            if piece and 0 < position < len(word):
                cuts.append(position)
        return tuple(cuts)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"num_merges": self.num_merges,
            "end_of_word": self.end_of_word, "start_of_word": self.start_of_word,
            "strategy": self.strategy, "merges": self.merges, "vocab": self.vocab}),
            encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(data["num_merges"], data["end_of_word"],
                  strategy=data.get("strategy", "clean"),
                  start_of_word=data.get("start_of_word"))
        tok.merges = [tuple(pair) for pair in data["merges"]]
        tok.vocab = data["vocab"]
        tok._prepare()
        return tok


class WordTokenizer:
    """8,000-word baseline; preserve whitespace, use <unk> for unseen words."""
    def __init__(self, max_words=8000, strategy="clean"):
        self.max_words = max_words
        self.strategy = strategy

    def train(self, text):
        text = normalize(text, self.strategy)
        counts = Counter(text.split())
        words = sorted(counts, key=lambda word: (-counts[word], word))[:self.max_words]
        whitespace = BytePairEncoder.get_vocabulary(c for c in text if c.isspace())
        self.vocab = {token: i for i, token in enumerate(["<unk>"] + whitespace + words)}
        self._id_to_token = {i: token for token, i in self.vocab.items()}
        self.unk_id = self.vocab["<unk>"]
        return self

    def _encode_word(self, word):
        return [self.vocab.get(word, self.unk_id)]

    def encode(self, text):
        ids = []
        for chunk in re.findall(r"\s+|\S+", normalize(text, self.strategy)):
            if chunk.isspace():
                ids.extend(self.vocab.get(c, self.unk_id) for c in chunk)
            else:
                ids.extend(self._encode_word(chunk))
        return ids

    def decode(self, ids):
        return "".join(self._id_to_token[int(i)] for i in ids)

    def cut_positions(self, word):
        return ()  # A word tokenizer makes no within-word cuts.
