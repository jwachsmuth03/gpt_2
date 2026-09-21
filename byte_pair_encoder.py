# Aufgabe 1: zusaetzliche Standardbibliotheks-Importe fuer Normalisierung, Persistenz
# und das corpus-weite Training/Encoding weiter unten. Die BytePairEncoder-Klasse
# selbst bleibt unveraendert.
import heapq
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


class BytePairEncoder:
    # An optimized version of the BPE algorithm
    # It has a runtime in O(n+k) because it loops through the whole corpus only 2 times in total
    # (once to count all bigrams in the beginning, then k times the count dictionary is updated only by direct
    # access, and in the end we loop a second time through the corpus to remove artifacts)

    def __init__(self, text):
        self.text = text
        self.vocabulary = self.get_vocabulary(text)
        self.corpus = list(text)

    def encode(self, k):
        # loop through corpus to count all bigrams initially
        bigram_counter, bigram_positions = self.count_bigrams(self.corpus)

        for _ in range(k):
            if self.no_bigrams_left(bigram_counter): break

            # get most frequent bigram
            most_frequent_bigram, most_frequent_bigram_positions = self.get_most_frequent_bigram(
                bigram_counter,bigram_positions)

            # update the counter, positions and vocabulary based on the newly built bigram
            bigram_counter, bigram_positions = self.update_bigram_counter(self.corpus, bigram_counter, bigram_positions,
                                                                     most_frequent_bigram_positions.copy())
            self.vocabulary.append(most_frequent_bigram)

            # replace the single tokens with the new bigram in the corpus and add a placeholder "\\empty" for now
            # empty places with the idea to remove all of them with a single loop in the end
            self.corpus = self.replace_in_corpus(corpus=self.corpus, positions=most_frequent_bigram_positions,
                                       new_entry=most_frequent_bigram)
            positions_after_bigram = [self.get_next_not_empty_position(self.corpus, i) for i in most_frequent_bigram_positions]
            self.corpus = self.replace_position_with_placeholder(self.corpus, positions_after_bigram)
            print(f"Bigram: #{most_frequent_bigram}#")

        # loop through the corpus once to delete the placeholders
        self.corpus = self.remove_placeholder_from_corpus(self.corpus)
        return self.vocabulary, self.corpus

    @staticmethod
    def no_bigrams_left(bigram_counter):
        return not any(count > 0 for count in bigram_counter.values())

    @staticmethod
    def get_vocabulary(text):
        return sorted(set(text))

    @staticmethod
    def count_bigrams(corpus):
        bigram_counter = {}
        bigram_positions = {}
        for i in range(len(corpus) - 1):
            bigram = corpus[i] + corpus[i + 1]
            # add plus 1 to the bigram counter
            bigram_counter[bigram] = bigram_counter.get(bigram, 0) + 1  # (use default 0 if not in dictionary yet)
            # add the positions to the bigram_positions dictionary
            if bigram not in bigram_positions:
                bigram_positions[bigram] = []
            bigram_positions[bigram].append(i)
        return bigram_counter, bigram_positions

    @staticmethod
    def get_most_frequent_bigram(bigram_counter, bigram_positions):
        most_frequent_bigram = max(bigram_counter, key=bigram_counter.get)
        most_frequent_bigram_positions = bigram_positions[most_frequent_bigram]
        return most_frequent_bigram, most_frequent_bigram_positions.copy()

    def update_bigram_counter(self, corpus, bigram_counter, bigram_positions, positions_of_new_bigram):
        # add occurences to the count of the new bigram
        for i in positions_of_new_bigram:
            # subtract 1 from the bigram counts of the 2 old bigrams
            bigram_1, bigram_1_position = self.get_bigram(corpus, position=i)
            bigram_2, bigram_2_position = self.get_bigram(corpus, position=i - 1, step_size=-1)
            bigram_3, bigram_3_position = self.get_bigram(corpus, position=i + 1)
            self.subtract_from_counter(bigram_counter, bigram_1, 1)
            self.subtract_from_counter(bigram_counter, bigram_2, 1)
            self.subtract_from_counter(bigram_counter, bigram_3, 1)
            self.remove_position_from_bigram(bigram_positions, bigram_1, bigram_1_position)
            self.remove_position_from_bigram(bigram_positions, bigram_2, bigram_2_position)
            self.remove_position_from_bigram(bigram_positions, bigram_3, bigram_3_position)

            # add 1 to the bigram counts of the 2 new bigrams (which were trigrams originally, so we look for trigrams)
            bigram_4, bigram_4_position = self.get_trigram(corpus, position=i)
            bigram_5, bigram_5_position = self.get_trigram(corpus, position=i - 1, step_size=-1)
            self.add_to_counter(bigram_counter, bigram_4, 1)
            self.add_to_counter(bigram_counter, bigram_5, 1)
            self.add_position_to_bigram(bigram_positions, bigram_4, bigram_4_position)
            self.add_position_to_bigram(bigram_positions, bigram_5, bigram_5_position)

        return bigram_counter, bigram_positions

    @staticmethod
    def add_position_to_bigram(bigram_positions_dict, bigram, position):
        if bigram is None: return
        bigram_positions_dict.setdefault(bigram, []).append(position)

    @staticmethod
    def remove_position_from_bigram(bigram_positions_dict, bigram, position):
        if bigram is None: return
        if bigram in bigram_positions_dict and position in bigram_positions_dict[bigram]:
            bigram_positions_dict[bigram].remove(position)

    @staticmethod
    def get_next_not_empty_position(corpus, position):
        try:
            i = position + 1
            while corpus[i] == '\\empty': i += 1
            return i
        except IndexError:
            return None, None

    @staticmethod
    def get_bigram(corpus, position, step_size=1):
        try:
            i = position
            while corpus[i] == '\\empty': i += step_size
            if i < 0: return None, None
            first_token = corpus[i]
            final_position = i
            i += 1
            while corpus[i] == '\\empty': i += 1
            second_token = corpus[i]
            return first_token + second_token, final_position
        except IndexError:
            return None, None

    @staticmethod
    def get_trigram(corpus, position, step_size=1):
        try:
            i = position
            while corpus[i] == '\\empty': i += step_size
            if i < 0: return None, None
            first_token = corpus[i]
            final_position = i
            i += 1
            while corpus[i] == '\\empty': i += 1
            second_token = corpus[i]
            i += 1
            while corpus[i] == '\\empty': i += 1
            third_token = corpus[i]
            return first_token + second_token + third_token, final_position
        except IndexError:
            return None, None

    @staticmethod
    def subtract_from_counter(counter, key, number):
        if key is None: return
        counter[key] = max(0, counter.get(key, 0)) - number

    @staticmethod
    def add_to_counter(counter, key, number):
        if key is None: return
        counter[key] = counter.get(key, 0) + number

    @staticmethod
    def replace_in_corpus(corpus, positions, new_entry):
        for i in positions:
            corpus[i] = new_entry
        return corpus

    @staticmethod
    def remove_positions_from_corpus(corpus, positions):
        result = []
        for i in range(len(corpus)):
            if i not in positions:
                result.append(corpus[i])
        return result

    @staticmethod
    def replace_position_with_placeholder(corpus, positions):
        for i in positions:
            corpus[i] = '\\empty'
        return corpus

    @staticmethod
    def remove_placeholder_from_corpus(corpus):
        result = []
        for i in range(len(corpus)):
            if corpus[i] != '\\empty':
                result.append(corpus[i])
        return result

class BPEPerCharacter(BytePairEncoder):
    # The BPEPerCharacter encodes the text with every initial token being a character
    def __init__(self, text):
        super().__init__(text)

class BPEPerWord(BytePairEncoder):
    # The BPEPerCharacter encodes the text with every initial token being a word
    def __init__(self, text):
        super().__init__(text)
        self.corpus = text.split(" ")
        self.corpus = [i + " " for i in self.corpus]
        self.vocabulary = self.get_vocabulary(self.corpus)

# Aufgabe 1: Normalisierungsstrategien (raw/lower/clean/split) fuer den corpus-weiten
# Tokenizer unten. raw laesst den Text unveraendert, lower lowercased nur, clean wendet
# lower -> NFKC -> Ziffern-Ersetzung -> Leerraum-Kollaps an, split umschliesst zusaetzlich
# jedes Unicode-Satzzeichen mit Leerzeichen.
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


# Aufgabe 2: Merge-Anwendung ueber Tupel-Paar-Keys statt verketteter Strings; ersetzt
# ueberlappende Treffer von links nach rechts ohne Ueberlappung.
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
    """# Aufgabe 2/3: corpus-weiter BPE-Tokenizer, der die urspruengliche
    BytePairEncoder-Klasse ergaenzt statt sie zu ersetzen.

    Im Unterschied zu BytePairEncoder.encode() oben: merged nie ueber Wortgrenzen
    hinweg, arbeitet mit Tupel-Paar-Keys statt verketteten Strings, speichert die
    gelernten Merge-Regeln in Reihenfolge (wiederverwendbar fuer Hold-out-Text) und
    unterstuetzt Normalisierung, Speichern und Laden.
    """

    # Aufgabe 2: Konstruktor mit Normalisierungsstrategie und Wortgrenzen-Marker
    # (Suffix </w> oder alternativ Praefix ▁).
    def __init__(self, num_merges=1000, end_of_word="</w>", strategy="clean",
                 start_of_word=None):
        if num_merges < 0:
            raise ValueError("num_merges must be nonnegative")
        normalize("", strategy)  # Strategie validieren
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

    # Aufgabe 2: Training mit Heap-basierter, deterministischer Merge-Auswahl
    # (alphabetischer Tie-Break, lokale Updates nur fuer betroffene Wortarten).
    def train(self, train_text):
        text = normalize(train_text, self.strategy)
        if self.marker in text:
            raise ValueError("The reserved boundary marker occurs in training text")
        frequencies = Counter(text.split())
        words = {w: self._with_boundary(tuple(w)) for w in sorted(frequencies)}
        # Aufgabe 2: Wiederverwendung der unveraenderten Vokabular-/Zaehl-Hilfsmethoden
        # von BytePairEncoder.
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
            # Sortierte Iteration + Tupel-Tie-Break machen das Training deterministisch.
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

    # Aufgabe 2: Wortgrenzen-Marker anhaengen (Suffix) oder voranstellen (Praefix).
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

    # Aufgabe 3a: Encoding von (auch ungesehenem) Text mit den gelernten, geordneten
    # Merge-Regeln. Unbekannte Zeichen werden einzeln auf <unk> abgebildet.
    def _encode_word(self, word):
        if word not in self._cache:
            pieces = tuple(c if c in self._characters else "<unk>" for c in word)
            # k=0 ist die reine Zeichen-Baseline: kein synthetischer Marker.
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

    # Aufgabe 3a: Decoding entfernt nur den synthetischen Wortgrenzen-Marker,
    # Leerzeichen/Zeilenumbrueche bleiben eigene Tokens und somit reversibel.
    def decode(self, ids):
        return "".join(self._without_boundary(self._id_to_token[int(i)]) for i in ids
                       if int(i) not in (self.bos_id, self.eos_id))

    def _without_boundary(self, symbol):
        if self.start_of_word is not None:
            return symbol.removeprefix(self.marker)
        return symbol.removesuffix(self.marker)

    # Aufgabe 3b: Speichern und Laden der gelernten Merge-Regeln und des Vokabulars
    # als JSON; Laden reproduziert dieselben IDs und decodierten Strings.
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


if __name__ == '__main__':
    example_string = "ABCDXBCX aber ich bin  ich bin auch inc "
    bpe = BPEPerWord(example_string)
    vocabulary, corpus = bpe.encode(k=5)
    print(f"Corpus: {corpus}")
    print(f"Vocabulary: {vocabulary}")
