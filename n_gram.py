from abc import abstractmethod
from collections import Counter
import random

from byte_pair_encoder import BPEPerCharacter, BPEPerWord


class NGram:
    def __init__(self, n):
        self.n = n
        self.unigram_counts = {}
        self.bigram_counts = {}
        self.trigram_counts = {}
        self.corpus = []
        self.vocabulary_size = 0

    def train(self, corpus):
        self.corpus = corpus
        self.vocabulary_size = len(set(corpus))

    def count_unigrams(self):
        return Counter(self.corpus)

    def count_bigrams(self):
        bigram_counter = {}
        for i in range(len(self.corpus) - 1):
            bigram = self.corpus[i] + self.corpus[i + 1]
            # add plus 1 to the bigram counter
            bigram_counter[bigram] = bigram_counter.get(bigram, 0) + 1
        return bigram_counter

    def count_trigrams(self):
        trigram_counter = {}
        for i in range(len(self.corpus) - 2):
            trigram = self.corpus[i] + self.corpus[i + 1] + self.corpus[i + 2]
            # add plus 1 to the trigram counter
            trigram_counter[trigram] = trigram_counter.get(trigram, 0) + 1
        return trigram_counter

    def get_most_probable_n_gram(self, n, last_n_minus_one_gram):
        if n == 1:
            return max(self.unigram_counts, key=self.unigram_counts.get)

        if n == 2:
            n_gram_counts = self.bigram_counts
            n_minus_one_gram_counts = self.unigram_counts

        if n == 3:
            n_gram_counts = self.trigram_counts
            n_minus_one_gram_counts = self.bigram_counts

        most_probable_next_token = ""
        probability_next_token = 0
        for next_token, _ in n_minus_one_gram_counts.items():
            n_gram = last_n_minus_one_gram + next_token
            probability = ((n_gram_counts.get(n_gram, 0) + 1) /
                           (n_minus_one_gram_counts.get(last_n_minus_one_gram, 0) + self.vocabulary_size))
            if probability > probability_next_token:
                most_probable_next_token = next_token
                probability_next_token = probability
        return most_probable_next_token

    def sample_n_gram(self, n, last_n_minus_one_gram):
        if n == 1:
            return self.sample_unigram()

        if n == 2:
            n_gram_counts = self.bigram_counts
            n_minus_one_gram_counts = self.unigram_counts

        if n == 3:
            n_gram_counts = self.trigram_counts
            n_minus_one_gram_counts = self.bigram_counts

        all_probabilities = {}
        for next_token, _ in n_minus_one_gram_counts.items():
            n_gram = last_n_minus_one_gram + next_token
            probability = ((n_gram_counts.get(n_gram, 0) + 1) /
                           (n_minus_one_gram_counts.get(last_n_minus_one_gram, 0) + self.vocabulary_size))
            all_probabilities[next_token] = probability
        return self.get_sample_from_counts(all_probabilities)

    def sample_unigram(self):
        return self.get_sample_from_counts(self.unigram_counts)

    def get_sample_from_counts(self, count_dict):
        return random.choices(list(count_dict.keys()), weights=list(count_dict.values()), k=1)[0]

    @abstractmethod
    def build_sentence_as_array(self, number_of_tokens):
        pass

    def build_sentence(self, number_of_tokens):
        return ''.join(self.build_sentence_as_array(number_of_tokens))

class UniGram(NGram):
    def __init__(self):
        super().__init__(1)

    def train(self, corpus):
        super().train(corpus)
        self.unigram_counts = self.count_unigrams()

    def build_sentence_as_array(self, number_of_tokens):
        result = []
        for i in range(number_of_tokens):
            result.append(self.sample_unigram())
        return result

class BiGram(NGram):
    def __init__(self):
        super().__init__(2)

    def train(self, corpus):
        super().train(corpus)
        self.unigram_counts = self.count_unigrams()
        self.bigram_counts = self.count_bigrams()

    def build_sentence_as_array(self, number_of_tokens):
        first_token = self.get_most_probable_n_gram(1, '')

        # builds the sentence
        result = [first_token]
        for i in range(number_of_tokens - 1):
            result.append(self.sample_n_gram(2, result[i]))
        return result


class TriGram(NGram):
    def __init__(self):
        super().__init__(3)

    def train(self, corpus):
        super().train(corpus)
        self.unigram_counts = self.count_unigrams()
        self.bigram_counts = self.count_bigrams()
        self.trigram_counts = self.count_trigrams()

    def build_sentence_as_array(self, number_of_tokens):
        first_token = self.get_most_probable_n_gram(1, '')
        second_token = self.get_most_probable_n_gram(2, first_token)

        # builds the sentence
        result = [first_token, second_token]
        for i in range(number_of_tokens - 2):
            result.append(self.sample_n_gram(3, result[i] + result[i + 1]))
        return result

if __name__ == '__main__':
    file_path = 'shakespeare_corpus.txt'

    with open(file_path, 'r') as file:
        file_content = file.read()
    # example_string = "This is an example string for byte pair encoding."

    bpe = BPEPerWord(file_content)
    vocabulary, corpus = bpe.encode(k=100)

    unigram = UniGram()
    unigram.train(corpus)

    bigram = BiGram()
    bigram.train(corpus)

    trigram = TriGram()
    trigram.train(corpus)

    print(f"Unigram Sentence: {unigram.build_sentence(20)}")
    print(f"Bigram Sentence: {bigram.build_sentence(20)}")
    print(f"Trigram Sentence: {trigram.build_sentence(20)}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
