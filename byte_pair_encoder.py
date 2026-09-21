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

if __name__ == '__main__':
    example_string = "ABCDXBCX aber ich bin  ich bin auch inc "
    bpe = BPEPerWord(example_string)
    vocabulary, corpus = bpe.encode(k=5)
    print(f"Corpus: {corpus}")
    print(f"Vocabulary: {vocabulary}")
