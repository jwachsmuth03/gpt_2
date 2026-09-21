from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from n_gram import UniGram, BiGram

from byte_pair_encoder import BPEPerCharacter, BPEPerWord


class NeuralNGram(nn.Module):
    def __init__(self, n, embedding_size, hidden_layer_size, corpus):
        super(NeuralNGram, self).__init__()
        self.n = n
        self.embedding_size = embedding_size
        self.hidden_layer_size = hidden_layer_size
        self.corpus = corpus
        self.vocabulary = set(corpus)
        self.vocabulary_size = len(set(corpus))
        self.embeddings = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.input_layer = nn.Linear((self.n-1) * self.embedding_size, self.hidden_layer_size)
        self.hidden_layer = nn.Linear(self.hidden_layer_size, self.vocabulary_size)
        self.n_grams = []
        self.token_index_dict = { token: i for i, token in enumerate(self.vocabulary) }
        self.index_token_dict = {i: token for token, i in self.token_index_dict.items()}

    def forward(self, inputs):
        # get embeddings of the last n-1 words and concatenates them into one vector
        input_embeddings = self.embeddings(inputs).view(inputs.size(0), -1) # (batch_size, (n-1)*embedding_size)

        # pass the input through the input and hidden layer
        input_layer_output = F.relu(self.input_layer(input_embeddings))
        logits = self.hidden_layer(input_layer_output)
        return logits

    def train_model(self, epochs=100, learning_rate=0.01, batch_size=64):
        loss_function = nn.CrossEntropyLoss()
        gradient_descent_optimizer = optim.SGD(self.parameters(), lr = learning_rate)

        # Precompute all context/target indices once
        context_tensor = torch.tensor(
            [[self.token_to_index(t) for t in context] for context, _ in self.n_grams],
            dtype=torch.long
        )  # shape: (num_examples, n-1)
        target_tensor = torch.tensor(
            [self.token_to_index(target) for _, target in self.n_grams],
            dtype=torch.long
        )  # shape: (num_examples,)

        dataset = torch.utils.data.TensorDataset(context_tensor, target_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for context_batch, target_batch in loader:
                gradient_descent_optimizer.zero_grad()
                logits = self.forward(context_batch)
                loss = loss_function(logits, target_batch)
                loss.backward()
                gradient_descent_optimizer.step()
                total_loss += loss.item()
            if epoch % 1 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    def token_to_index(self, token):
        return self.token_index_dict[token]

    def index_to_token(self, index):
        return self.index_token_dict[index]

    def predict_next_token(self, last_tokens, temperature=1.0):
        context_indices = torch.tensor(
            [self.token_to_index(token) for token in last_tokens], dtype=torch.long
            ).unsqueeze(0)
        with torch.no_grad(): # skips pytorchs automatic gradient computation
            logits = self.forward(context_indices)
            # sampling
            probabilities = F.softmax(logits / temperature, dim=1)
            predicted_index = torch.multinomial(probabilities, num_samples=1).item()
        return self.index_to_token(predicted_index)

    @abstractmethod
    def build_sentence_as_array(self, number_of_tokens):
        pass

    def build_sentence(self, number_of_tokens):
        return ''.join(self.build_sentence_as_array(number_of_tokens))

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()

class NeuralBiGram(NeuralNGram):
    def __init__(self, embedding_size, hidden_layer_size, corpus):
        super().__init__(2, embedding_size, hidden_layer_size, corpus)
        self.n_grams = [ ([self.corpus[i]], self.corpus[i+1]) for i in range(len(self.corpus)-1) ]
        self.unigram_model = UniGram()  # the unigram model is used for generating the first token without context
        self.unigram_model.train(corpus)

    def build_sentence_as_array(self, number_of_tokens):
        result = self.unigram_model.build_sentence_as_array(1) # generate a first token with the standard unigram model

        # builds the sentence
        for i in range(number_of_tokens - 1):
            result.append(self.predict_next_token([result[i]]))
        return result


class NeuralTriGram(NeuralNGram):
    def __init__(self, embedding_size, hidden_layer_size, corpus):
        super().__init__(3, embedding_size, hidden_layer_size, corpus)
        self.n_grams = [((self.corpus[i], self.corpus[i + 1]), self.corpus[i + 2]) for i in range(len(self.corpus) - 2)]
        self.bigram_model = BiGram() # the bigram model is used for generating a first 2 tokens without context
        self.bigram_model.train(corpus)

    def build_sentence_as_array(self, number_of_tokens):
        result = self.bigram_model.build_sentence_as_array(2) # generate a first token with the standard bigram model


        # builds the sentence
        for i in range(number_of_tokens - 2):
            result.append(self.predict_next_token([result[i], result[i + 1]]))
        return result

if __name__ == '__main__':
    EMBEDDING_SIZE = 10
    HIDDEN_LAYER_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 0.01

    file_path = 'shakespeare_corpus.txt'

    with open(file_path, 'r') as file:
        file_content = file.read()

    bpe = BPEPerWord(file_content)
    _, shakespeare_corpus = bpe.encode(k=500)

    #neural_bigram = NeuralBiGram(EMBEDDING_SIZE, HIDDEN_LAYER_SIZE, shakespeare_corpus)
    #neural_bigram.train_model(epochs=EPOCHS, learning_rate=LEARNING_RATE)
    #neural_bigram.save_model('saved_models/neural_bigram.pt')

    #neural_trigram = NeuralTriGram(EMBEDDING_SIZE, HIDDEN_LAYER_SIZE, shakespeare_corpus)
    #neural_trigram.train_model(epochs=EPOCHS, learning_rate=LEARNING_RATE)
    #neural_trigram.save_model('saved_models/neural_trigram.pt')

    loaded_bigram = NeuralBiGram(EMBEDDING_SIZE, HIDDEN_LAYER_SIZE, shakespeare_corpus)
    loaded_bigram.load_model('saved_models/neural_bigram.pt')

    loaded_trigram = NeuralTriGram(EMBEDDING_SIZE, HIDDEN_LAYER_SIZE, shakespeare_corpus)
    loaded_trigram.load_model('saved_models/neural_trigram.pt')

    print("\n\n##################################################")
    print(f"Bigram Sentence: {loaded_bigram.build_sentence(20)}")
    print("\n\n##################################################")
    print(f"Trigram Sentence: {loaded_trigram.build_sentence(20)}")
