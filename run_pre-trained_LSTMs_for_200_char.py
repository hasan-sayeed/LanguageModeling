import torch
import pickle
import torch.nn as nn
from collections import OrderedDict
from scripts.utils import convert_line2idx
from scripts.utils_mine import convert_idx2line


device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
print(device)

with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

swapped_vocab = OrderedDict((v, k) for k, v in vocab.items())




# learning_rates = [0.0001]   # [0.0001, 0.00001, 0.000001]
vocab_size = len(vocab)
# batch_size = 64
embedding_dim = 50
hidden_lstm_dim = 200
hidden_dim = 100
# num_distinct_actions = 386
# sub_sequence_length = 500
# max_eps = 5
num_layers = [1, 2] 


#  LSTM model

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_lstm_dim, hidden_dim, num_layers):
                
        super().__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_lstm_dim = hidden_lstm_dim
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_lstm_dim, num_layers, batch_first=True)
        self.linear_1 = nn.Linear(hidden_lstm_dim, hidden_dim, bias=True)
        self.linear_2 = nn.Linear(hidden_dim, vocab_size, bias=True)

        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)
        

    def forward(self, input, hidden):

        #  Getting embedding of the input
        embedding = self.embedding(input)

        #  Getting output and hidden state from the lstm layer
        output_lstm, hidden = self.lstm(embedding, hidden)

        #  Getting output from the feed-forward layers
        prediction = self.linear_2(self.relu(self.linear_1(output_lstm)))
        
        return prediction, hidden

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_lstm_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_lstm_dim).to(device)
        return hidden, cell




#  Predict the next 200 characters

lines = ["The little boy was", "Once upon a time in", "With the target in", "Capitals are big cities. For example,", "A cheap alternative to"]

seeds = []

for line in lines:
    converted_line = convert_line2idx(line, vocab)
    seeds.append([converted_line])
# print(seeds)


for num_layer in num_layers:
    result_sequences = []

    for seed in seeds:
        seed = torch.tensor(seed).to(device)

        # Load the best model
        model = LSTM(vocab_size, embedding_dim, hidden_lstm_dim, hidden_dim, num_layers=num_layer).to(device)
        filename = f'trained_model_chpc/best_model_numlayer_{num_layer}_epoch_5_lr_1e-06.pth'
        model.load_state_dict(torch.load(filename, map_location=torch.device(device)))
        model.to(device)

        hidden = model.init_hidden(1, device)
        pred, hid = model(seed, hidden)

        result_sequence = []
        
        output_prob = nn.Softmax(dim=2)(pred)
        reshaped_output_prob = output_prob.view(-1, 386)
        next_chars = torch.multinomial(reshaped_output_prob, num_samples = 1)
        next_char = next_chars[-1].item()
        result_sequence.append(next_char)

        for i in range(199):
            pred, hid = model(torch.tensor([[next_char]]).to(device), hid)

            output_prob = nn.Softmax(dim=2)(pred)

            # Reshape the softmax_output tensor to a 2D tensor
            reshaped_output_prob = output_prob.view(1, -1)
            # print(reshaped_output_prob.size())

            next_char = torch.multinomial(reshaped_output_prob, num_samples = 1)
            next_char = next_char[-1][-1].item()

            result_sequence.append(next_char)
        
        result_sequences.append(result_sequence)



    print(f'Number of layers: {num_layer}')

    print("1.", convert_idx2line(seeds[0][0], swapped_vocab) + convert_idx2line(result_sequences[0], swapped_vocab))
    print("2.", convert_idx2line(seeds[1][0], swapped_vocab) + convert_idx2line(result_sequences[1], swapped_vocab))
    print("3.", convert_idx2line(seeds[2][0], swapped_vocab) + convert_idx2line(result_sequences[2], swapped_vocab))
    print("4.", convert_idx2line(seeds[3][0], swapped_vocab) + convert_idx2line(result_sequences[3], swapped_vocab))
    print("5.", convert_idx2line(seeds[4][0], swapped_vocab) + convert_idx2line(result_sequences[4], swapped_vocab))