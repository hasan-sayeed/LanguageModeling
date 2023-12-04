import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scripts.utils import get_files, convert_files2idx
from scripts.utils_mine import get_input_and_target_subsequences, count_ngrams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)


#  Read files and convert to list of idx. Finally get lists of subsequences.

with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

train_files = get_files('data/train')
dev_files = get_files('data/dev')
test_files = get_files('data/test')

list_of_file2idx_train = convert_files2idx(train_files, vocab)
list_of_file2idx_dev = convert_files2idx(dev_files, vocab)
list_of_file2idx_test = convert_files2idx(test_files, vocab)

processed_train_input, processed_train_target = get_input_and_target_subsequences(list_of_file2idx_train, vocab, subsequence_length=500)
processed_dev_input, processed_dev_target = get_input_and_target_subsequences(list_of_file2idx_dev, vocab, subsequence_length=500)
processed_test_input, processed_test_target = get_input_and_target_subsequences(list_of_file2idx_test, vocab, subsequence_length=500)


#  Hyperparameters

learning_rates = [0.0001, 0.00001, 0.000001]
vocab_size = len(vocab)
batch_size = 64
embedding_dim = 50
hidden_lstm_dim = 200
hidden_dim = 100
num_distinct_actions = 386
sub_sequence_length = 500
max_eps = 5
num_layers = [1,2]


#  Dataset and dataloader

processed_train_input, processed_train_target = get_input_and_target_subsequences(list_of_file2idx_train, vocab, subsequence_length=500)
processed_dev_input, processed_dev_target = get_input_and_target_subsequences(list_of_file2idx_dev, vocab, subsequence_length=500)
processed_test_input, processed_test_target = get_input_and_target_subsequences(list_of_file2idx_test, vocab, subsequence_length=500)


# Creating the datasets and dataloaders

train_dataset = TensorDataset(
                          torch.tensor(np.array(processed_train_input)),
                          torch.tensor(np.array(processed_train_target))
                        )
dev_dataset = TensorDataset(
                          torch.tensor(np.array(processed_dev_input)),
                          torch.tensor(np.array(processed_dev_target))
                        )
test_dataset = TensorDataset(
                          torch.tensor(np.array(processed_test_input)),
                          torch.tensor(np.array(processed_test_target))
                          )

train_loader = DataLoader(train_dataset, batch_size=batch_size)
dev_loader = DataLoader(dev_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1)

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

    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_lstm_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_lstm_dim).to(device)
        return hidden, cell
        

    def forward(self, input, hidden):

        #  Getting embedding of the input
        embedding = self.embedding(input)

        #  Getting output and hidden state from the lstm layer
        output_lstm, hidden = self.lstm(embedding, hidden)

        #  Getting output from the feed-forward layers
        prediction = self.linear_2(self.relu(self.linear_1(output_lstm)))
        
        return prediction, hidden


#  Loss weight

ngram_counts_1 = count_ngrams(list_of_file2idx_train, 1)
sorted_ngram_counts_1 = {key: ngram_counts_1[key] for key in sorted(ngram_counts_1)}

loss_weights_array = [0] * len(vocab)

for key, value in sorted_ngram_counts_1.items():
    loss_weights_array[key] = value/sum(sorted_ngram_counts_1.values())

# print(loss_weights_array)
# print(len(loss_weights_array))

loss_weights_array = torch.tensor(loss_weights_array).to(device)

#  Perplexity for LSTM

def perplexity_for_LSTM(pred, lab):
    loss_fn = nn.CrossEntropyLoss(ignore_index=384, weight=loss_weights_array, reduction='none')
    loss = loss_fn(pred.view(-1, pred.shape[-1]), lab.view(-1).to(torch.long).to(device))
    pad_count = (lab[0] == 384).sum().item()
    perplexity = np.exp((sum(loss).item())/(len(lab[0]) - pad_count + 0.0001))
    return perplexity


#  Training

# Initialize some variables for keeping track of the best model
best_model = None
best_perplexity = float('inf')
best_learning_rate = None
best_epoch = None

for num_layer in num_layers:

    for learning_rate in learning_rates:

        best_overall_perplexity = np.inf  # Initialize the best LAS as the worst possible score
        best_epoch = None

        # Initializing the model
        model = LSTM(vocab_size, embedding_dim, hidden_lstm_dim, hidden_dim, num_layer).to(device)

        
        

        # Using cross entropy loss for loss computation
        loss_fn = nn.CrossEntropyLoss(ignore_index=384, weight=loss_weights_array)

        # Using Adam optimizer for optimization
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for ep in range(1, max_eps+1):
            print(f"Epoch {ep}")
            train_loss = []
            
            # for inp, lab in train_loader:
            #     hidden = model.init_hidden(inp.size(0), device)
            #     print(inp.size(0)) 
            # print(hidden[0].size())
            # print(hidden[1].size())     
            
            for inp, lab in train_loader:
                hidden = model.init_hidden(inp.size(0), device)
                model.train()
                optimizer.zero_grad()
                pred, hid = model(inp.to(device), hidden)     #Forward pass
                # outt = torch.tensor(out)
                # print(inp.size())
                # print(pred.size())
                # print(lab.size())
                loss = loss_fn(pred.view(-1, pred.shape[-1]), lab.view(-1).to(torch.long).to(device))

                # print(f"Loss: {loss}")   # with shape {loss.shape}
                # print(loss.size())
                # print(sum(loss).item())
                # print(loss.mean().item())
                # print(lab.view(-1).size())

                loss.backward() # computing the gradients
                optimizer.step()  # Performs the optimization

                train_loss.append(loss.item())    # Appending the batch loss to the list

            average_train_loss = np.mean(train_loss)
            print(f"Average training batch loss for Epoch {ep}: {average_train_loss}")


            # Check if this is the best model so far based on validation loss
            val_perplexity = []
            model.eval()  # Switch to evaluation mode
            with torch.no_grad():
                for inp, lab in dev_loader:
                    # print(lab.size())
                    # print(lab)
                    # print(len(lab[0]))
                    # count = (lab[0] == 384).sum().item()
                    # print(count)
                    hidden = model.init_hidden(inp.size(0), device)
                    pred, hid = model(inp.to(device), hidden)
                    perplexity = perplexity_for_LSTM(pred, lab)
                    val_perplexity.append(perplexity)

            average_val_perplexity = np.mean(val_perplexity)
            print(f"Average validation perplexity for Epoch {ep}: {average_val_perplexity}")


            # Save the model if it's the best one so far for this learning rate
            # Check if the current model is the best based on validation loss
            if average_val_perplexity < best_perplexity:
                best_perplexity = average_val_perplexity
                best_model = model.state_dict()
                best_learning_rate = learning_rate
                best_epoch = ep
                
                
    # Save the best model so far
    torch.save(best_model, f'best_model_numlayer_{num_layer}_epoch_{ep}_lr_{learning_rate}.pth')         

    # After the training loop completes, report the best learning rate and lowest development set loss
    print(f"Results for {num_layer} layer LSTM Network")
    print(f"Best Learning Rate: {best_learning_rate}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Lowest Development Set Perplexity: {best_perplexity}")
    num_param = sum(p.numel() for p in model.parameters())
    print(f"Nuber of parameters: {num_param}")


    #  Calculate the perplexity for the test set
    test_perplexity = []
    with torch.no_grad():
        for inp, lab in test_loader:
            hidden = model.init_hidden(inp.size(0), device)
            pred, hid = model(inp.to(device), hidden)
            # print(pred.size())
            # print(pred)
            perplexity = perplexity_for_LSTM(pred, lab)
            test_perplexity.append(perplexity)
    
    average_test_perplexity = np.mean(test_perplexity)
    print(f"Average test perplexity for {num_layer} layer LSTM Network: {average_test_perplexity}")