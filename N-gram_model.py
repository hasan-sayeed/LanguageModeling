import pickle
from scripts.utils import get_files, convert_files2idx
from scripts.utils_mine import add_padding_to_sequences, calculate_perplexity_score, count_ngrams



#  Reading vocabulary from pickle file



with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)



#  Read files and convert to list of idx



train_files = get_files('data/train')
dev_files = get_files('data/dev')
test_files = get_files('data/test')

list_of_file2idx_train = convert_files2idx(train_files, vocab)
list_of_file2idx_dev = convert_files2idx(dev_files, vocab)
list_of_file2idx_test = convert_files2idx(test_files, vocab)



#  N-gram model



padded_data_train = add_padding_to_sequences(list_of_file2idx_train, n_gram = 4, vocab = vocab)

# Count 3-grams
ngram_counts_3 = count_ngrams(padded_data_train, 3)

# Count 4-grams
ngram_counts_4 = count_ngrams(padded_data_train, 4)

# print(ngram_counts_4)



#  Calculate perplexity score



padded_data_test = add_padding_to_sequences(list_of_file2idx_test, n_gram = 4, vocab = vocab)

perplexity = calculate_perplexity_score(padded_data_test, ngram_counts_3, ngram_counts_4, vocab)
print("Test Perplexity Score: ", perplexity)


print(len(ngram_counts_3))
print(len(ngram_counts_4))

print("Number of Parameters: ", len(ngram_counts_3) + len(ngram_counts_4))