# from collections import Counter
import math
import numpy as np


def add_padding_to_sequences(sequences, n_gram, vocab):
    padded_sequences = []
    for sequence in sequences:
        padding_count = n_gram - 1
        padding = [vocab["[PAD]"]] * padding_count
        padded_sequence = padding + sequence
        padded_sequences.append(padded_sequence)
    return padded_sequences




def count_ngrams(list_of_lists, n):
    ngram_counts = {}

    for sentence in list_of_lists:
        for i in range(len(sentence) - n + 1):
            if n == 1:  # Handle 1-grams differently
                ngram = sentence[i]
            else:
                ngram = tuple(sentence[i:i + n])
            if ngram in ngram_counts:
                ngram_counts[ngram] += 1
            else:
                ngram_counts[ngram] = 1

    return ngram_counts




def calculate_perplexity_score(data, ngram_counts_3, ngram_counts_4, vocab):
    counts_3grams = []
    counts_4grams = []

    for sentence in data:
        ngrams_3 = [tuple(sentence[i:i + 3]) for i in range(len(sentence) - 3 + 1)]
        ngrams_4 = [tuple(sentence[i:i + 4]) for i in range(len(sentence) - 4 + 1)]

        counts_3 = [ngram_counts_3.get(ngram, 0) for ngram in ngrams_3[:-1]]  # Discard the last element
        counts_4 = [ngram_counts_4.get(ngram, 0) for ngram in ngrams_4]

        counts_3grams.append(counts_3)
        counts_4grams.append(counts_4)

    # Element-wise division
    counts_ratios = []
    for counts_3, counts_4 in zip(counts_3grams, counts_4grams):
        ratio = [(counts4 + 1) / (counts3 + len(vocab)) for counts3, counts4 in zip(counts_3, counts_4)]
        counts_ratios.append(ratio)

    # Element-wise logarithm (base 2)
    counts_logs = []
    for ratios in counts_ratios:
        logs = [np.log2(ratio) for ratio in ratios]
        counts_logs.append(logs)

    # Sum the values for each inner list
    sums = [sum(logs) for logs in counts_logs]

    # Divide the sums by the lengths of the original inner lists
    sums_d_n = [-(s / (len(sentence)-3)) for s, sentence in zip(sums, data)]

    # Calculate perplexity
    perplexity_score_per_sentence = [2 ** x for x in sums_d_n]

    # Calculate the mean of the sums
    avg_perplexity_score = sum(perplexity_score_per_sentence) / len(perplexity_score_per_sentence)

    

    # return counts_3grams, counts_4grams, counts_ratios, counts_logs, sums, sums_d_n, sum_of_probabilities, perplexity_score
    return avg_perplexity_score



def get_input_and_target_subsequences(list_of_sequences, vocab, subsequence_length):
    """ Converts a List[List[int]] we get using `convert_files2idx` into
        a list of subsequence characters. Length of each subsequence is
        determined by the `subsequence_length` parameter.
    Input
    ------------
    List[List[int]]: A list of lists where each inner list is a list of indexes
    vocab: dict. A dictionary mapping characters to unique indices
    subsequence_length: int. The length of each subsequence.

    Output
    -------------
    input_subsequences: List[List[int]]. List of subsequence characters with a fixed length. 
    Appropriate padding is added to the end of each subsequence.

    target_subsequences: List[List[int]]. List of subsequence characters for target with a fixed length. 
    Appropriate padding is added to the end of each subsequence. Basically the same as input_subsequences just shifted by 1 character to the left.
    """

    input_subsequences = []
    target_subsequences = []

    for sequence in list_of_sequences:
        for i in range(0, len(sequence), subsequence_length):
            input_subsequence = sequence[i:i + subsequence_length]
            target_subsequence = sequence[i + 1:i + 1 + subsequence_length]
            # subsequences.append(subsequence)
            input_subsequence_data = []
            target_subsequence_data = []
            
            
            for ind in input_subsequence:
                input_subsequence_data.append(ind)

            while len(input_subsequence_data) < subsequence_length:
                input_subsequence_data.append(vocab["[PAD]"])

            input_subsequences.append(input_subsequence_data)

            
            for ind in target_subsequence:
                target_subsequence_data.append(ind)

            while len(target_subsequence_data) < subsequence_length:
                target_subsequence_data.append(vocab["[PAD]"])
            
            target_subsequences.append(target_subsequence_data)

    return input_subsequences, target_subsequences





def convert_idx2line(line_data, reversed_vocab):
    """ Converts a string into a list of character indices
    Input
    ------------
    line: str. A line worth of data
    vocab: dict. A dictionary mapping characters to unique indices

    Output
    -------------
    line_data: List[int]. List of indices corresponding to the characters
                in the input line.
    """
    line = ""
    for idx in line_data:
        line += reversed_vocab[idx]
    return line






"""
Code Graveyard
"""

# import glob

# def process_line(line, vocab, subsequence_length):
#     """ Converts a string into a list of subsequence characters
#     Input
#     ------------
#     line: str. A line worth of data
#     vocab: dict. A dictionary mapping characters to unique indices
#     subsequence_length: int. The length of each subsequence.

#     Output
#     -------------
#     line_data: List[List[str]]. List of subsequence characters with a fixed length.
#     """
#     line_data = []
#     for i in range(0, len(line), subsequence_length):
#         subsequence = line[i:i + subsequence_length]
#         subsequence_data = []
#         for charac in subsequence:
#             if charac not in vocab:
#                 # subsequence_data.append(vocab["<unk>"])
#                 subsequence_data.append("<unk>")
#             else:
#                 # subsequence_data.append(vocab[charac])
#                 subsequence_data.append(charac)
#         while len(subsequence_data) < subsequence_length:
#             # subsequence_data.append(vocab["[PAD]"])
#             subsequence_data.append("[PAD]")
#         line_data.append(subsequence_data)
#     return line_data




# def generate_subsequence_pairs_input_target(line, vocab, subsequence_length):
    
#     input_data = process_line(line, vocab, subsequence_length)

#     line = line[1:]
#     target_data = process_line(line, vocab, subsequence_length)

#     return input_data, target_data




# def process_files(files, vocab, subsequence_length):

#     """ This method iterates over files. In each file, it iterates over
#     every line. Every line is then split into characters and the characters are 
#     converted to their respective unique indices based on the vocab mapping. All
#     converted lines are added to a central list containing the mapped data.
#     Input
#     --------------
#     files: List[str]. List of files in a particular split
#     vocab: dict. A dictionary mapping characters to unique indices

#     Output
#     ---------------
#     data: List[List[int]]. List of lists where each inner list is a list of character
#             indices corresponding to a line in the training split.
#     """
#     data = []

#     for file in files:
#         with open(file) as f:
#             lines = f.readlines()
        
#         for line in lines:
#             toks = process_line(line, vocab, subsequence_length)
#             # toks = generate_subsequence_pairs_input_target(line, vocab, subsequence_length)
#             data.append(toks)

#     return data