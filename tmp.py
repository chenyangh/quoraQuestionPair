'''
Written by Nawshad
@param1: Clean Tweet File
@param2: Mean embedding Output File
@param3: Word Embedding File
@param4: Word embedding dimension

example: for 4K:
python create_mean_we.py ../Data/4k_test_clean_tweets.csv ../Data/4k_test_mean_embedding.txt ../Data/glove.6B/glove.6B.300d.txt 300

'''

import numpy as np
import os
from operator import add
import csv
import sys

# data_path = "../Data/"
output_file_delimiter = ' '


# this function returns all the tweet tokens in the list of lists.
def create_list_of_tweet_tokens():
    list_of_tweet_tokens = []
    data_loc = sys.argv[1]
    with open(data_loc, 'rt', encoding='latin-1') as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            list_row = []
            # print('processed tweet:', row[0])
            list_row.append(row[0])
            tokens = row[1].split()
            for token in tokens:
                list_row.append(token)

            print('list_row:', list_row)
            list_of_tweet_tokens.append(list_row)
    return list_of_tweet_tokens


def write_mean_embeddings_to_file(list_of_mean_embeddings):
    print('list_of_mean_embeddings:', len(list_of_mean_embeddings))
    embeddingsToFile = open(sys.argv[2], 'w')
    for embeddings in list_of_mean_embeddings:
        row = ''
        for each_embeddings in embeddings:
            row += str(each_embeddings) + output_file_delimiter
        embeddingsToFile.write(row + '\n')
        # print('Written:',row+'\n')


def element_wise_add(list_of_word_embeddings):
    sum_of_embeddings = [sum(x) for x in zip(*list_of_word_embeddings)]
    return sum_of_embeddings


def extract_mean_word_embeddings(list_of_tweet_tokens):
    we_loc = sys.argv[3]

    with open(we_loc, "rt") as lines:
        w2v = {line.split()[0]: [float(v) for v in line.split()[1:]]
               for line in lines}

    # find an arbitrary word and its dimension to initialize list_of_mean_embeddings

    embedding_dim = sys.argv[4]
    list_of_mean_embeddings = []

    for tokens_list in list_of_tweet_tokens:
        list_of_word_embeddings = []
        tokens = tokens_list[1:]
        # print('tokens:',tokens)
        print('Processed tweet:', tokens_list[0])
        for token in tokens:
            word_embedding = []
            if w2v.get(token):
                for item in w2v.get(token):
                    word_embedding.append(item)
                list_of_word_embeddings.append(word_embedding)

        if len(list_of_word_embeddings) == 0:
            print('tokens list which is absent in w2v:', tokens)
            list_of_word_embeddings = [[0] * int(embedding_dim)]

        sum_of_embeddings = element_wise_add(list_of_word_embeddings)
        mean_embeddings = [x / len(list_of_word_embeddings) for x in sum_of_embeddings]
        list_of_mean_embeddings.append(mean_embeddings)

        print('sum_of_embeddings size:', len(sum_of_embeddings))
        print('mean_embeddings size:', len(mean_embeddings))

    # if len(list_of_word_embeddings) == 0:
    # print('Mean embeddings:', list_of_mean_embeddings)

    # print("Size of tweet tokens:", len(tokens_list))
    # print("Size of OOV:", len(tokens_list) - len(list_of_word_embeddings))

    # print('list_of_mean_embeddings:', len(list_of_mean_embeddings))

    return list_of_mean_embeddings


def main():
    write_mean_embeddings_to_file(extract_mean_word_embeddings(create_list_of_tweet_tokens()))


if __name__ == "__main__":
    main()