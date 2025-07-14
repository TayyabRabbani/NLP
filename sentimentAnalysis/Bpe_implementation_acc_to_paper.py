import tensorflow_datasets as tfds
from collections import defaultdict


def get_stats(vocab):
    pairs= defaultdict(int)
    for words,freq in vocab.items():
        symbols = words.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, vocab_in):
    vocab_out = {}
    pair_str = ''.join(pair)
    for word, freq in vocab_in.items():
        symbols = word.split()
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols)-1 and symbols[i] == pair[0] and symbols[i+1] == pair[1]:
                new_symbols.append(pair_str)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_word = ' '.join(new_symbols)
        vocab_out[new_word] = freq
    return vocab_out

def build_bpe_vocab(corpus):
    vocab = defaultdict(int)
    for sentence in corpus:
        sentence = sentence.strip()
        words = sentence.split()
        for word in words:
            chars = ' '.join(list(word)) + ' </w>'
            vocab[chars] += 1
    return dict(vocab)

def main():
    raw_train_set, raw_valid_set, raw_test_set = tfds.load(
        name='imdb_reviews',
        split=['train[:90%]', 'train[90%:]', 'test'],
        as_supervised=True
    )

    corpus = [text.numpy().decode('utf-8') for text, _ in raw_train_set.take(100)]

    # Build BPE-style vocab
    vocab = build_bpe_vocab(corpus)

    n_merges = 1000
    for i in range(n_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    print(best)


if __name__ == '__main__':
    main()
