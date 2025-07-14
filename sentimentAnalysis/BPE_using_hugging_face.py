import tensorflow_datasets as tfds
import tensorflow as tf

raw_train_set,raw_valid_data,raw_test_data = tfds.load(
    name = 'imdb_reviews',
    split = ['train[:90%]', 'train[90%:]', 'test'],
    as_supervised = True,
)

corpus = [text.numpy().decode('utf-8') for text, _ in raw_train_set.take(20000)]

from tokenizers import Tokenizer,models,trainers,pre_tokenizers
from tokenizers.normalizers import Lowercase,NFD,StripAccents,Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=8000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)
tokenizer.train_from_iterator(corpus, trainer=trainer)
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
)
tokenizer.save("imdb_bpe_tokenizer.json")
