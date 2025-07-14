import tensorflow_datasets as tfds
import tensorflow as tf
from transformers import PreTrainedTokenizerFast

# Load tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="imdb_bpe_tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]"
)

# Load IMDB dataset
raw_train, raw_val, raw_test = tfds.load(
    'imdb_reviews',
    split=['train[:90%]', 'train[90%:]', 'test'],
    as_supervised=True
)

# Tokenize function
def encode(text_tensor, label):
    encoded = tokenizer.encode_plus(
        text_tensor.numpy().decode('utf-8'),
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='np'
    )
    return encoded['input_ids'][0], label

# Wrap for tf.data
def tf_encode(text, label):
    result = tf.py_function(encode, inp=[text, label], Tout=(tf.int32, tf.int64))
    result[0].set_shape([256])
    result[1].set_shape([])
    return result

# Prepare datasets
train_ds = raw_train.map(tf_encode).shuffle(5000).batch(32).prefetch(1)
val_ds = raw_val.map(tf_encode).batch(32).prefetch(1)
test_ds = raw_test.map(tf_encode).batch(32).prefetch(1)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=128, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(train_ds, validation_data=val_ds, epochs=3)

# Evaluate
print(model.evaluate(test_ds))
