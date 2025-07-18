import tensorflow_datasets as tfds
import tensorflow as tf

raw_train_set,raw_valid_set,raw_test_set = tfds.load(
    name='imdb_reviews',
    split=['train[:90%]','train[90%:]','test'],
    as_supervised=True
)

tf.random.set_seed(42)
train_set = raw_train_set.shuffle(5000,seed=42).batch(32).prefetch(1)
valid_set = raw_valid_set.batch(32).prefetch(1)
test_set = raw_test_set.batch(32).prefetch(1)

for review,label in raw_train_set.take(4):
    print(review.numpy().decode('utf-8'))
    print("label:",label.numpy())

vocab_size=1000
text_vec_layer = tf.keras.layers.TextVectorization(max_tokens=vocab_size)
text_vec_layer.adapt(train_set.map(lambda reviews,labels: reviews))
embed_size=128

tf.random.set_seed(42)
model=tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Embedding(vocab_size,embed_size,mask_zero=True),
    tf.keras.layers.GRU(128),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer="nadam",metrics=['accuracy'])
history=model.fit(train_set,validation_data= valid_set,epochs=2)
print(model.evaluate(test_set))