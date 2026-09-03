import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GRU, Dense
import dataset 


dataset_loader = dataset(name="sadeed_tashkeal", batch_size=32)
train_dataset = dataset_loader.load_dataset(split="train",batch_size=32, shuffle=True)
validation_dataset = dataset_loader.load_dataset(split="test",batch_size=32)
tiny_train_dataset = train_dataset.take(5)

# 1. Build Model Architecture (Stacked BiLSTM / BiGRU)
model = Sequential([
    # Embedding layer with masking support to ignore padding tokens (index 0)
    Embedding(
        input_dim=50, 
        output_dim=128, 
        mask_zero=True
    ),

    # First recurrent layer (Bidirectional LSTM)
    Bidirectional(
        LSTM(
            128, 
            return_sequences=True
        )
    ),

    # Second recurrent layer (Bidirectional GRU)
    Bidirectional(
        GRU(
            128, 
            return_sequences=True
        )
    ),

    # Output layer: Predict probability distribution across diacritic classes per character
    Dense(
        16, 
        activation="softmax"
    )
])

# 2. Configure training parameters
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 3. Sanity check run on a small slice of data
model.fit(
    tiny_train_dataset,
    epochs=1
)

# 4. Full training run with validation data
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)