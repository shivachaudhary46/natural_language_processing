# Seq2Seq Encoder-Decoder Model: English-to-Japanese Translation

---

## Table of Contents

1. [Introduction: What Is This And Why Should You Care?](#introduction)
2. [What Is a Seq2Seq Model?](#what-is-a-seq2seq-model)
3. [Project Overview](#project-overview)
4. [Dataset: What, Where, and How?](#dataset)
5. [Installation and Environment Setup](#installation)
6. [Data Preparation: Step-by-Step](#data-preparation)
7. [Model Architecture Explained (With Purpose!)](#model-architecture)
    - [Encoder](#encoder)
    - [Decoder](#decoder)
8. [Training the Model](#training)
9. [Inference: Translating New Sentences](#inference)
10. [Usage Example: How To Actually Use The Model](#usage-example)
11. [Troubleshooting & FAQ](#troubleshooting)
12. [Credits & References](#credits)
13. [Summary & Next Steps](#summary)

---

## 1. Introduction

This project implements a **character-level sequence-to-sequence (seq2seq) model** using Keras to translate English sentences into Japanese.  
**Don’t know what seq2seq means?** Read the next section or you’ll be lost.

---

## 2. What Is a Seq2Seq Model?

A **sequence-to-sequence (seq2seq) model** is a neural network architecture that takes a sequence as input (like an English sentence) and produces another sequence as output (like a Japanese translation).  
- **Encoder:** Reads the input sequence and summarizes it into a “state.”
- **Decoder:** Takes that “state” and generates the output sequence, one token at a time.

---

## 3. Project Overview

- **Goal:** Translate English sentences to Japanese at the character level.
- **Framework:** Keras (with TensorFlow backend).
- **Data:** English-Japanese sentence pairs (see below for source & format).

---

## 4. Dataset

- **Source:** [https://www.manythings.org/anki/](https://www.manythings.org/anki/)
- **Format:** Text file with each line:  
  ```
  English[TAB]Japanese
  ```
  Example:
  ```
  Hi.	こんにちは。
  Thanks.	ありがとう。
  ```
- **How to Use:** Download the dataset and extract the file. You’ll read in lines, split by tab, and build your own character vocabularies.

---

## 5. Installation

**Requirements:**
- Python 3.7+
- [TensorFlow](https://www.tensorflow.org/install) (Tested with 2.x)
- Keras (usually included with TensorFlow 2.x)
- NumPy

**Install using pip:**
```bash
pip install tensorflow numpy
```
**Check your Python version and package installations before running anything.**

---

## 6. Data Preparation

1. **Load Data:**
   - Read each line, split by tab.
   - `input_texts`: English sentences.
   - `target_texts`: Japanese sentences, prepended with `\t` (start) and appended with `\n` (end).

2. **Build Character Sets:**
   - `input_characters`: All unique chars in English data.
   - `target_characters`: All unique chars in Japanese data.

3. **Create Char-to-Index Dictionaries:**
   ```python
   input_token_index = {char: i for i, char in enumerate(input_characters)}
   target_token_index = {char: i for i, char in enumerate(target_characters)}
   ```

4. **Vectorize Data as One-Hot:**
   ```python
   encoder_input_data = np.zeros(
       (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
   )
   decoder_input_data = np.zeros(
       (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
   )
   decoder_target_data = np.zeros(
       (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
   )
   ```
   **Why all these zeros?** One-hot encoding: every character is represented as a vector with one “1” and the rest “0”.

---

## 7. Model Architecture

### Encoder

- **Purpose:** Reads the input (English) and summarizes it into a state vector (think of it as “memory” of the sentence).
- **Implementation:**
   ```python
   encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
   encoder = keras.layers.LSTM(latent_dim, return_state=True)
   encoder_outputs, state_h, state_c = encoder(encoder_inputs)
   encoder_states = [state_h, state_c]
   ```
- **Why LSTM?** Handles sequences and remembers information over time.

### Decoder

- **Purpose:** Uses the state vector from the encoder to generate the output (Japanese) one character at a time.
- **Teacher Forcing:** During training, the decoder gets the correct previous character as input (not its own prediction).
- **Implementation:**
   ```python
   decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))
   decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
   decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
   decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
   decoder_outputs = decoder_dense(decoder_outputs)
   ```

---

## 8. Training

```python
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)
model.save("s2s_model.keras")
```
**What’s going on?**  
- Model learns to predict the next character in Japanese given the English input and previous Japanese character.

---

## 9. Inference

**How translation works after training:**
1. Encode the input sentence to get initial states.
2. Start decoding with the special start token (`\t`).
3. Predict the next character, feed it back into the decoder.
4. Repeat until you get the stop token (`\n`) or hit max length.

**Sampling Models:**
```python
model = keras.models.load_model("s2s_model.keras")

# Encoder model
encoder_inputs = model.input[0]
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

# Decoder model
decoder_inputs = model.input[1]
decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_state_input_c = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)
```
**Reverse lookup dictionaries:**  
```python
reverse_input_char_index = {i: char for char, i in input_token_index.items()}
reverse_target_char_index = {i: char for char, i in target_token_index.items()}
```

---

## 10. Usage Example

**How to actually use your trained model to translate a new sentence:**
```python
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]
    return decoded_sentence
```
**Example usage:**
```python
# Assume input_seq is the vectorized English sentence
print(decode_sequence(input_seq))
```

---

## 11. Troubleshooting & FAQ

- **“Model won’t train / Error about shapes / dimensions?”**  
  Double-check your input and target data shapes. Print them out!
- **“Predicted sentences are gibberish?”**  
  Not enough data, not enough epochs, or your model is too small.
- **“Can’t import Keras / TensorFlow?”**  
  Check your installation and Python version (`pip show tensorflow`).
- **“What is teacher forcing?”**  
  During training, the decoder gets the correct previous output as input (not its own guess). [Read more here.](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/)

---

## 12. Credits & References

- [TensorFlow Seq2Seq Example](https://keras.io/examples/nlp/lstm_seq2seq/)
- Dataset: [https://www.manythings.org/anki/](https://www.manythings.org/anki/)
- [Keras Documentation](https://keras.io/)
- [Original Paper: Sequence to Sequence Learning with Neural Networks (Sutskever et al.)](https://arxiv.org/abs/1409.3215)

---

## 13. Summary & Next Steps

- **You now have a working seq2seq model for EN→JA character-level translation.**
- Try using a different dataset, tweaking the architecture, or switching to word-level translation.
- For production: research attention mechanisms and transformer models for far better results.

---

> **If you still don’t understand what’s going on,  
> please reread the sections above, look up the links,  
> or ask someone who actually enjoys reading documentation.**