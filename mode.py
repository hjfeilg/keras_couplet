from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import os
import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

batch_size = 128  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
# Path to the data txt file on disk.

voc = {}
with open('data/vocabs', encoding='utf-8')as vocab:
    for index, line in enumerate(vocab.readlines()):
        voc[line.strip()] = index

# print(voc)

input_token = []
target_token = []
input_token_index = []
target_token_index = []

max_encoder_seq_length = 7
max_decoder_seq_length = 9

import threading


class createBatchGenerator:

    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            data = pd.read_csv('data/data.csv')
            count = 0
            for line in data.values:
                X1 = []
                X2 = []
                y = []
                count += 1
                encoder_input_data = np.zeros((max_encoder_seq_length, len(voc)), dtype='float16')
                decoder_input_data = np.zeros((max_decoder_seq_length, len(voc)), dtype='float16')
                decoder_target_data = np.zeros((max_decoder_seq_length, len(voc)), dtype='float16')
                for t, char in enumerate(str(line[0]).split()):
                    encoder_input_data[t, voc[char]] = 1
                for t, char in enumerate(str('<s> ' + line[1] + ' </s>').split()):
                    decoder_input_data[t, voc[char]] = 1
                    if t > 0:
                        decoder_target_data[t - 1, voc[char]] = 1

                X1.append(encoder_input_data)
                X2.append(decoder_input_data)
                y.append(decoder_target_data)

                if count == self.batch_size:
                    count = 0
                    # yield (np.array(X1), np.array(X2),np.array(y))
                    yield ({'input_1': np.array(X1), 'input_2': np.array(X2)}, {'dense_1': np.array(y)})


def get_data(batch_size=128):
    while True:
        data = pd.read_csv('data/data.csv')
        count = 0
        for line in data.values:
            X1 = []
            X2 = []
            y = []
            count += 1
            encoder_input_data = np.zeros((max_encoder_seq_length, len(voc)), dtype='float16')
            decoder_input_data = np.zeros((max_decoder_seq_length, len(voc)), dtype='float16')
            decoder_target_data = np.zeros((max_decoder_seq_length, len(voc)), dtype='float16')
            for t, char in enumerate(str(line[0]).split()):
                encoder_input_data[t, voc[char]] = 1
            for t, char in enumerate(str('<EOS> ' + line[1] + ' </EOS>').split()):
                decoder_input_data[t, voc[char]] = 1
                if t > 0:
                    decoder_target_data[t - 1, voc[char]] = 1

            X1.append(encoder_input_data)
            X2.append(decoder_input_data)
            y.append(decoder_target_data)

            if count == batch_size:
                count = 0
                yield ({'input_1': np.array(X1), 'input_2': np.array(X2)}, {'dense_1': np.array(y)})


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, len(voc)))
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, len(voc)))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(len(voc), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training

print('model train')
# filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=5, save_best_only=True,
#                              mode='max')
# callbacks_list = [checkpoint]
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
# [encoder_input_data, decoder_input_data], decoder_target_data,
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit_generator(get_data(128),
                    steps_per_epoch=1024,
                    epochs=500)

# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

encoder_model.save('encoder_model.h5')
decoder_model.save('decoder_model.h5')
print('save model done')


# encoder_model = load_model('encoder_model.h5')
# decoder_model = load_model('decoder_model.h5')

# Reverse-lookup token index to decode sequences back to
# something readable.
# reverse_input_char_index = dict(
#     (i, char) for char, i in voc.items())
# reverse_target_char_index = dict(
#     (i, char) for char, i in voc.items())


def decode_sequence(input_seq, seq_len):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, len(voc)))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, voc['<s>']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if len(decoded_sentence) >= seq_len:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, len(voc)))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

# if __name__ == '__main__':
#     while True:
#         sentence = input('sentence: ')
#
#         encoder_input_data = np.zeros((1, 32, len(voc)), dtype='float32')
#         for index, word in enumerate(sentence):
#             encoder_input_data[:, index, voc[word]] = 1
#
#         decoded_sentence = decode_sequence(encoder_input_data, seq_len=len(sentence))
#         print('Input sentence:', sentence)
#         print('Decoded sentence:', decoded_sentence)
# #
# for seq_index in range(100):
#     # Take one sequence (part of the training set)
#     # for trying out decoding.
#     input_seq = encoder_input_data[seq_index: seq_index + 1]
#     decoded_sentence = decode_sequence(input_seq)
#     print('-')
#     print('Input sentence:', input_token[seq_index])
#     print('Decoded sentence:', decoded_sentence)
