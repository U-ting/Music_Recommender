import os
import soundfile as sf
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from python_speech_features import logfbank
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, GRU
from keras.models import Model
import keras as K


VALIDATION_SPLIT = 0.2

# read data
musics = {}
folder = "data/converted-7200"
for file in os.listdir(folder):
    # print(file)
    filepath = os.path.join(folder, file)
    samples, sample_rate = sf.read(filepath)
    # print(file)
    file = file.split(' - ')[1]
    musics[file] = samples

# print(musics)

# read labels
labels_list = []
with open('all_final.txt', encoding='utf-8') as file:
    labels = file.readlines()

for label in labels:
    # print(label)
    label_split = label.split('，')
    label_split[1] = label_split[1][1:]
    labels_list.append(label_split)

train_data = []
i = 0
for l in labels_list:
    # print(l)
    try:
        if int(l[-1]) == 0 or int(l[-1]) == 1:
            train_data.append([l[0], musics[l[0]+'.wav'], 'dislike'])
        if int(l[-1]) == 2 or int(l[-1]) == 3:
            pass
            # train_data.append([l[0], musics[l[0]+'.wav'], 'normal'])
        if int(l[-1]) == 4 or int(l[-1]) == 5:
            train_data.append([l[0], musics[l[0]+'.wav'], 'like'])
    except KeyError:
        i = i + 1
        print(l[0]+'.wav')
        pass

print(train_data)
print(i)


# extract features from musics
musics = list()
labels = list()

processed_musics = list()
train_label = list()
for data in train_data:
    musics.append(data[1])
    labels.append(data[2])

count = 0
for music, label in zip(musics, labels):
    fbank_feat = logfbank(music, sample_rate, nfft=1200)
    print(fbank_feat.shape)
    processed_musics.append(fbank_feat[:5000])
    if label == 'like':
        train_label.append(0)
    # elif label == 'normal':
    #     train_label.append(1)
    else:
        train_label.append(1)

    if count % 10 == 0:
        print('processed ', count, ' musics.')
    count += 1

print(train_label)
# # save processed music
# with open('data/processed_music.txt', 'w') as file:
#     for music in processed_musics:
#         file.write(music + '/n')

data = np.array(processed_musics)
print(data.shape, 'data shape')
labels = to_categorical(np.asarray(train_label))
print(labels)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
print(x_train.shape)
y_train = labels[:-num_validation_samples]
print(y_train.shape)
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

sequence_input = Input(shape=(5000, 26), dtype='float32')
x = GRU(units=128)(sequence_input)
preds = Dense(2, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=1000,
          validation_data=(x_val, y_val))
