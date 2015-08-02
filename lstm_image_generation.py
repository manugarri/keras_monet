from __future__ import print_function
import os
import scipy
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np
import random


from color_quantization import quantize

'''
    Example script to generate text from Nietzsche's writings.

    At least 20 epochs are required before the generated text
    starts sounding coherent.

    It is recommended to run this script on GPU, as recurrent
    networks are quite computationally intensive.

    If you try this script on new data, make sure your corpus 
    has at least ~100k characters. ~1M is better.
'''

def rgb_to_hex(array):
    array = map(lambda x: int(x * 255), array)
    return '{:02x}{:02x}{:02x}'.format(*array)

def hex_to_rgb(*array):
    rgb_array = []
    for value in array:
        value = value.lstrip('#')
        lv = len(value)
        rgb_value = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        rgb_array.append(rgb_value)
    rgb_array = np.array(rgb_array)
    return rgb_array

h2rgb = np.vectorize(hex_to_rgb)

def hex_image_to_rgb(image_data):
    image_data = image_data.tolist()
    for i, row in enumerate(image_data):
        image_data[i] = h2rgb(image_data[i])[0]
    return np.array(image_data)

input_folder = 'monet'
image_width = image_height = 75
input_layer_size = image_width*image_height
input_images = os.listdir(input_folder)
n_samples = len(input_images)
n_colors = 1000

text = np.empty((0,0))
text = []

print('Reading and Quantizing input images')
for input_image in input_images:
    print('loading image {}'.format(input_image))
    image_data = scipy.misc.imread(os.path.join(input_folder,input_image))
    image_data = scipy.misc.imresize(image_data, (image_height, image_width))
    image_data = image_data.reshape(input_layer_size,3)
    text.append(image_data)
text = np.array(text)
text = quantize(text, n_colors)
import pdb;pdb.set_trace()
text = np.apply_along_axis(rgb_to_hex, 2, text)
text = text.reshape(n_samples*input_layer_size)

chars = np.unique(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = image_width
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype='uint8')
y = np.zeros((len(sentences), len(chars)), dtype='uint8')
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1.
    y[i, char_indices[next_chars[i]]] = 1.

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(len(chars), 512, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512, 512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(512, len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# helper function to sample an index from a probability array
def sample(a, diversity=0.75):
    if random.random() > diversity:
        return np.argmax(a)
    while 1:
        i = random.randint(0, len(a)-1)
        if a[i] > random.random():
            return i

# train the model, output generated text after each iteration
main_iteration = 0
while 1:
    main_iteration +=1
    print()
    print('-' * 50)
    print('Iteration', main_iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.4, 0.6, 0.8]:
        print()
        print('----- diversity:', diversity)

        generated = []
        sentence = text[start_index : start_index + maxlen]
        generated.extend(sentence)
        pred_image = []
        for iteration in range(input_layer_size):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence  = np.append(sentence[1:], next_char) 
            pred_image.append(next_char)
        pred_image = np.array(pred_image)
        pred_image = hex_image_to_rgb(pred_image)
        pred_image = pred_image.reshape(image_width, image_height, 3) 
        pred_image_filename =  'output/{}_{}.png'.format(main_iteration, diversity)
        print('Saving output image as {}'.format(pred_image_filename))
        scipy.misc.imsave(pred_image_filename, pred_image)
