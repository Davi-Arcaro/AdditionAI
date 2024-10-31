#Example following the page https://keras.io/examples/nlp/addition_rnn/
#Train a model that learns to add strings of numbers, e.g '535+61' -> '596'

import keras
from keras import layers
import numpy as np

#Parameters for the model and dataset
training_size = 50000
digits = 3
reverse = True

#Max length of input is 'int + int'. Max length of int is variable digits
maxlen = digits + 1 + digits

#Generating the data
class CharacterTable:
    #Given a set of characters:
    #Encode them to a one-hot integer representation;
    #Decode the one-hot or integer representation to their character output;
    #Decode a vector of probabilities to their character output

    def __init__(self, chars):
        #initialize character table
        #Arguments -> chars: Characters that can appear in the input
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars)) 
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, C, num_rows):
        #One-hot encode given string C
        #Arguments -> C: string, to be encoded; num_rows: Number of rows in the returned one-hot encoding. This is used to keep the # of rows for each data the same.
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
            return x
    
    def decode(self, x, calc_argmax= True):
        #Decode the given vector or 2D array to their character output;
        #Arguments -> x: A vector or a 2D array of probabilities or one-hot representations or a vector of character indices (used with 'calc_argmax=False')
        #calc_argmax: Whether to find the character index with maximum probability, defaults to 'True'
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

#All the numbers, plus sign and space for padding
chars = '0123456789+ '
ctable = CharacterTable(chars)

questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < training_size:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, digits + 1))))
    a, b = f(), f()
    #Skip any addition questions we've already seen, also skip any such that x+Y == Y+x (hence the sorting)
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    #Pad the data with spaces such that it is always maxlen
    q = '{} + {}'.format(a, b)
    query = q + ' ' * (maxlen - len(q))
    ans = str(a + b)
    #Answers can be of maximum size digits + 1
    ans += ' ' * (digits + 1 - len(ans))
    if reverse:
        # reverse the query, e.g, '12+345' becomes '543+21'. (Note the space used for padding)
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total questions', len(questions))

#Vectorize the data
print('Vectorization...')
x = np.zeros((len(questions), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(questions), digits + 1, len(chars)), dtype=bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, digits + 1)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, digits + 1)

#Shuffle (x,y) in unison as the later parts of x will almost all be larger digits
indices = np.arrange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

#Explicitly set apart 10% for validation data that we never train over
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data: ')
print(x_train.shape)
print(y_train.shape)

print('Validation Data: ')
print(x_val.shape)
print(y_val.shape)


#Building the model
print('Build model...')
num_layers = 1

model =keras.Sequential()
#'Encode' the input sequence using LSTM, producing an output of size 128.
model.add(layers.Input((maxlen, len(chars))))
model.add(layers.LSTM(128))
model.add(layers.LSTM(64))
model.add(layers.LSTM(32))
model.add(layers.LSTM(16))
model.add(layers.LSTM(8))
#As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 'digits + 1' times as that's the maximum length of output.

model.add(layers.RepeatVector(digits + 1))
#The decoder RNN could be multiple layers stacked or a single layer
for _ in range(num_layers):
    #By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim).
    #This is necessary as TimeDistributed in the below expects the first dimension to be the timesteps.
    model.add(layers.LSTM(128, return_sequences=True))

#Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen
model.add(layers.Dense(len(chars), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='[accuracy]')
model.summary()

#Training the model
epochs = 30
batch_size = 32

#Formatting characters for results display
green_color = '\033[92m'
red_color = '\033[91m'
end_char = '\033[0m'

#Train the model each generation and show predictions against the validation dataset
for epoch in range(1, epochs):
    print()
    print('Iteration', epoch)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_val, y_val))

    #Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind]), y_val[np.array([ind])]]
        preds = np.argmax(model.predict(rowx, verbose=0), axis=-1)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if reverse else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(f'{green_color} {guess}{end_char}')
        else:
            print(f'{red_color} {guess}{end_char}')
