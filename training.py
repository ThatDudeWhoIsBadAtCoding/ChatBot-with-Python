import json
import numpy as np
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import load_model

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)


lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
words = []
classes = []
docs = []
ignore = ["?", ".", "!", ","]

for intent in intents['intents']:
    for pattern in intent['pattern']:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        docs.append((word, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower())
         for word in words if word not in ignore]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
Empty_output = [0] * len(classes)
for doc in docs:
    bag = []
    word_pattern = doc[0]
    word_pattern = [lemmatizer.lemmatize(
        word.lower()) for word in word_pattern]
    for word in words:
        bag.append(1) if word in word_pattern else bag.append(0)

    Row_output = list(Empty_output)
    Row_output[classes.index(doc[1])] = 1
    training.append([bag, Row_output])

random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
                   optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=300, batch_size=5, verbose=1)
model.save("chatbot_model_hehe.h5", hist)
print("Done m8")
