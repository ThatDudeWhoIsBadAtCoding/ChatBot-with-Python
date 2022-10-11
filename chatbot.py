from os import system
import random
import json
import pickle
import nltk
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model('chatbot_model_hehe.h5')


def cleanup(sentence):
    words_inside = nltk.word_tokenize(sentence)
    words_inside = [lemmatizer.lemmatize(word) for word in words_inside]
    return words_inside


def bagofdemwords(sentence):
    words_inside = cleanup(sentence)
    bag = [0] * len(words)
    for w in words_inside:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predictclass(sentence):
    bow = bagofdemwords(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_LINE = 0.1
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_LINE]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append(
            {"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getdatresponse(intents_list, intents_json):
    tag = intents_list[0]['intent']
    listofthoseintents = intents_json['intents']
    for i in listofthoseintents:
        if i['tag'] == tag:
            resp = random.choice(i["response"])
            break
    return resp


# system("cls")
# print("HERE WE GOOOOOOOOOOOOOOOOOOO!")

# while True:
#     msg = input("You: ")
#     ints = predictclass(msg.lower())
#     res = getdatresponse(ints, intents)
#     print(F"Bot: {res}")
