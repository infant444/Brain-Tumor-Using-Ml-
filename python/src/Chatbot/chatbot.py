import random
import json
import torch as tr
from src.Chatbot.model import NeuralNetwork
from src.Chatbot.nltk_utils import back_of_word,tokenize
# from model import NeuralNetwork
# from nltk_utils import back_of_word,tokenize
def modelload():
    device = tr.device('cude' if tr.cuda.is_available() else 'cpu')

    with open('Chatbot\intents.json', 'r') as f:
        intents = json.load(f)

    TRANI_fILE = 'Chatbot\data.pth'

    train_data = tr.load(TRANI_fILE)

    input_size = train_data['input_size']
    output_size = train_data['output_size']
    hidden_size = train_data['hidden_size']
    tags = train_data['tags']
    all_words = train_data['all_words']
    model_state = train_data['model_state']
    model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model,all_words,device,tags,intents

def process(sentence):
    model, all_words, device, tags,intents=modelload();
    sentence = tokenize(sentence)
    x = back_of_word(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = tr.from_numpy(x).to(device)

    output = model(x)
    _, predicted = tr.max(output, dim=1)
    tag = tags[predicted.item()]
    # print(tag)

    for intent in intents['intents']:
        if tag == intent['tag']:
            return random.choice(intent['responses'])




if __name__=='__main__':
    bot_name = 'Joe'

    print("Let's chat! type 'quit' to exit")

    while True:
        sentence = input("You: ")
        if sentence == 'quit':
            break
        print(f"{bot_name}: {process(sentence)}")