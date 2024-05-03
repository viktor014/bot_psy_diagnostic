import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle


def Predictor1(sen):
    model = load_model('Models/text/model2.h5')
    with open('Models/text/tokenizer2.pickle', 'rb') as handle:
      tokenizer = pickle.load(handle)
    sen = tokenizer.texts_to_sequences(sen)
    sen = pad_sequences(sen, maxlen=96, dtype='int32', value=0)
    sentiment = model.predict(sen, batch_size=1, verbose=0)[0]
    sentiment = [float(i) / sum(sentiment) for i in sentiment]
    final_prediction = ((sentiment[0] * 100) ** 1.3 + (sentiment[2] * 100) ** 1.5) / 10
    prediction_print = "{:.2f}".format(final_prediction)
    return prediction_print

# позитивные
# sen=["i feel so good while talking to you"]
# print(Predictor1(sen))

#sen = ["i am playing games"]
#sen=["smiling through it all can not believe this is my life"]
# sen=["I had a black coffee last night to complete the assignments!"]
# sen = ["The sun's warm embrace enveloped the world, casting a golden glow upon fields of blooming flowers, as laughter filled the air, echoing the joyous spirit of the moment."]
# грустные
# sen = ["i commit suicide"]
#sen = ["i am discouraged and feeling lonely"]
# sen = ["i do not like where my life is going I feel hopeless"]
#sen = ["I do not want to be here anymore the only reason I stop myself from committing suicide is my parents It is not fair It was not my choice to come into this shit world now it is not my choice to leave"]
#sen = ["In the depths of solitude, the echoes of forgotten dreams whisper, reminding me of the relentless passage of time and the crushing weight of missed opportunities."]

