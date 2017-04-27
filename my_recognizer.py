import warnings
from asl_data import SinglesData
import re # Regular Expression library


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # implement the recognizer
    words = test_set.get_all_Xlengths()

    for i in words:
        word = words[i][0]
        probability = {}
        for key in models:
            model = models[key]
            try:
                probability[key] = model.score(word)
            except:
                probability[key] = 0

        probabilities.append(probability)
        guess = max(probability, key=probability.get)
        guess = re.sub('\d', '', guess) # Remove digits
        guesses.append(guess)

    # return probabilities, guesses
    return probabilities, guesses
