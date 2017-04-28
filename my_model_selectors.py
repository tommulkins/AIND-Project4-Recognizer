import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on BIC scores
        X, lengths = self.hwords[self.this_word]
        max_score = 0
        num_components = self.min_n_components

        for i in range(self.min_n_components, self.max_n_components):
            try:
                model = GaussianHMM(n_components=i, n_iter=1000).fit(X, lengths)
                logL = model.score(X, lengths)
                p = i ** 2 + 2 * i * len(X[0]) - 1
                bic_score = -2 * logL + p * math.log(len(X))
                if bic_score < max_score or max_score == 0:
                    max_score = bic_score
                    num_components = i
            except ValueError:
                pass

        return self.base_model(num_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on DIC scores
        X, lengths = self.hwords[self.this_word]
        max_score = 0
        num_components = self.min_n_components

        scores = {}
        antiRes = {}

        for i in range(self.min_n_components, self.max_n_components):
            antiLogL = 0.0
            wc = 0

            try:
                model = GaussianHMM(n_components=i, n_iter=1000).fit(X, lengths)
                for word in self.hwords:
                    if word == self.this_word:
                        continue
                    antiLog_X, antiLog_lengths = self.hwords[word]
                    antiLogL += model.score(antiLog_X, antiLog_lengths)
                    wc += 1
                scores[i] = model.score(X, lengths)
                antiLogL /= float(wc)
                antiRes[i] = antiLogL

                dic_score = scores[i] - antiRes[i]

                if (dic_score > max_score or max_score == 0):
                    max_score = dic_score
                    num_components = i

            except ValueError:
                pass

        return self.base_model(num_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection using CV
        try:
            best_score = float("Inf")
            best_model = None

            for i in range(self.min_n_components, self.max_n_components + 1):
                split_method = KFold(n_splits=2)

                model = self.base_model(i)
                scores = []

                for train_i, test_i in split_method.split(self.sequences):
                    self.X, self.lengths = combine_sequences(train_i, self.sequences)
                    X, lengths = combine_sequences(test_i, self.sequences)
                    scores.append(model.score(X, lengths))

                mean_score = np.mean(scores)
                if mean_score < best_score:
                    best_score = mean_score
                    best_model = model

            return best_model
        except:
            return self.base_model(self.n_constant)

