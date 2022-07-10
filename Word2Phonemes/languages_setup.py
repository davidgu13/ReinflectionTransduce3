from itertools import chain
from typing import List, Union, Tuple
from Word2Phonemes.g2p_config import idx2feature, feature2idx, p2f_dict, f2p_dict, langs_properties, punctuations

def joinit(iterable, delimiter):
    # Inserts delimiters between elements of some iterable object.
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x

# Foe debugging purposes:
import numpy as np
def editDistance(str1, str2):
    """Simple Levenshtein implementation"""
    table = np.zeros([len(str2) + 1, len(str1) + 1])
    for i in range(1, len(str2) + 1):
        table[i][0] = table[i - 1][0] + 1
    for j in range(1, len(str1) + 1):
        table[0][j] = table[0][j - 1] + 1
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) + 1):
            if str1[j - 1] == str2[i - 1]:
                dg = 0
            else:
                dg = 1
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + dg)
    return int(table[len(str2)][len(str1)])

def tuple_of_phon_tuples2phon_sequence(tupleOfTuples) -> [str]:
    return list(chain(*joinit(tupleOfTuples, ('$',))))


class LanguageSetup:
    """
    Stores the letters and phonemes of the language. Further logic includes converting words to phonemes and vice versa.
    Note: the class is implemented to fit to Georgian, Russian and several Indo-European languages. For languages with more complex phonology,
    this class might need to be extended/be inherited from.
    """
    def __init__(self, lang_name: str, graphemes2phonemes:dict, max_phoneme_size: int,
                 phon_use_attention: bool, manual_word2phonemes=None, manual_phonemes2word=None):
        self._name = lang_name
        self._graphemes2phonemes = graphemes2phonemes
        self._graphemes2phonemes.update(dict(zip(punctuations, punctuations)))
        self._phonemes2graphemes = {v:k for k, v in self._graphemes2phonemes.items()} # _graphemes2phonemes.reverse

        self._alphabet = list(self._graphemes2phonemes.keys()) # the graphemes of the language
        self._phonemes = list(self._graphemes2phonemes.values())

        self._manual_word2phonemes = manual_word2phonemes
        self.manual_phonemes2word = manual_phonemes2word

        self.max_phoneme_size = max_phoneme_size
        self.phon_use_attention = phon_use_attention

    def get_lang_name(self): return self._name
    def get_lang_alphabet(self): return self._alphabet
    def get_lang_phonemes(self): return self._phonemes

    def word2phonemes(self, word:str, mode:str) -> Union[List[List[int]], List[str]]:
        """
        Convert a word (sequence of graphemes) to a list of phoneme tuples.
        :param word: word
        :param mode: can be either 'features' or 'phonemes' (see more above).
        :return: ((,,), (,,), (,,), ...) or list(*IPA symbols*)
        """
        assert mode in {'features', 'phonemes'}, f"Mode {mode} is invalid"

        word = word.casefold() # lower-casing
        graphemes = list(word) # beware of Niqqud-like symbols

        if self._manual_word2phonemes:
            phonemes = self._manual_word2phonemes(graphemes)
        else:
            phonemes = [self._graphemes2phonemes[g] for g in graphemes]

        if mode=='phonemes':
            return phonemes
        else: # mode=='features'
            features=[]
            for p in phonemes:
                feats = [str(feature2idx[e]) for e in p2f_dict[p]]
                feats.extend(['NA']*(self.max_phoneme_size-len(feats)))
                if self.phon_use_attention:
                    feats.append(p)
                features.append(tuple(feats))
            features = tuple_of_phon_tuples2phon_sequence(features)
            return features

    def _phonemes2word(self, phonemes: [[str]], mode:str) -> str:
        """
        Convert a list of phoneme tuples to a word (sequence of graphemes)
        :param phonemes: [(,,), (,,), (,,), ...] or (*IPA symbols*)
        :param mode: can be either 'features' or 'phonemes' (see more above).
        :return: word
        """
        assert mode in {'features', 'phonemes'}, f"Mode {mode} is invalid"

        if mode=='phonemes':
            if self.manual_phonemes2word:
                graphemes = self.manual_phonemes2word(phonemes)
            else:
                graphemes = [self._phonemes2graphemes[p] for p in phonemes]
        else: # mode=='features'
            if self.phon_use_attention:
                phoneme_tokens = [f_tuple[-1] for f_tuple in phonemes]
            else:
                phoneme_tokens = []
                for f_tuple in phonemes:
                    f_tuple = tuple(idx2feature[int(i)] for i in f_tuple if i != 'NA')
                    p = f2p_dict.get(f_tuple)
                    if p is None or p not in self._phonemes: p = "*" # the predicted bundle is illegal or doesn't exist in this language
                    phoneme_tokens.append(p)
            graphemes = self._phonemes2word(phoneme_tokens, 'phonemes')
        return ''.join(graphemes)

    def phonemes2word(self, sequence: Union[List[str], Tuple[str]], mode:str) -> str:
        """
        Wrapper for _phonemes2word.
        sequence is list, of either features (['1', '2', '3', '$', '4', '5', 'NA', '$', ...]),
        phonemes (['p', 't', 'o:', ...]), or combined (['1', '2', '3', 'p', '$', '4', '5', 'NA', 'o:', '$', ...])
        """
        assert mode in {'features', 'phonemes'}, f"Mode {mode} is invalid"

        if mode == 'features':
            phon_feats = ','.join(sequence).split(',$,')
            if self.manual_phonemes2word and self.phon_use_attention:
                new_sequence = self._phonemes2word([e.split(',')[-1] for e in phon_feats], mode='phonemes')
            else:
                new_sequence = self._phonemes2word([p.split(',') for p in phon_feats], mode='features')
        else: # mode=='phonemes'
            new_sequence = self._phonemes2word(sequence, mode='phonemes')

        return new_sequence

# For debugging purposes:
def two_way_conversion(w, lang_phonology):
    print(f"PHON_USE_ATTENTION, lang = false, '{language}'")
    print(f"w = {w}")
    ps = lang_phonology.word2phonemes(w, mode='phonemes')
    feats = lang_phonology.word2phonemes(w, mode='features')
    print(f"phonemes = {ps}\nfeatures = {feats}")

    p2word = lang_phonology.phonemes2word(ps, mode='phonemes')
    print(f"p2word: {p2word}\nED(w, p2word) = {editDistance(w, p2word)}")

    feats = [f.split(',') for f in ','.join(feats).split(',$,')]
    f2word = lang_phonology.phonemes2word(feats, mode='features')
    print(f"f2word: {f2word}\nED(w, f2word) = {editDistance(w, f2word)}")

if __name__ == '__main__':
    # made-up words to test the correctness of the g2p/p2g conversions algorithms (for debugging purposes):
    example_words = {'kat': 'არ მჭირდ-ებოდყეტ', 'swc': "magnchdhe-ong jwng'a", 'sqi': 'rdhëije rrçlldgj-ijdhegnjzh', 'lav': 'abscā t-raķkdzhēļšanģa',
                     'bul': 'най-ясюногщжто', 'hun': 'hűdályiokró- l eéfdzgycsklynndzso nyoyaxy', 'tur': 'yığmalılksar mveğateğwypûrtâşsmış', 'fin': 'ixlmksnngvnk- èeé aatööböyynyissä'}
    language = 'kat'
    max_features_size = max([len(p2f_dict[p]) for p in langs_properties[language][0].values() if p in p2f_dict])
    phon_use_attention = False
    lang_phonology = LanguageSetup(language, langs_properties[language][0], max_features_size,
                                   phon_use_attention, langs_properties[language][1], langs_properties[language][2])

    two_way_conversion(example_words[language], lang_phonology)
