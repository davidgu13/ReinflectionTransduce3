from DataRelatedClasses.Vocabs.Vocab import Vocab
from defaults import (UNK, UNK_CHAR, BEGIN_WORD, BEGIN_WORD_CHAR,
    END_WORD, END_WORD_CHAR)

class VocabBox(object):
    def __init__(self, acts, pos_emb, avm_feat_format, param_tying, encoding):

        self.w2i_acts = acts
        self.act = Vocab(acts, encoding=encoding)

        # number of special actions
        self.number_specials = len(self.w2i_acts)

        # special features
        w2i_feats = {UNK_CHAR: UNK}
        self.feat = Vocab(w2i_feats, encoding=encoding)

        if pos_emb:
            # pos features get special treatment
            self.pos = Vocab(w2i_feats, encoding=encoding)
            print('VOCAB will index POS separately.')
        else:
            self.pos = self.feat

        if avm_feat_format:
            # feature types get encoded, too
            self.feat_type = Vocab(dict(), encoding=encoding)
            print('VOCAB will index all feature types.')
        else:
            self.feat_type = self.feat

        if param_tying:
            # use one set of indices for acts and chars
            self.char = self.act
            print('VOCAB will use same indices for actions and chars.')
        else:
            # special chars
            w2i_chars = {BEGIN_WORD_CHAR: BEGIN_WORD,
                         END_WORD_CHAR: END_WORD,
                         UNK_CHAR: UNK}
            self.char = Vocab(w2i_chars, encoding=encoding)

        # encoding of words
        self.word = Vocab(encoding=encoding)

        # training set cut-offs
        self.act_train = None
        self.feat_train = None
        self.pos_train = None
        self.char_train = None
        self.feat_type_train = None

    def __repr__(self):
        return f'VocabBox (act, feat, pos, char, feat_type) with the following ' \
               f'special actions: {self.w2i_acts}'

    def train_cutoff(self):
        # store indices separating training set elements
        # from elements encoded later from unseen samples
        self.act_train = len(self.act)
        self.feat_train = len(self.feat)
        self.pos_train = len(self.pos)
        self.char_train = len(self.char)
        self.feat_type_train = len(self.feat_type)

