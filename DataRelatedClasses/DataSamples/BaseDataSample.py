from DataRelatedClasses.utils import feats2string
from defaults import BEGIN_WORD, END_WORD, SPECIAL_CHARS
from DataRelatedClasses.Vocabs.VocabBox import VocabBox

def assert_inputs_are_valid(input_str, output_str, in_feats_str, out_feats_str):
    # encode features as integers
    # POS feature treated separately
    row_items = [input_str, output_str, in_feats_str, out_feats_str]
    for item in row_items:
        assert isinstance(item, str), item

    assert not any(c in input_str for c in SPECIAL_CHARS), (input_str, SPECIAL_CHARS)
    assert not any(c in output_str for c in SPECIAL_CHARS), (output_str, SPECIAL_CHARS)

class BaseDataSample(object):
    # data sample with encoded features
    def __init__(self, lemma, lemma_str, in_pos, in_feats, in_feat_str, word,
                 word_str, out_pos, out_feats, out_feat_str, tag_wraps, vocab):
        self.vocab = vocab  # vocab of unicode strings

        self.lemma = lemma  # list of encoded lemma characters
        self.lemma_str = lemma_str  # unicode string

        self.word = word  # encoded word
        self.word_str = word_str  # unicode string

        self.in_pos = in_pos  # encoded input pos feature
        self.in_feats = in_feats  # set of encoded input features

        self.out_pos = out_pos  # encoded output pos feature
        self.out_feats = out_feats  # set of encoded output features

        self.in_feat_str = in_feat_str  # original serialization of features, unicode
        self.out_feat_str = out_feat_str  # original serialization of features, unicode

        # new serialization of features, unicode
        self.in_feat_repr = feats2string(self.in_pos, self.in_feats, self.vocab)
        self.out_feat_repr = feats2string(self.out_pos, self.out_feats, self.vocab)

        self.tag_wraps = tag_wraps  # were lemma / word wrapped with word boundary tags '<' and '>'?

    def __repr__(self):
        return f'Input: {self.lemma_str},Features: {self.in_feat_repr}, Output: {self.word_str}, ' \
               f'Features: {self.out_feat_repr}, Wraps: {self.tag_wraps}'

    @classmethod
    def from_row(cls, vocab: VocabBox, tag_wraps: str, verbose, row):
        in_feats_str, input_str, out_feats_str, output_str = row
        feats_delimiter = ';'
        # feats_delimiter = u','

        assert_inputs_are_valid(input_str, output_str, in_feats_str, out_feats_str)

        """
        Insert here the conversion to phonological sample
        """

        # encode input characters
        input = [vocab.char[c] for c in input_str]  # .split()]
        # encode word
        word = vocab.word[output_str]  # .replace(' ','')]

        in_feats = in_feats_str.split(feats_delimiter)
        out_feats = out_feats_str.split(feats_delimiter)

        in_pos, out_pos = None, None

        in_feats = [vocab.feat[f] for f in set(in_feats)]
        out_feats = [vocab.feat[f] for f in set(out_feats)]

        # wrap encoded input with (encoded) boundary tags
        if tag_wraps == 'both':
            input = [BEGIN_WORD] + input + [END_WORD]
        elif tag_wraps == 'close':
            input = input + [END_WORD]

        # print features and input at a high verbosity level
        if verbose == 2:
            # print u'POS & features from {}, {}, {}: {}, {}'.format(feat_str, output_str, input_str, pos, feats)
            print(f'POS & features from {input_str}, {output_str}: {in_pos}, {in_feats} --> {out_pos}, {out_feats}')
            print(f'input encoding: {input}')

        return cls(input, input_str, in_pos, in_feats, in_feats_str, word, output_str, out_pos, out_feats,
                   out_feats_str, tag_wraps, vocab)
