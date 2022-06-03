from DataRelatedClasses.utils import feats2string, remove_pipe
from defaults import BEGIN_WORD, END_WORD, SPECIAL_CHARS

class BaseDataSample(object):
    # data sample with encoded features
    def __init__(self, lemma, lemma_str, in_pos, in_feats, in_feat_str, word, word_str, out_pos, out_feats,
                 out_feat_str, tag_wraps, vocab):
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
    def from_row(cls, vocab, training_data, tag_wraps, verbose, row, sigm2017format=True,
                 no_feat_format=False, pos_emb=True, avm_feat_format=False):
        if sigm2017format:
            input_str, in_feats_str, output_str, out_feats_str = row
            feats_delimiter = ';'
            input_str = remove_pipe(input_str)
            output_str = remove_pipe(output_str)
        else:
            in_feats_str, input_str, out_feats_str, output_str = row
            feats_delimiter = ';'
            # feats_delimiter = u','

        # encode features as integers
        # POS feature treated separately
        assert isinstance(input_str, str), input_str
        assert not any(c in input_str for c in SPECIAL_CHARS), (input_str, SPECIAL_CHARS)
        assert isinstance(output_str, str), output_str
        assert not any(c in output_str for c in SPECIAL_CHARS), (output_str, SPECIAL_CHARS)
        assert isinstance(in_feats_str, str), in_feats_str
        assert isinstance(out_feats_str, str), out_feats_str
        # `avm_feat_format=True` implies that `pos_emb=False`
        if avm_feat_format: assert not pos_emb

        # encode input characters
        input = [vocab.char[c] for c in input_str]  # .split()] # todo oracle
        # encode word

        word = vocab.word[output_str]  # .replace(' ','')] # todo oracle
        in_feats = in_feats_str.split(feats_delimiter) if not no_feat_format else ['']
        out_feats = out_feats_str.split(feats_delimiter) if not no_feat_format else ['']
        if pos_emb:
            # encode features and, separately, pos
            in_pos = vocab.pos[in_feats[0]]
            out_pos = vocab.pos[out_feats[0]]
            in_feats = [vocab.feat[f] for f in set(in_feats[1:])]
            out_feats = [vocab.feat[f] for f in set(out_feats[1:])]
        else:
            in_pos, out_pos = None, None
            if avm_feat_format:
                # map from encoded feature names to encoded features
                in_feats = {vocab.feat_type[f.split('=')[0]]: vocab.feat[f] for f in set(in_feats)}
                out_feats = {vocab.feat_type[f.split('=')[0]]: vocab.feat[f] for f in set(out_feats)}
            else:
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
