from DataRelatedClasses.utils import feats2string
from defaults import BEGIN_WORD, END_WORD, SPECIAL_CHARS
from DataRelatedClasses.Vocabs.VocabBox import VocabBox
from DataRelatedClasses.DataSamples.BaseDataSample import BaseDataSample
from DataRelatedClasses.DataSamples.AlignedDataSample import AlignedDataSample
from Word2Phonemes.languages_setup import LanguageSetup

def assert_inputs_are_valid(input_str, output_str, in_feats_str, out_feats_str):
    # encode features as integers
    # POS feature treated separately
    row_items = [input_str, output_str, in_feats_str, out_feats_str]
    for item in row_items:
        assert isinstance(item, str), item

    assert not any(c in input_str for c in SPECIAL_CHARS), (input_str, SPECIAL_CHARS)
    assert not any(c in output_str for c in SPECIAL_CHARS), (output_str, SPECIAL_CHARS)

class PhonologicalDataSample(AlignedDataSample):
    # data sample with encoded features
    def __init__(self, lemma, lemma_str, in_pos, in_feats, in_feat_str, word, word_str, out_pos, out_feats,
                 out_feat_str, tag_wraps, vocab, language):
        super().__init__(lemma, lemma_str, in_pos, in_feats, in_feat_str, word, word_str, out_pos, out_feats,
                         out_feat_str, tag_wraps, vocab)
        self.language = language

    # Make sure this doesn't collide with the overridden method
    @classmethod
    def from_row_to_phonemes(cls, vocab: VocabBox, tag_wraps: str, verbose, row, lang_phonology: LanguageSetup):
        in_feats_str, input_str, out_feats_str, output_str = row
        feats_delimiter = ';'

        assert_inputs_are_valid(input_str, output_str, in_feats_str, out_feats_str)

        input_by_phon_features = lang_phonology.word2phonemes(input_str, 'features')
        output_by_phon_features = lang_phonology.word2phonemes(output_str, 'features')
        # Make sure they will be used

        # encode input characters
        input = [vocab.char[c] for c in input_by_phon_features]  # .split()] # todo oracle
        # encode word
        # word = vocab.word[output_str]  # .replace(' ','')] # todo oracle
        word = vocab.word[output_by_phon_features]  # .replace(' ','')] # todo oracle

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

        # Debug here: find out what is the value of every property, and where each one is used later in the code.
        # Note: the entire pre-process of converting the lines to samples can be run independently of dynet and the rest of the flow! ("offline")

        # print features and input at a high verbosity level
        if verbose == 2:
            # print u'POS & features from {}, {}, {}: {}, {}'.format(feat_str, output_str, input_str, pos, feats)
            print(f'POS & features from {input_str}, {output_str}: {in_pos}, {in_feats} --> {out_pos}, {out_feats}')
            print(f'input encoding: {input}')

        return cls(input, input_str, in_pos, in_feats, in_feats_str, word, output_str, out_pos, out_feats,
                   out_feats_str, tag_wraps, vocab, lang_phonology)
