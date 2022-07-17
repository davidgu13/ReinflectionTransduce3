import os
import codecs
from random import shuffle
from typing import List

from DataRelatedClasses.DataSamples.BaseDataSample import BaseDataSample
from DataRelatedClasses.Vocabs.VocabBox import VocabBox
from Word2Phonemes.languages_setup import LanguageSetup

class BaseDataSet(object):
    # class to hold an encoded dataset
    def __init__(self, filename, samples: List[BaseDataSample], vocab: VocabBox, training_data: bool, tag_wraps: str, verbose: bool, **kwargs):
        self.filename = filename
        self.samples = samples
        self.vocab = vocab
        self.length = len(self.samples)
        self.training_data = training_data
        self.tag_wraps = tag_wraps
        self.verbose = verbose
        self.phonology_converter = kwargs.get('phonology_converter') # None if kwargs['use_phonology] is None, otherwise of type LanguageSetup

    def __len__(self):
        return self.length

    @classmethod
    def from_file(cls, filename, vocab: VocabBox, DataSample=BaseDataSample,
                  encoding='utf8', delimiter='\t', tag_wraps='both', verbose=True, **kwargs):
        # filename (str):   tab-separated file containing morphology reinflection data:
        #                   lemma word feat1;feat2;feat3...

        kwargs['use_phonology'] = kwargs.get('language') is not None
        if kwargs['use_phonology']:
            language = kwargs['language']
            phon_use_attention = False
            phonology_converter = LanguageSetup.create_phonology_converter(language, phon_use_attention)
        else:
            phonology_converter = None
        kwargs['phonology_converter'] = phonology_converter

        if isinstance(filename, list):
            filename, hallname = filename
            print('adding hallucinated data from', hallname)
        else:
            hallname = None

        training_data = any([c in os.path.basename(filename) for c in ['inflec_data', 'all_ns', 'train']])
        if training_data:
            print('=====TRAIN TRAIN TRAIN=====')
        else:
            print('=====TEST TEST TEST=====')

        datasamples = []

        print(f'Loading data from file: {filename}')
        print(f"These are {'training' if training_data else 'holdout'} data.")
        print('Word boundary tags?', tag_wraps)
        print('Verbose?', verbose)

        with codecs.open(filename, encoding=encoding) as f:
            for row in f:
                split_row = row.strip().split(delimiter)
                sample = DataSample.from_row(vocab, tag_wraps, verbose, split_row,
                         sigm2017format=kwargs['sigm2017format'], phonology_converter=phonology_converter)
                datasamples.append(sample)

        if hallname:
            old_len = len(datasamples)
            if len(datasamples) > 5000:
                shuffle(datasamples)
                datasamples = datasamples[:5000]
            with codecs.open(hallname, encoding=encoding) as f:
                for row in f:
                    split_row = row.strip().split(delimiter)
                    letters = set(split_row[0]) | set(split_row[3])
                    if '|' in letters:
                        continue
                    sample = DataSample.from_row(vocab, tag_wraps, verbose, split_row)
                    datasamples.append(sample)
            print(f'hallucinated data added. training expanded from {old_len} to {len(datasamples)} examples')

        from Word2Phonemes.g2p_config import feature2idx
        kwargs['space_character'] = vocab.char.w2i.get(str(feature2idx[' ']))
        return cls(filename=filename, samples=datasamples, vocab=vocab,
                   training_data=training_data, tag_wraps=tag_wraps, verbose=verbose, **kwargs)
