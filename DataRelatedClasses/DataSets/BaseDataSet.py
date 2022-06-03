import os
import codecs
from random import shuffle

from DataRelatedClasses.DataSamples.BaseDataSample import BaseDataSample

class BaseDataSet(object):
    # class to hold an encoded dataset
    def __init__(self, filename, samples, vocab, training_data, tag_wraps, verbose, **kwargs):
        self.filename = filename
        self.samples = samples
        self.vocab = vocab
        self.length = len(self.samples)
        self.training_data = training_data
        self.tag_wraps = tag_wraps
        self.verbose = verbose

    def __len__(self):
        return self.length

    @classmethod
    def from_file(cls, filename, vocab, DataSample=BaseDataSample,
                  encoding='utf8', delimiter='\t', sigm2017format=True, no_feat_format=False,
                  pos_emb=True, avm_feat_format=False, tag_wraps='both', verbose=True, **kwargs):
        # filename (str):   tab-separated file containing morphology reinflection data:
        #                   lemma word feat1;feat2;feat3...

        if isinstance(filename, list):
            filename, hallname = filename
            print('adding hallucinated data from', hallname)
        else:
            hallname = None

        training_data = True if 'inflec_data' in os.path.basename(filename) or 'all_ns' in os.path.basename(filename) \
                                or 'train' in os.path.basename(filename) else False
        if training_data:
            print('=====TRAIN TRAIN TRAIN=====')
        else:
            print('=====TEST TEST TEST=====')

        print(filename)

        # training_data = True if 'train' in os.path.basename(filename) else False
        datasamples = []

        print(f'Loading data from file: {filename}')
        print(f"These are {'training' if training_data else 'holdout'} data.")
        print('Word boundary tags?', tag_wraps)
        print('Verbose?', verbose)

        if avm_feat_format:
            # check that `avm_feat_format` and `pos_emb` does not clash
            if pos_emb:
                print('Attribute-value feature matrix implies that no specialized pos embedding is used.')
                pos_emb = False

        with codecs.open(filename, encoding=encoding) as f:
            for row in f:
                split_row = row.strip().split(delimiter)
                sample = DataSample.from_row(vocab, training_data, tag_wraps, verbose,
                                             split_row, sigm2017format, no_feat_format,
                                             pos_emb, avm_feat_format)
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
                    sample = DataSample.from_row(vocab, training_data, tag_wraps, verbose,
                                                 split_row, sigm2017format, no_feat_format,
                                                 pos_emb, avm_feat_format)
                    datasamples.append(sample)
            print(f'hallucinated data added. training expanded from {old_len} to {len(datasamples)} examples')

        return cls(filename=filename, samples=datasamples, vocab=vocab,
                   training_data=training_data, tag_wraps=tag_wraps, verbose=verbose, **kwargs)
