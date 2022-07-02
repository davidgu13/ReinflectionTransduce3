from DataRelatedClasses.DataSets.BaseDataSet import BaseDataSet
from DataRelatedClasses.DataSamples.AlignedDataSample import AlignedDataSample
from defaults import BEGIN_WORD_CHAR, END_WORD_CHAR
from aligners import smart_align

class AlignedDataSet(BaseDataSet):
    # this dataset aligns its inputs
    def __init__(self, aligner=smart_align, **kwargs):
        super(AlignedDataSet, self).__init__(**kwargs)

        self.aligner = aligner

        # wrapping lemma / word with word boundary tags
        if self.tag_wraps == 'both':
            self.wrapper = lambda s: BEGIN_WORD_CHAR + s + END_WORD_CHAR
        elif self.tag_wraps == 'close':
            self.wrapper = lambda s: s + END_WORD_CHAR
        else:
            self.wrapper = lambda s: s

        print(f'Started aligning with {self.aligner} aligner...')
        aligned_pairs = self.aligner([(s.lemma_str, s.word_str) for s in self.samples], **kwargs)
        print('Finished aligning.')

        print('Started building oracle actions...')
        for (al, aw), s in zip(aligned_pairs, self.samples):
            al = self.wrapper(al)
            aw = self.wrapper(aw)
            self._build_oracle_actions(al, aw, sample=s, **kwargs)
        print('Finished building oracle actions.')
        print(f'Number of actions: {len(self.vocab.act)}')
        print(f'Action set: {" ".join(sorted(self.vocab.act.keys()))}')

        if self.verbose:
            print('Examples of oracle actions:')
            for a in (s.act_repr for s in self.samples[:20]):
                print(a)  # .encode('utf8')

    def _build_oracle_actions(self, al_lemma, al_word, sample, **kwargs):
        pass

    @classmethod
    def from_file(cls, filename, vocab, **kwargs):
        return super(AlignedDataSet, cls).from_file(filename, vocab, AlignedDataSample, **kwargs)
