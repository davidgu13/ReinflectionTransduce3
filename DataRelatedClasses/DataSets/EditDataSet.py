from DataRelatedClasses.DataSets.AlignedDataSet import AlignedDataSet
from DataRelatedClasses.Vocabs.EditVocab import EditVocab
from DataRelatedClasses.utils import action2string
from aligners import dumb_align
from defaults import DELETE, COPY, ALIGN_SYMBOL, DELETE_CHAR,COPY_CHAR

class EditDataSet(AlignedDataSet):
    # this dataset uses COPY action
    def __init__(self, try_reverse=False, substitution=False, copy_as_substitution=False,
                 reorder_deletes=True, freq_check=(0.1, 0.3), **kwargs):
        # "try reverse" only makes sense with dumb aligner
        self.try_reverse = try_reverse and self.aligner == dumb_align  # @TODO Fix bug
        if self.try_reverse:
            print('USING STRING REVERSING WITH DUMB ALIGNMENT...')
            print('USING DEFAULT ALIGN SYMBOL ~')

        self.copy_as_substitution = copy_as_substitution
        self.substitution = substitution
        if copy_as_substitution is True:
            self.substitution = True
            print('TREATING COPY AS SUBSTITUTIONS')

        if self.substitution is True:
            self.reorder_deletes = False
            print('USING SUBSTITUTION ACTIONS, NOT REORDERING DELETES')
        else:
            self.reorder_deletes = reorder_deletes

        # "frequency check" for COPY and DELETE actions
        self.freq_check = freq_check

        super(EditDataSet, self).__init__(**kwargs)

        if self.freq_check:
            copy_low, delete_high = self.freq_check
            # some stats on actions
            action_counter = self.vocab.act.freq()
            # print action_counter.values()
            freq_delete = action_counter[DELETE] / sum(action_counter.values())
            freq_copy = action_counter[COPY] / sum(action_counter.values())

            print(('Alignment results: COPY action freq {:.3f}, '
                   'DELETE action freq {:.3f}'.format(freq_copy, freq_delete)))

            if freq_copy < copy_low:
                print('WARNING: Too few COPY actions!\n')
            if freq_delete > delete_high:
                print('WARNING: Many DELETE actions!\n')

    def _build_oracle_actions(self, lemma, word, sample, **kwargs):
        # Makarov et al 2017 Algorithm 1
        def _build(lemma, word):
            actions = []
            alignment_len = len(lemma)
            has_copy = False
            for i, (l, w) in enumerate(zip(lemma, word)):
                if l == ALIGN_SYMBOL:
                    actions.append(self.vocab.act[w])
                elif w == ALIGN_SYMBOL:
                    actions.append(self.vocab.act[DELETE_CHAR])
                elif l == w:
                    if i + 1 == alignment_len:
                        # end of string => insert </s>
                        actions.append(self.vocab.act[w])
                    elif self.copy_as_substitution:
                        # treat copy as another substitution action
                        actions.append(self.vocab.act[w + '@'])
                    else:
                        # treat copy as a special action
                        actions.append(self.vocab.act[COPY_CHAR])
                        has_copy = True
                else:
                    # substitution
                    if self.substitution:
                        subt = self.vocab.act[w + '@'],
                        # subt = (self.vocab.act[u'@' + l + w + u'@'],)
                    else:
                        subt = self.vocab.act[DELETE_CHAR], self.vocab.act[w]
                    actions.extend(subt)
            return actions, has_copy

        actions, has_copy = _build(lemma, word)

        if self.try_reverse and has_copy:
            # no copying is being done, probably this sample uses prefixation.
            # Try aligning original pair from the end:
            reversed_pair = sample.lemma[::-1], sample.word[::-1]
            [(new_al_lemma, new_al_word)] = self.aligner([reversed_pair], ALIGN_SYMBOL)
            ractions, has_copy = _build(new_al_lemma[::-1], new_al_word[::-1])

            if has_copy:
                print(('Reversed aligned: {} => {}\n'
                       'Forward alignment: {}, REVERSED alignment: {}'.format(
                    al_lemma, al_word,
                    action2string(actions, self.vocab),
                    action2string(ractions, self.vocab))))
                actions = ractions

        if self.reorder_deletes:
            reordered_actions = []
            suffix = []
            for i, c in enumerate(actions):
                if i == 0 or c == COPY:
                    reordered_actions.append(c)
                    # count deletes and store inserts
                    # between two copy actions
                    inserts = []
                    deletes = 0
                    for b in actions[i + 1:]:
                        if b == COPY:
                            # copy
                            break
                        elif b == DELETE:
                            # delete
                            deletes += 1
                        else:
                            inserts.append(b)
                    between_copies = [DELETE] * deletes + inserts
                    reordered_actions.extend(between_copies)
            actions = reordered_actions + suffix

        if self.verbose == 2:
            print('{}\n{}\n{}\n'.format(word,
                                        action2string(actions, self.vocab),
                                        lemma))

        sample.set_actions(actions, lemma, word)

    @classmethod
    def from_file(cls, filename, vocab=None, pos_emb=True, avm_feat_format=False,
                  param_tying=False, **kwargs):
        if vocab:
            assert isinstance(vocab, EditVocab)
        else:
            vocab = EditVocab(pos_emb=pos_emb, avm_feat_format=avm_feat_format, param_tying=param_tying)
        print(vocab)
        return super(EditDataSet, cls).from_file(filename, vocab, pos_emb=pos_emb,
                                                 avm_feat_format=avm_feat_format, **kwargs)
