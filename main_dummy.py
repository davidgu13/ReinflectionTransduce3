"""Trains and evaluates a state-transition model for inflection generation, using the sigmorphon 2017 shared task
data files and evaluation script.

Usage:
  run_transducer.py [--dynet-seed SEED] [--dynet-mem MEM] [--dynet-autobatch ON]
  [--transducer=TRANSDUCER] [--sigm2017format] [--no-feat-format]
  [--input=INPUT] [--feat-input=FEAT] [--action-input=ACTION] [--pos-emb] [--avm-feat-format]
  [--enc-hidden=HIDDEN] [--dec-hidden=HIDDEN] [--enc-layers=LAYERS] [--dec-layers=LAYERS]
  [--vanilla-lstm] [--mlp=MLP] [--nonlin=NONLIN] [--lucky-w=W]
  [--pretrain-dropout=DROPOUT] [--dropout=DROPOUT] [--l2=L2]
  [--optimization=OPTIMIZATION] [--batch-size=BATCH-SIZE] [--decbatch-size=BATCH-SIZE]
  [--patience=PATIENCE] [--epochs=EPOCHS] [--pick-loss]
  [--align-smart | --align-dumb | --align-cls] [--tag-wraps=WRAPS] [--try-reverse | --iterations=ITERATIONS]
  [--substitution | --copy-as-substitution] [--param-tying]
  [--mode=MODE] [--verbose] [--beam-width=WIDTH] [--beam-widths=WIDTHS]
  [--pretrain-epochs=EPOCHS | --pretrain-until=ACC] [--sample-size=SAMPLE-SIZE] [--scale-negative=S]
  [--alpha=ALPHA] [--beta=BETA] [--no-baseline]
  TRAIN-PATH DEV-PATH RESULTS-PATH [--test-path=TEST-PATH] [--reload-path=RELOAD-PATH] [--hall-path=HALL-PATH]

Arguments:
  TRAIN-PATH    destination path, possibly relative to "data/all/", e.g. task1/albanian-train-low
  DEV-PATH      development set path, possibly relative to "data/all/"
  RESULTS-PATH  results file to be written, possibly relative to "results"

Options:
  -h --help                     show this help message and exit
  --dynet-seed SEED             DyNET seed
  --dynet-mem MEM               allocates MEM bytes for DyNET
  --dynet-autobatch ON          perform automatic minibatching
  --transducer=TRANSDUCER       transducer model to use: hacm / st-haem / haem / hard [default: haem]
  --sigm2017format              assume sigmorphon 2017 input format (lemma, word, feats)
  --no-feat-format              no features format (input, *, output)
  --input=INPUT                 character embedding dimension [default: 100]
  --feat-input=FEAT             feature embedding dimension.  "0" denotes "bag-of-features". [default: 20]
  --action-input=ACTION         action embedding dimension [default: 100]
  --pos-emb                     embedding POS (or the first feature in the sequence of features) as a non-atomic feature
  --avm-feat-format             features are treated as an attribute-value matrix (`=` pairs attributes with values)
  --enc-hidden=HIDDEN           hidden layer dimension of encoder RNNs [default: 200]
  --enc-layers=LAYERS           number of layers in encoder RNNs [default: 1]
  --dec-hidden=HIDDEN           hidden layer dimension of decoder RNNs [default: 200]
  --dec-layers=LAYERS           number of layers in decoder RNNs [default: 1]
  --vanilla-lstm                use vanilla LSTM instead of DyNet 1's default coupled LSTM
  --mlp=MLP                     MLP hidden layer dimension. "0" denotes "no hidden layer". [default: 0]
  --nonlin=NONLIN               if mlp, this non-linearity is applied after the hidden layer. ReLU/tanh [default: ReLU]
  --lucky-w=W                   if feat-input==0, scale the "bag-of-features" vector by W [default: 55]
  --dropout=DROPOUT             variotional dropout in decoder RNN [default: 0.5]
  --pretrain-dropout=DROPOUT    if pretraining with MLE, this dropout rate is used, otherwise the rate of '--dropout'
  --optimization=OPTIMIZATION   optimization method ADAM/SGD/ADAGRAD/MOMENTUM/ADADELTA [default: ADADELTA]
  --l2=L2                       l2-regularization coefficient. Regularization is applied to RNNs and classifier. [default: 0]
  --batch-size=BATCH-SIZE       batch size [default: 1]
  --decbatch-size=BATCH-SIZE    batch size for decoding [default: 1]
  --patience=PATIENCE           maximal patience for early stopping [default: 10]
  --epochs=EPOCHS               number of training epochs [default: 30]
  --pick-loss                   best model should have the highest dev loss (and not dev accuracy)
  --align-smart                 align with Chinese restaurant process like in Aharoni & Goldberg paper. Default.
  --align-dumb                  align by padding the shortest string on the right (lemma or inflected word)
  --align-cls                   align by aligning the strings' common longest substring first and then padding both strings.
  --try-reverse                 if align-dumb, try reversing lemma and word strings if no COPY action is generated
                                (this will be the case with prefixating morphology)
  --iterations=ITERATIONS       if align-smart, use this number of iterations in the aligner [default: 150]
  --substitution                use substitution of y_i (for any x_i) as an action instead of (insert of y_i + delete)
  --copy-as-substitution        treat copy as substitutions?
  --param-tying                 use same embeddings for characters and actions inserting them
  --tag-wraps=WRAPS             wrap lemma and word with word boundary tags?
                                  both (use opening and closing tags)/close (only closing tag)/None [default: both]
  --verbose                     visualize results of internal evaluation, display train and dev set alignments
  --mode=MODE                   various operation modes of the trainer:
                                    eval (run evaluation without training)/mle (MLE training)/rl (reinforcement
                                    learning training) [default: mle]
  --alpha=ALPHA                 if mode==mrt, MRT distribution-smoothing parameter alpha [default: 0.05]
  --beta=BETA                   if mode==mrt or mode==rl, MRT normalized edit distance scaling factor [default: 1]
  --no-baseline                 if mode==rl, RL does not use baseline correction of reward
  --beam-width=WIDTH            beam width for beam-search decoding [default: 8]
  --beam-widths=WIDTHS          a comma-separated sequence of beam widths for the final (dev/test) decoding with beam search
  --pretrain-epochs=EPOCHS      number of epochs to pretrain the model with MLE training [default: 0]
  --pretrain-until=ACC          MLE pretraining stops as soon as training accuracy score ACC is reached [default: 0]
  --sample-size=SAMPLE-SIZE     if mode==mrt or mode=rl, number of samples drawn from the model per training sample [default: 20]
  --scale-negative=S            if mode==rl, scale negative rewards by S [default: 0.1]
  --test-path=TEST-PATH         test set path
  --reload-path=RELOAD-PATH     reload a pretrained model at this path (possibly relative to RESULTS-PATH)
  --hall-path=HALL-PATH         path with hallucinated data
"""

# from docopt import docopt
#
# print(f"__doc__ = {__doc__}")
# arguments = docopt(__doc__)
# print(f"docopt(__doc__) = \n{arguments}")
# print()

from DataRelatedClasses.DataSets.EditDataSet import EditDataSet

def Vocab_example():
    from DataRelatedClasses.Vocabs.Vocab import Vocab
    # obj = Vocab()
    s = 'fndsoanofinviabfsainvdvnsiubviorwgru ghruebiudsabdiufbv uiahfudaid'
    obj = Vocab.from_list(list(s), encoding=None)
    obj.printer()

    print(obj['f'])
    print(obj['f'])
    print(obj['f'])

    obj.printer()



def dataset_example():
    import pdb
    pdb.set_trace()
    # dset = EditDataSet(False, False, False)
    print("\n\nPrinting info about the dataset")
    train_data = EditDataSet.from_file(os.path.join('.data', 'Reinflection', 'kat.V', 'kat.V.form.dev.txt'), EditVocab())
    print(f"train_data.vocab = {train_data.vocab}")
    print(f"train_data.training_data = {train_data.training_data}")
    print(f"train_data.length = {train_data.length}")
    print(f"train_data.filename = {train_data.filename}")
    print(f"type(train_data.samples) = {type(train_data.samples)}")
    print(f"train_data.samples[5] = {train_data.samples[5]}")
    print(f"train_data.tag_wraps = {train_data.tag_wraps}")
    print(f"train_data.verbose = {train_data.verbose}")

if __name__ == '__main__':
    # Import EditVocab class. See how it looks for a naive g-g reinflection dataset
    # Do the same for AlignedDataSample class.
    # Do the same for EditDataset class.
    # from DataRelatedClasses.EditVocab import EditVocab
    # obj = EditVocab(pos_emb=False, avm_feat_format=False, param_tying=True, encoding=None)

    Vocab_example()