# Efficient methods for operations, such as Decode
from operator import itemgetter

import dynet as dy
import numpy as np


def transformer_with_decoder_calculate_loss(transformer, source, target, options, fDev=False):
    # encode
    dy.renew_cg()
    out_enc, _, _ = transformer.forward(source, target, dropout=0.1 if not fDev else 0.0)

    losses = []
    tlen = max(map(len, target))

    # go through each batch
    for iW in range(tlen):  # for each word
        gold_preds = [kTGT_EOS if iW >= len(sent) else sent[iW] for sent in source]

        # pick out the correct dimension
        scores = dy.pick(out_enc, t, 1)
        # log softmax and loss
        if not fDev and options.use_label_smoothing:
            # do the punishment on the sum calculation
            i_log_soft = dy.log_softmax(scores)
            pre_loss = -dy.pick(i_log_soft, gold_preds)
            ls_loss = -dy.mean_elems(i_log_soft)
            losses.append((1.0 - options.label_smoothing_weight) * pre_loss + options.label_smoothing_weight * ls_loss)
        else:
            # just do the neglogsoftmax
            losses.append(dy.pickneglogsoftmax(scores, gold - pred))

            # return double sum
    return dy.sum_batches(dy.sum(losses))


def transformer_greedy_decode(transformer, source):
    dy.renew_cg()
    memory, mem_mask, _ = transformer.run_encoder([source])
    pred_target = [kTGT_SOS]

    while len(pred_target) < 100 and pred_target[-1] != kTGT_EOS:
        # dy.cg_checkpoint()
        # calculate the distribution
        cur_ydist = transformer.calc_step(memory, mem_mask, pred_target, False)
        # find the best word
        w = np.argmax(cur_ydist.npvalue())
        pred_target.append(w)
        # dy.cg_revert()
    return pred_target


def transformer_beam_decode(transformer, source, beam_size=5):
    dy.renew_cg()
    memory, mem_mask, _ = transformer.run_encoder([source])

    cur_beams = [([kTGT_SOS], 0)]
    while True:
        new_beams = []
        found_new_beams = False
        for beam in cur_beams:
            if beam[0][-1] == kTGT_EOS or len(beam[0]) >= 100:
                new_beams.append(beam)
                continue
            # dy.cg_checkpoint()
            # calculate the distribution (log-softmax)
            cur_ydist = transformer.calc_step(memory, mem_mask, beam[0], True)
            # find the best words
            npval = cur_ydist.npvalue()
            top_ws = np.argsort(npval)[-beam_size:]
            # add them in
            new_beams.extend([(beam[0] + [w], beam[1] + npval[w]) for w in top_ws])
            found_new_beams = True

            # dy.cg_revert()
        if not found_new_beams:
            break
        # take the top X
        cur_beams = list(sorted(new_beams, key=itemgetter(1), reverse=True))[:beam_size]
    return cur_beams[0][0]
