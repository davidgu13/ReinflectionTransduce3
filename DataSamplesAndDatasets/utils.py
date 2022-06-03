
def remove_pipe(string):
    string = string.strip('|')
    string = string.strip()
    try:
        if string[-3] == '|':
            string = string[:-3]
    except IndexError:
        pass
    return string


def action2string(actions, vocab):
    return ''.join(vocab.act.i2w[a] for a in actions)

def feats2string(pos, feats, vocab):
    if pos:
        pos_str = vocab.pos.i2w[pos] + ';'
    else:
        pos_str = ''
    return  pos_str + ';'.join(vocab.feat.i2w[f] for f in feats)
