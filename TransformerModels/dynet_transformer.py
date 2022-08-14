# More generic classes, e.g. EncoderLayer and Transformer + example main
import math
from codecs import open

import dynet as dy

from TransformerModels.transformer_classes import *


class EncoderLayer:
    """
        A Transformer Encoder layer, defined as follows:
        f(x) = feed-forward(norm(attn(norm(x)) + x))
    """

    def __init__(self, model, options, name=''):
        # self attention
        self.self_attention = ParallelMultiHeadAttentionLayer(model, options.num_units, options.nheads,
                                                              options.use_bias_in_attn, False, name + '.enc-layer.self')
        # feed-forward
        self.feed_forward = FeedForwardLayer(model, options.num_units, options.n_ff_factor,
                                             activation=options.ffl_activation, name=name + '.enc-layer')
        # two normalization layers
        self.norm_self_attn = NormalizationLayer(model, options.num_units, name + '.self-attn')
        self.norm_ffl = NormalizationLayer(model, options.num_units, name + '.feedforward')
        self.norm_residual_layer = options.norm_residual_layer

    def __call__(self, x, mask, dropout=0.0):
        # it's defined as follows:
        # f(x) = feed-forward(norm(attn(norm(x)) + x))+x
        # but, we apply dropout between
        attn_x = self.norm_self_attn(x)
        if self.norm_residual_layer:
            x = attn_x
        attn_x = self.self_attention(attn_x, attn_x, mask, dropout=dropout)
        if dropout > 0.0:
            attn_x = dy.dropout_dim(attn_x, 1, dropout)
        # apply residual connectoin
        x = x + attn_x

        # apply the feedforward
        ffl_x = self.norm_ffl(x)
        if self.norm_residual_layer:
            x = ffl_x
        ffl_x = self.feed_forward(ffl_x, dropout=dropout)
        # residual connection
        return x + ffl_x


class DecoderLayer:
    """
        A Transformer Decoder layer, defined as follows:
        f(x) = feed-forward(norm(mem-attn(norm(self-attn(x) + x)) + x)) + x
    """

    def __init__(self, model, options, name=''):
        # self attention
        self.self_attention = ParallelMultiHeadAttentionLayer(model, options.num_units, options.nheads,
                                                              options.use_bias_in_attn, True, name + '.dec-layer.self')
        self.src_attention = ParallelMultiHeadAttentionLayer(model, options.num_units, options.nheads,
                                                             options.use_bias_in_attn, False, name + '.dec-layer.src')
        # feed-forward
        self.feed_forward = FeedForwardLayer(model, options.num_units, options.n_ff_factor,
                                             activation=options.ffl_activation, name=name + '.dec-layer')
        # three normalization layers
        self.norm_self_attn = NormalizationLayer(model, options.num_units, name + '.self-attn')
        self.norm_src_attn = NormalizationLayer(model, options.num_units, name + '.src-attn')
        self.norm_ffl = NormalizationLayer(model, options.num_units, name + '.feedforward')
        self.norm_residual_layer = options.norm_residual_layer

    def __call__(self, x, memory, self_mask, src_mask, dropout=0.0):
        # it's defined as follows:
        # f(x) = feed-forward(norm(mem-attn(norm(self-attn(x) + x)) + x)) + x
        # but, we apply dropout between
        attn_x = self.norm_self_attn(x)
        if self.norm_residual_layer:
            x = attn_x
        attn_x = self.self_attention(attn_x, attn_x, self_mask, dropout=dropout)
        if dropout > 0.0:
            attn_x = dy.dropout_dim(attn_x, 1, dropout)
        # apply residual connectoin
        x = x + attn_x

        # src attention
        attn_x = self.norm_src_attn(x)
        if self.norm_residual_layer:
            x = attn_x
        attn_x = self.src_attention(attn_x, memory, src_mask, dropout=dropout)
        if dropout > 0.0:
            attn_x = dy.dropout_dim(attn_x, 1, dropout=dropout)
        # apply residual connection
        x = x + attn_x

        ffl_x = self.norm_ffl(x)
        if self.norm_residual_layer:
            x = ffl_x
        # apply the feedforward
        ffl_x = self.feed_forward(ffl_x, dropout=dropout)
        # residual connection
        return x + ffl_x


class Embeddings:
    def __init__(self, model, size, options, force_no_seg=False, name=''):
        self._embed_lp = model.add_lookup_parameters((size, options.num_units), name=name + ".embeddings.lp")

        self.use_sinusoidal_encodings = options.use_sinusoidal_encodings
        if self.use_sinusoidal_encodings:
            self._positional_enc = SinusoidalPositionalEncoder(int(options.num_units))
        else:
            self._positional_enc = model.add_lookup_parameters((options.max_pos_len, options.num_units),
                                                               name=name + '.embeddings.poslp')

        self._seg_enc = model.add_lookup_parameters((options.seg_tok_types, options.num_units),
                                                    name=name + '.embeddings.seglp') if options.seg_tok_types and not force_no_seg else None
        self._scale_emb = math.sqrt(options.num_units) if options.fscale_emb else 1
        self.nheads = options.nheads
        self.num_units = options.num_units

    def __call__(self, sentences, enc_mask=None, dropout=0.1):
        if self._seg_enc is not None:
            sentences, segments = sentences
        # embeds = []
        seq_masks = [[] for _ in range(len(sentences))]
        cur_max_len = max(map(len, sentences))
        inds = []
        inds_seg = []
        inds_pos = []
        for i in range(cur_max_len):
            for j, sent in enumerate(sentences):
                seq_masks[j].append(0 if i < len(sent) else 1)
            inds.extend([sent[i] if i < len(sent) else kSRC_EOS for sent in sentences])
            if self._seg_enc is not None:
                inds_seg.extend([seg[i] if i < len(seg) else 0 for seg in segments])
            if not self.use_sinusoidal_encodings:
                inds_pos.extend([i] * len(sentences))
            # embeds.append(dy.lookup_batch(self._embed_lp, [sent[i] if i < len(sent) else kSRC_EOS for sent in sentences]))
            # if self._seg_enc is not None:
            #    embeds[-1] += dy.lookup_batch(self._seg_enc, [seg[i] if i < len(seg) else 0 for seg in segments])
            # if not self.use_sinusoidal_encodings:
            #    embeds[-1] += self._positional_enc[i]
        # concatenate and apply positional encodings
        # embeds = dy.concatenate_cols(embeds) * self._scale_emb
        # if self.use_sinusoidal_encodings:
        #    embeds = self._positional_enc(embeds)

        # lookup all the inds
        embeds = lookup_batch(self._embed_lp, inds)
        if inds_seg:
            embeds += lookup_batch(self._seg_enc, inds_seg)
        # multiply by the scale, and then do positional
        embeds *= self._scale_emb
        # add the positional embeddings
        if not self.use_sinusoidal_encodings:
            embeds += lookup_batch(self._positional_enc, inds_pos)
        else:
            embeds = self._positional_enc(embeds)

        # reshape!
        embeds = reshape(transpose(embeds), (self.num_units, cur_max_len), len(sentences))
        if dropout > 0.0:
            embeds = dy.dropout_dim(embeds, 1, dropout)

        # create the sequence mask
        self_mask = MaskBase()
        # for self mask, apply the future blinding mask
        self_mask.create_future_blinding_mask(cur_max_len)
        self_mask.create_seq_mask_expr(seq_masks)
        self_mask.create_padding_positions_masks(self.nheads)
        # source mask
        src_mask = MaskBase()
        src_mask.create_seq_mask_expr(seq_masks, False)  # *not* self attn
        if enc_mask is not None:
            src_mask.create_padding_positions_masks_with_src(enc_mask.get_seq_mask(),
                                                             self.nheads)  # note: if we were using the not-parallel attention, we would put here 1

        # return all 3
        return embeds, self_mask, src_mask


class Encoder:
    def __init__(self, model, embed, options):
        # the embeddings
        self.embed = embed

        # add the layers
        self.enc_layers = []
        for i in range(options.nlayers):
            self.enc_layers.append(EncoderLayer(model, options, name='encoder.' + str(i)))

        self.nheads = options.nheads
        # and add an extra layer normarlization
        self.final_norm = NormalizationLayer(model, options.num_units, name="encoder")

        self.add_pooler_output = options.add_pooler_output
        if self.add_pooler_output:
            self._pooler = LinearLayer(model, options.num_units, options.num_units, True, use_he_init=False,
                                       name="encoder.pooler")
            self.pooler = lambda x: dy.tanh(self._pooler(x))

    def __call__(self, sentences, dropout=0.0):
        """
            sentence is a list of list of ints
        """
        enc_output, mask, _ = self.embed(sentences, dropout=dropout)
        # run through each of the layers
        for enc in self.enc_layers:
            enc_output = enc(enc_output, mask, dropout=dropout)
        enc_output = self.final_norm(enc_output)

        pooled_output = None
        if self.add_pooler_output:
            pooled_output = self.pooler(dy.pick(enc_output, 0, 1))
        # apply a final norm
        return enc_output, mask, pooled_output


class Decoder:
    def __init__(self, model, embed, options):
        self.embed = embed
        # add the layers
        self.dec_layers = []
        for i in range(options.nlayers):
            self.dec_layers.append(DecoderLayer(model, options, name='decoder.' + str(i)))

        self.nheads = options.nheads

        # and add an extra layer normarlization
        self.final_norm = NormalizationLayer(model, options.num_units, name="decoder")

    def __call__(self, sentences, memory, enc_mask, dropout=0.0):
        """
            sentence is a list of list of ints
        """
        embeds, self_mask, src_mask = self.embed(sentences, enc_mask=enc_mask, dropout=dropout)
        dec_output = embeds
        # run through each of the layers
        for dec in self.dec_layers:
            dec_output = dec(dec_output, memory, self_mask, src_mask, dropout=dropout)
        # apply a final norm
        return self.final_norm(dec_output)


class Generator:
    def __init__(self, model, options):
        self.layers = []
        # create the layers
        for i in range(options.generator_layers - 1):
            self.layers.append(
                LinearLayer(model, options.num_units, options.num_units, True, use_he_init=False, name="generator"))
            self.layers.append(options.ffl_activation)
            self.layers.append(NormalizationLayer(model, options.num_units, "generator"))
        # add final layer
        self.layers.append(LinearLayer(model, options.num_units, options.tgt_vocab_size, True, use_he_init=False,
                                       name="generator.out"))

    def __call__(self, x):
        out = x
        for layer in self.layers[:-1]:
            out = layer(out)
        return self.layers[-1](out)


class Transformer:
    def __init__(self, options):
        self.model = dy.Model()
        self.encoder = Encoder(self.model, Embeddings(self.model, options.src_vocab_size, options, name='encoder'),
                               options)
        self.has_decoder = options.has_decoder
        if self.has_decoder:
            self.decoder = Decoder(self.model,
                                   Embeddings(self.model, options.tgt_vocab_size, options, force_no_seg=True,
                                              name='decoder'), options)
        self.generator = Generator(self.model, options)
        self.options = options

    def load_model(self, fname):
        self.model.populate(fname)

    def get_model(self):
        return self.model

    def run_encoder(self, source, dropout=0.1):
        # encode
        return self.encoder(source, dropout=dropout)

    def forward(self, source, target=None, dropout=0.1):
        # decode?
        src_enc, src_mask, src_pooled = self.run_encoder(source, dropout=dropout)
        out_enc = src_enc
        if self.has_decoder:
            assert target is not None, "cannot decode without target"
            out_enc = self.decoder(target, out_enc, src_mask, dropout=dropout)
        # generate
        return self.generator(out_enc), src_pooled

    def calc_step(self, memory, mem_mask, partial_tgt, flog_prob):
        if not self.has_decoder:
            raise Exception("Cannot calculate step with no decoder defined")
        # build the graph with the decoder
        tgt_enc = self.decoder([partial_tgt], memory, mem_mask)
        # pick the last column
        tgt_enc_col = tgt_enc if len(partial_tgt) == 1 else dy.pick(tgt_enc, len(partial_tgt) - 1, 1)

        # run through the generator
        tgt_out = self.generator(tgt_enc_col)
        return dy.log_softmax(tgt_out) if flog_prob else dy.softmax(tgt_out)


class BertModel(Transformer):
    def __init__(self, options):
        super().__init__(options)
        self.cls = LinearLayer(self.get_model(), options.num_units, 2, True, use_he_init=False,
                               name='cls.seq-relationship')

    def forward(self, source, target=None, dropout=0.1):
        pred_output, pooled_out = super().forward(source, target, dropout)
        return pred_output, self.cls(pooled_out)


class TransformerOptions:
    def __init__(self, src_vocab_size, tgt_vocab_size, num_units=512, nheads=8, has_decoder=True, use_bias_in_attn=True,
                 n_ff_factor=4, nlayers=6, use_label_smoothing=True, label_smoothing_weight=0.1,
                 ffl_activation=dy.rectify, seg_tok_types=0, fscale_emb=True, add_pooler_output=False,
                 generator_layers=1, norm_residual_layer=False, use_sinusoidal_encodings=True, max_pos_len=-1):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.num_units = num_units
        self.nheads = nheads
        self.has_decoder = has_decoder
        self.use_bias_in_attn = use_bias_in_attn
        self.n_ff_factor = n_ff_factor
        self.nlayers = nlayers
        self.use_label_smoothing = use_label_smoothing
        self.label_smoothing_weight = label_smoothing_weight
        self.ffl_activation = ffl_activation
        self.seg_tok_types = seg_tok_types
        self.fscale_emb = fscale_emb
        self.add_pooler_output = add_pooler_output
        self.generator_layers = generator_layers
        self.norm_residual_layer = norm_residual_layer
        self.use_sinusoidal_encodings = use_sinusoidal_encodings
        self.max_pos_len = max_pos_len


if __name__ == '__main__':
    # use the translate pytorch opennmt model
    transformer = Transformer(TransformerOptions(31538, 31538))
    transformer.load_model('dynet_transformer_filled.model')

    w2i_src = {}
    with open('vocab_src.txt', 'r', encoding='utf8') as w:
        for l in w:
            word, ind = l.split('\t')
            w2i_src[word] = int(ind)
    w2i_tgt = {}
    with open('vocab_tgt.txt', 'r', encoding='utf8') as w:
        for l in w:
            word, ind = l.split('\t')
            w2i_tgt[word] = int(ind)

    i2w_src = {i: w for w, i in w2i_src.items()}
    i2w_tgt = {i: w for w, i in w2i_tgt.items()}

    import transformer_train_decode as ttd

    ttd.kSRC_UNK = w2i_src['<unk>']
    ttd.kSRC_BLANK = w2i_src['<blank>']
    ttd.kSRC_SOS = w2i_src['<s>']
    ttd.kSRC_EOS = w2i_src['</s>']
    ttd.kTGT_UNK = w2i_tgt['<unk>']
    ttd.kTGT_BLANK = w2i_tgt['<blank>']
    ttd.kTGT_SOS = w2i_tgt['<s>']
    ttd.kTGT_EOS = w2i_tgt['</s>']

    from transformer_train_decode import transformer_beam_decode

    with open('test.en', 'r', encoding='utf8') as w:
        for iL, l in enumerate(w):
            ws = l.split()
            ws = [ttd.kSRC_SOS] + [w2i_src[w] for w in ws] + [ttd.kSRC_EOS]
            res = transformer_beam_decode(transformer, ws)
            print(l)
            print(' '.join([i2w_tgt[w] for w in res]))
            if iL > 5: break
