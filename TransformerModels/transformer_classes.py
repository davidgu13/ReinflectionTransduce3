from dynet import * 

def make_time_distributed(x):
    """
    Flatten multi-dim matrix to single batched vector for efficient calculation
    Based on work by Cong Duy Vu Hoang
    """
    d = x.dim()
    b = d[1]
    d = d[0]
    
    total_words = d[1] * b
    return reshape(x, (d[0], 1), total_words)

def make_reverse_time_distributed(x, seq_len, b_):
    d = x.dim()[0]
    return reshape(x, (d[0], seq_len), b_)

def create_triangle_mask(length, upper):
    return inputTensor(np.triu(np.ones((length, length))))
    

class LinearLayer(object):
    """
    Simple Linear Layer (w/ or w/o bias)
    
    Class which employs a basic Linear Layer, with an option to efficiently run on sequences of large batches
    """
    def __init__(self, model, input_dim, output_dim, have_bias, use_he_init=True, name=""):
        self._have_bias = have_bias
        
        init_w = None
        init_bias = None
        if use_he_init:
            # may not be defined, so just ry
            try:
                init_w = LeCunUniformInitializer(input_dim)
                init_bias = LeCunUniformInitializer(output_dim)
            except:
                pass
        
        self._p_W = model.add_parameters((output_dim, input_dim), init=init_w, name=name + '.ll.w')
        if have_bias:
            self._p_b = model.add_parameters((output_dim), init=init_bias, name=name + '.ll.b')
    
    def __call__(self, x, flatten_sequence=False):
        """
        Run the input Expression x through the linear layer
        
        Args:
            x (Expression): The vector to run through the linear layer (can be batched) 
            flatten_sequence (bool): If the expression is 2 dimensions, and we run to all the columns through the linear layer (treat ({x,y},n) as ({x}, y*n) for efficient multiplication)
        """
        x_in = x
        if flatten_sequence:
            x_in = make_time_distributed(x)
        
        if self._have_bias:
            x_out = affine_transform([self._p_b, self._p_W, x_in])
        else: x_out = self._p_W * x_in
        
        if flatten_sequence:
            d = x.dim()
            b = d[1]
            d = d[0]
            x_out = make_reverse_time_distributed(x_out, d[1], b)
        
        return x_out

class FeedForwardLayer(object):
    """
    FeedForward Layer w/ activation function
    
    Class which employs a basic Feed-Forward network with 2 layers and activation function
    
    Accepts an input/output dimension, and allows a factor for internal dim
    """
    def __init__(self, model, num_units, n_ff_factor, activation=rectify, name=''):
        self._l_inner = LinearLayer(model, num_units, num_units * n_ff_factor, True, name=name + '.feedforward')
        self._l_outer = LinearLayer(model, num_units * n_ff_factor, num_units, True, name=name + '.feedforward')
        self._activation = activation
    
    def __call__(self, x, dropout=0.0):
        x = self._l_outer(self._activation(self._l_inner(x)))
        if dropout > 0.0:
            # we use column-major dropout
            x = dropout_dim(x, 1, dropout)
        return x

PSEUDO_MIN_VALUE = -99999999999.0
class MaskBase(object):
    """
        MaskBase consists of all functions for maskings (both padding positions and future blinding)
        
        Useful for applying batches, especially with transformers
    """
    def __init__(self):
        self._i_seq_mask = None
        self._i_mask_pp_k = None
        self._i_mask_pp_q = None
        self._i_mask_fb = None
        
    def get_seq_mask(self):
        return self._i_seq_mask
        
    def get_k_mask(self):
        return self._i_mask_pp_k
    
    def get_q_mask(self):
        return self._i_mask_pp_q
        
    def get_fb_mask(self):
        return self._i_mask_fb
       
    def create_future_blinding_mask(self, len):
        self._i_mask_fb = create_triangle_mask(len, False)
    
    def create_seq_mask_expr(self, v_seq_masks, self_attn=True):
        l = len(v_seq_masks[0])
        
        v_i_seq_masks = []
        for i in range(len(v_seq_masks)):
            i_mask = inputVector(v_seq_masks[i])
            if not self_attn: i_mask = reshape(i_mask, (1, len(v_seq_masks[i])))
            # for self attn put in neg-infinity, and for ctx put in 1s with the words and 0s where no words
            # use a pseduo neg-infinity 
            v_i_seq_masks.append(i_mask * PSEUDO_MIN_VALUE if self_attn else 1.0 - i_mask)
        
        self._i_seq_mask = concatenate_to_batch(v_i_seq_masks)
            
    def create_padding_positions_masks(self, nheads): 
        """
            Creating the padding masks for self-attention, using internal seq-mask
        """
        l = self._i_seq_mask.dim()[0][0]
        self._i_mask_pp_k = concatenate_to_batch([concatenate_cols([self._i_seq_mask] * l)] * nheads) # (l, l), batch_size * nheads
        self._i_mask_pp_q = 1.0 - self._i_mask_pp_k / PSEUDO_MIN_VALUE
        
    def create_padding_positions_masks_with_src(self, _i_src_seq_mask, nheads): 
        """
            Creating the padding masks for source-attention, using seq-mask
        """
        ly = self._i_seq_mask.dim()[0][1]
        lx = _i_src_seq_mask.dim()[0]
        if type(lx) is tuple:
            lx = lx[0]
        
        # key+query mask
        self._i_mask_pp_k = concatenate_to_batch([concatenate_cols([_i_src_seq_mask] * ly)] * nheads) # (lx, ly), batch_size * nheads
        self._i_mask_pp_q = concatenate_to_batch([concatenate([self._i_seq_mask] * lx)] * nheads) # (lx,ly), batch_size * nheads

def split_rows(x, h):
    d = x.dim()[0]
    steps = int(d[0] / h)
    out = []
    for i in range(0, d[0], steps):
        out.append(pick_range(x, i, i + steps))
    return out
    
def split_batch(x, h):
    d = x.dim()
    b = d[1]
    steps = int(b / h)
    
    out = []
    for i in range(0, b, steps):
        out.append(pick_batch_elems(x, range(i, i + steps)))
    return out

class ParallelMultiHeadAttentionLayer(object):
    """
        Multi-Head Attention Layer, pseudo-batching for multi-head attention computing (faster)
        
        Currently only support the luong attention type (dot-product)
    """
    def __init__(self, model, dim, nheads, use_bias=False, apply_future_blinding=False, name=''):
        self._l_Q = LinearLayer(model, dim, dim, use_bias, name=name + '.attn.linear-query')
        self._l_K = LinearLayer(model, dim, dim, use_bias, name=name + '.attn.linear-keys')
        self._l_V = LinearLayer(model, dim, dim, use_bias, name=name + '.attn.linear-values')
        self._l_O = LinearLayer(model, dim, dim, use_bias, name=name + '.attn.linear-final')
        self._att_scale = 1.0 / np.sqrt(dim / nheads)
        self.dim = dim
        self.nheads = nheads
        self._apply_future_blind_mask = apply_future_blinding
    
    def __call__(self, query, keys, i_mask, dropout=0.0):
        """
            Calculate the attention weights and apply them to the keys using the dot-product attention type
        """
        lQ = query.dim()[0][1]
        Q_batch = query.dim()[1]
        lK = keys.dim()[0][1]
        K_batch = query.dim()[1]
        i_Q = self._l_Q(query) #((dim,lQ), batch)
        i_K = self._l_K(keys) #((dim,lK), batch)
        i_V = self._l_V(keys) #((dim,lK), batch)
        # we have ((dim,lq), batch)
        # we want ((dim/nheads, lq) batch*nheads)
        # so:
        # transpose = (lq,dim), batch)
        # reshape = (lq, dim/nheads), batch*nheads)
        # transpose= (dim/nheads, lq), batch*nheads)
        i_batch_Q = transpose(reshape(transpose(i_Q), (lQ, self.dim/self.nheads), self.nheads * Q_batch))
        i_batch_K = transpose(reshape(transpose(i_K), (lK, self.dim/self.nheads), self.nheads * K_batch))
        i_batch_V = transpose(reshape(transpose(i_V), (lK, self.dim/self.nheads), self.nheads * K_batch))
        # calculate alphas and apply scale
        i_batch_alphas = (transpose(i_batch_K) * i_batch_Q) * self._att_scale
        # apply source mask - give a very low score to items not in keys
        if i_mask is not None:
            i_batch_alphas = i_batch_alphas + i_mask.get_k_mask()
        # apply the tgt mask
        if self._apply_future_blind_mask:
            i_batch_alphas = i_batch_alphas + i_mask.get_fb_mask()
        
        # softmax
        i_batch_alphas = softmax(i_batch_alphas)

        # apply query mask - zero out values not in query
        if i_mask is not None:
            i_batch_alphas = cmult(i_batch_alphas, i_mask.get_q_mask())

        if dropout > 0.0:
            # as before, column-major dropout
            i_batch_alphas = dropout_dim(i_batch_alphas, 1, dropout)
        
        i_proj = i_batch_V * i_batch_alphas
        # reshape it back
        i_proj = transpose(reshape(transpose(i_proj), (lQ, self.dim), Q_batch))
        return self._l_O(i_proj)

class MultiHeadAttentionLayer(object):
    """
        Multi-Head Attention Layer for multi-head attention computing (faster)
        
        Currently only support the luong attention type (dot-product)
    """
    def __init__(self, model, dim, nheads, use_bias=False, apply_future_blinding=False, name=''):
        self._v_l_Q = []
        self._v_l_K = []
        self._v_l_V = []
        for i in range(nheads):
            self._v_l_Q.append(LinearLayer(model, dim, dim / nheads, use_bias, name=name + '.attn.h' + str(i) + '.linear-query'))
            self._v_l_K.append(LinearLayer(model, dim, dim / nheads, use_bias, name=name + '.attn.h' + str(i) + '.linear-keys'))
            self._v_l_V.append(LinearLayer(model, dim, dim / nheads, use_bias, name=name + '.attn.h' + str(i) + '.linear-values'))
        # final layer
        self._l_O = LinearLayer(model, dim, dim, use_bias, name=name + '.attn.linear-final')
            
        self._att_scale = 1.0 / np.sqrt(dim / nheads)
        self.dim = dim
        self.nheads = nheads
        self._apply_future_blind_mask = apply_future_blinding
    
    def __call__(self, query, keys, i_mask, dropout=0.0):
        """
            Calculate the attention weights and apply them to the keys using the dot-product attention type
        """
        
        v_atts = []
        for (_l_Q, _l_K, _l_V) in zip(self._v_l_Q, self._v_l_K, self._v_l_V):
            i_Q = _l_Q(query) #((dk,lQ), batch) 
            i_K = _l_K(keys) #((dk,lK), batch)
            i_V = _l_V(keys) #((dk,lK), batch)
            
            # calculate alphas and apply scale
            i_alphas = (transpose(i_K) * i_Q) * self._att_scale
            # apply source mask - give a very low score to items not in keys
            if i_mask is not None:
                i_alphas = i_alphas + i_mask.get_k_mask()
                
            # apply the tgt mask
            if self._apply_future_blind_mask:
                i_alphas = i_alphas + i_mask.i_mask_fb            
            # softmax
            i_alphas = softmax(i_alphas)
            # apply query mask - zero out values not in query
            if i_mask is not None:
                i_alphas = cmult(i_alphas, i_mask.get_q_mask())

            if dropout > 0.0:
                # as before, column-major dropout
                i_alphas = dropout_dim(i_alphas, 1, dropout)
            
            i_proj = i_V * i_alphas
            v_atts.append(i_proj) # ((dk, ly), batch_size)        
            
        return self._l_O(concatenate(v_atts)) # ((dim, ly), batch_size)

class SinusoidalPositionalEncoder(object):
    """
        Sinusoidal Positional Encoder
        
        Container to apply the sinusoidal positional encodings to a batched embeddings matrix
    """
    def __init__(self, dim, max_len=5000):
        log_timescale_increment = np.log(10000.0) / dim
        positions = np.arange(max_len).reshape((max_len, 1))
        div_term = np.exp(np.arange(0, dim, 2) * - log_timescale_increment).reshape((1,int(dim/2)))
        pe = np.zeros((max_len, dim))
        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term)
        self.pe = pe.T
        
    def __call__(self, x):
        d = x.dim()[0]
        return x + inputTensor(self.pe[:, :d[1]])
    
class NormalizationLayer(object):
    """
        Simple Normalization Layer
        
        Creates trainable parameters g,b, and performs layer normalization : 

    .. math::

        \\begin{split}
           \mu &= \\frac 1 n \sum_{i=1}^n x_i\\\\
           \sigma &= \sqrt{\\frac 1 n \sum_{i=1}^n (x_i-\mu)^2}\\\\
           y&=\\frac {\\boldsymbol{g}} \sigma \circ (\\boldsymbol{x}-\mu) + \\boldsymbol{b}\\\\
        \end{split}
 
        
    Reference : `Ba et al., 2016 <http://arxiv.org/abs/1607.06450>`_
    """
    def __init__(self, model, dim, name=''):
        self._p_ln_g = model.add_parameters((dim), 1.0, name + '.layer-norm.g')
        self._p_ln_b = model.add_parameters((dim), 0.0, name + '.layer-norm.b')
    
    def __call__(self, x):
        x_in = make_time_distributed(x)
        x_out = layer_norm(x_in, self._p_ln_g, self._p_ln_b)
        d = x.dim()
        b = d[1]
        d = d[0]
        x_out = make_reverse_time_distributed(x_out, d[1], b)    
        return x_out
    