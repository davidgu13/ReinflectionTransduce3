use_cuda = False
if use_cuda:
    import dynet_config
    dynet_config.set_gpu()
import dynet as dy
import dynet_transformer as dytf
import math
import ntpath
import random
import numpy as np
import timeit
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel

# define the activation functon
def gelu(x):
    return 0.5 * dy.cmult(x, (1 + dy.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * dy.pow(x, dy.scalarInput(3))))))
#gelu = dy.rectify


# the defaults are the OpenNMT model, so anything that is different is marked in the options
# store the transformer options and the model filename, and test!
pairs = [('bert-base-cased-dynet_model.model', dytf.TransformerOptions(28996, 28996, num_units=768, nheads=12, nlayers=12, has_decoder=False, ffl_activation=gelu, use_bias_in_attn=True, seg_tok_types=2, fscale_emb=False, add_pooler_output=True, generator_layers=2, norm_residual_layer=True, use_sinusoidal_encodings=False, max_pos_len=512)), \
        #('averaged-10-epoch-opennmt-dynet_model.model', dytf.TransformerOptions(31538, 31538)), \
        ]

def batch_it(arr, size):
    start = 0
    while True:
        yield arr[start:start + size]
        start += size
        if start + size >= len(arr):
            start = 0

random.seed(424255)
for model_fname,config in pairs:
    print('Benchmarking:', ntpath.basename(model_fname))
    print('')
    
    # create our tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    from input_text import input_text
    # encode it
    tokenized_sequence = tokenizer.encode(input_text, add_special_tokens=False)
    
    batch_sizes = [1, 2, 4, 8]
    slice_sizes = [8, 64, 128, 256, 512, 1024]

    # create the model
    print(' ** Creating the model...')
    # bert model is essentially a configured transformer with an addition classifier at the end for the
    # sequential output and the pooled output.
    transformer = dytf.BertModel(config)
    
    config = AutoConfig.from_pretrained("bert-base-cased")
    pt_model = AutoModel.from_config(config)

    print(' ** Loading the parameters...')
    #transformer.load_model(model_fname)
    print(' ** Loaded! Time to evaluate!')
    print('')
    # go through each len - 8/64/128/256
    for slice_size in [8, 64, 128, 256]:
        if use_cuda:
            pt_model.cuda()
        pt_model.eval()
        # go through each batch size 1/2/4/8
        for batch_size in batch_sizes:
            print(' Batch size =', batch_size, ', Sentence len =', slice_size)
            
            dy.renew_cg()#immediate_compute=True)
            def infer_dynet_renew_cg(seq):
                dy.renew_cg()#immediate_compute=True)
                infer_dynet(seq)
                
            def infer_dynet(seq):
                seq = next(seq)
                pred_out, seq_out = transformer.forward((seq, np.zeros(seq.shape)), dropout=0.0)
                # evaluate them both
                pred_out.npvalue()
                
            def infer_torch(seq):
                seq = next(seq)
                if use_cuda:
                    seq = seq.to("cuda")
                ret = pt_model(seq)
            
            #seq_dy = np.array([tokenized_sequence[:slice_size]] * batch_size)
            #seq_pt = torch.tensor(tokenized_sequence[:slice_size], device='cpu').repeat(batch_size, 1)
            seq_dy = (np.array([x] * batch_size) for x in batch_it(tokenized_sequence, slice_size))
            seq_pt = (torch.tensor(x, device='cpu').repeat(batch_size, 1) for x in batch_it(tokenized_sequence, slice_size))

            # print the data
            runtimes = timeit.repeat(lambda: infer_dynet_renew_cg(seq_dy), repeat=30, number=3)
            average_time = sum(runtimes) / float(len(runtimes)) / 3.0
            print('   === time dynet (renew_cg) = ', average_time)

            runtimes = timeit.repeat(lambda: infer_dynet(seq_dy), repeat=30, number=3)
            average_time = sum(runtimes) / float(len(runtimes)) / 3.0
            print('   === time dynet (no renew_cg) = ', average_time)
                
            runtimes = timeit.repeat(lambda: infer_torch(seq_pt), repeat=30, number=3)
            average_time = sum(runtimes) / float(len(runtimes)) / 3.0
            print('   === time pytorch = ', average_time)

