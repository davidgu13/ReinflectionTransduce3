# ReinflectionTransduce3

Repo for the paper [Future-Pape-Name](https://github.com/davidgu13/ReinflectionTransduce3). This is actually a fork of the model by [Makarov & Clematide](https://github.com/ZurichNLP/coling2018-neural-transition-based-morphology), adjusted to the 2 new formats that use phonology. Also, it is converted to Python3.

## Running Instructions
See the dependencies at `requirements.txt`.

For Linux, run `dummy.sh`. Configure the 4 arguments and the paths at `defaults.py`.

For Windows, run `run_transducer.py` with the relevant arguments. 

### Data
The script expects train, dev & test files to be located at `DATA_PATH`

### Orthography-Based Formats
To reformat words as sequences of phonological features, use the flag `--use-phonology`. To model them as self-representation of phonemes, use also the flag `--self-attn`.