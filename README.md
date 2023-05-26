# ReinflectionTransducer3

Repo for the Transducer model we used in the paper [**Morphological Inflection with Phonological Features**](https://github.com/OnlpLab/InflectionWithPhonology). This is actually a fork of the model by [Makarov & Clematide](https://github.com/ZurichNLP/coling2018-neural-transition-based-morphology), adjusted to the 2 new formats that use phonology. Also, it is converted to Python3.

## Running Instructions
See the dependencies at `requirements.txt`.

For Linux, run `dummy.sh`. Configure the 4 arguments and the paths at `defaults.py`.

For Windows, run `run_transducer.py` with the relevant arguments. 

### Data
The script expects train, dev & test files to be located at `DATA_PATH`

### Using the Phonology Component
If you wish to experiment with a new language, provide a list of the alphabet and their corresponding phonemes. Optionally, you can add methods for manual conversions (digraphs and trigraphs are supported). Also, you could provide a method for data normalizing.

### Orthography-Based Formats
To reformat words as sequences of phonological features, use the flag `--use-phonology`. To model them as self-representation of phonemes, use also the flag `--self-attn`.