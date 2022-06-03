"""
This is a script for cleaning the Reinflection samples of every language. It is inspired by the cleaning methods
 implemented at g2p_config.py in the baseline LSTM+Attention model.
Note that this script is relevant for files of Unimorph 3.0, and will not necessarily be needed for later releases.
"""
import os
from os.path import join, isdir, isfile
from itertools import product
from shutil import copy2

datasets_path = join(os.getcwd(), ".data", "Reinflection")
cleaned_datasets_path = join(datasets_path, "CleanedData")
if not isdir(cleaned_datasets_path): os.mkdir(cleaned_datasets_path)

languages = ['kat', 'swc', 'sqi', 'lav', 'bul', 'hun', 'tur', 'fin']
parts_of_speech = ['V','N','ADJ']
splits = ['form', 'lemma']
file_types = ['train', 'dev', 'test']

kat_clean_sample = None # lambda x: x
swc_clean_sample = None # lambda x: x
bul_clean_sample = None # lambda x: x

def sqi_clean_sample(x:str) -> str:
    return x.replace("',","") # appears in the data only as part of "për t'u ..." (NFIN)

def lav_clean_sample(x:str) -> str:
    return x.replace('í', 'ī').replace('ŗ', 'r').replace("LgSPEC8", "LGSPEC8") # replace the 3 occurences of 'í' with 'ī', and the 28 occ. of 'ŗ'

def hun_clean_sample(x:str) -> str:
    # the " |or| " is a bug of the scraping from Wiktionary. It can appear at the end of a form
    # (search in the data for "jósolj|or|") or between 2 forms (search for "jóslok |or| jósolok").
    # There are also pipes ("|"), alone or preceded by a space " |".
    chars_to_remove = [" |or| ", "|or|", " |", "|"]
    for p in chars_to_remove: x = x.replace(p, "")
    return x

def tur_clean_sample(x:str) -> str:
    return x.replace('İ', 'i')

def fin_clean_sample(x:str) -> str:
    chars_to_remove = ['\xa0', ":", "/"]
    for p in chars_to_remove: x = x.replace(p, "")
    x = x.replace("á", "a").replace("â", "a").replace("û", "u").replace("ü", "u")
    return x

existing_combinations = os.listdir(datasets_path)
cleaners = {'kat': kat_clean_sample, 'swc': swc_clean_sample, 'sqi': sqi_clean_sample, 'lav': lav_clean_sample,
            'bul': bul_clean_sample, 'hun': hun_clean_sample, 'tur': tur_clean_sample, 'fin': fin_clean_sample}

def main():
    lang_pos_combs_with_cleaners = {lang: {'combinations': [f"{lang}.{pos}" for pos in parts_of_speech if f"{lang}.{pos}" in existing_combinations],
                                            'cleaner': cleaners[lang]} for lang in languages}

    for lang in languages:
        combinations = lang_pos_combs_with_cleaners[lang]['combinations']
        cleaner = lang_pos_combs_with_cleaners[lang]['cleaner']

        for lang_pos in combinations:
            cleaned_files_dir = join(cleaned_datasets_path, lang_pos)
            if not isdir(cleaned_files_dir): os.mkdir(cleaned_files_dir)

            for split, file_type in product(splits, file_types):
                relative_file_path = f"{lang_pos}.{split}.{file_type}.txt"
                original_file = join(datasets_path, lang_pos, relative_file_path)
                cleaned_file = join(cleaned_files_dir, relative_file_path)

                if isfile(original_file):
                    if cleaner is None:
                        copy2(original_file, cleaned_file)
                    else:
                        # read -> replace -> write. assume reinflection format
                        data = open(original_file, encoding='utf8').read().split('\n')
                        data = [line.split('\t') for line in data]

                        for i in range(len(data)):
                            src_features, src_form, trg_features, trg_form = data[i]
                            src_form, trg_form = cleaner(src_form), cleaner(trg_form)
                            data[i] = [src_features, src_form, trg_features, trg_form]

                        open(cleaned_file, mode='w', encoding='utf8').write('\n'.join(['\t'.join(item) for item in data]))

if __name__ == '__main__':
    main()