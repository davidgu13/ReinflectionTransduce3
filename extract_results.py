# This is a Python3 script for extracting the runs results to an Excel file, including final dev and test accuracies.
import os
from os.path import basename, isdir, join
from ast import literal_eval
import re
from itertools import product
from typing import Tuple, Dict

from editdistance import eval

from Word2Phonemes.g2p_config import p2f_dict, langs_properties
from Word2Phonemes.languages_setup import LanguageSetup

def create_phonology_converter(language: str) -> LanguageSetup:
    # Calculating and instantiating the dynamic objects
    max_feat_size = max([len(p2f_dict[p]) for p in langs_properties[language][0].values() if p in p2f_dict])  # composite phonemes aren't counted in that list
    phonology_converter = LanguageSetup(language, langs_properties[language][0], max_feat_size, False, langs_properties[language][1], langs_properties[language][2])
    return phonology_converter

def are_substrings_in_string(target_string: str, substrings: tuple) -> bool:
    return all([substring in target_string for substring in substrings])

def find_float_in_line(line):
    matches = re.findall("\d+\.\d+", line)
    assert len(matches) == 1, "Invalid output!"
    return float(matches[0])

def extract_accuracies_from_single_folder(folder):
    # The folder has the format: Outputs_YYYY-MM-DD hhmmss_lang_pos_training-mode_IO_analogy_seed
    datetime, lang, pos, training_mode = basename(folder).split("_")[1:5]

    lines = open(join(folder, 'f.stats'), encoding='utf8').read().split('\n')
    dev_accuracy, test_accuracy = find_float_in_line(lines[-3]), find_float_in_line(lines[-2])
    return datetime, lang, pos, training_mode, dev_accuracy, test_accuracy


def main():
    results_folder = join('.', 'Results')
    analogy_types, seeds, io_formats  = ['None'], ['100', '42', '21'], ['g_g'] # to be extended

    lang_pos_groups = dict(zip(range(1,16), ['kat_V', 'kat_N', 'fin_ADJ', 'swc_V', 'swc_ADJ', 'fin_V', 'sqi_V', 'hun_V',
                                             'bul_V', 'bul_ADJ', 'lav_V', 'lav_N', 'tur_V', 'tur_ADJ', 'fin_N']))
    excel_results_file = f"Test-Results {'_'.join(analogy_types)}-{'_'.join(seeds)}-{'_'.join(io_formats)}.xlsx"

    accuracies = []
    for analogy, seed, io_type in product(analogy_types, seeds, io_formats):
        current_parent_folder = join(results_folder, f"{analogy}_{seed}_{io_type}")
        assert isdir(current_parent_folder), f"Folder Results doesn't exist!"

        folders = [join(current_parent_folder, f) for f in os.listdir(current_parent_folder)
                    if isdir(join(current_parent_folder, f)) and are_substrings_in_string(f, (analogy, seed, io_type))]

        for folder in folders:
            datetime, lang, pos, training_mode, dev_accuracy, test_accuracy = extract_accuracies_from_single_folder(folder)
            accuracies.append([datetime, analogy, seed, lang, pos, training_mode, io_type, dev_accuracy, test_accuracy])

    from pandas import DataFrame
    df = DataFrame(accuracies, columns=['DateTime', 'AnalogyMode', 'Seed', 'Language', 'POS', 'Form/Lemma',
                                        'IO mode', 'Dev Accuracy', 'Test Accuracy'])
    df = df.fillna("")
    df.to_excel(excel_results_file)

def measure_pred_vs_gold(features_prediction: Tuple[str] , graphemes_gold: str, phonology_converter: LanguageSetup) -> Dict:
    graphemes_prediction = phonology_converter.phonemes2word(features_prediction, 'features')
    features_gold = tuple(phonology_converter.word2phonemes(graphemes_gold, 'features'))
    
    graphemes_equality = graphemes_gold == graphemes_prediction
    features_equality = features_gold == features_prediction
    features_ed = eval(features_prediction, features_gold)
    graphemes_ed = eval(graphemes_prediction, graphemes_gold) 
    
    return {'graphemes_equality': graphemes_equality, 'graphemes_ed': graphemes_ed,
            'features_equality': features_equality, 'features_ed': features_ed}


def extract_from_features_predictions(outputs_file: str):
    # Takes a file of the output format ('{IFET}\t{IN}\t{FET}\t{WORD}\t{GOLD}\n'). Only the last 2 matter
    language = 'kat'
    rows = open(outputs_file, encoding='utf8').readlines()
    pairs = [line.strip().split('\t')[-2:] for line in rows]
    pairs = [(literal_eval(pair[0]), pair[1]) for pair in pairs]
    phonology_converter = create_phonology_converter(language)

    measures_per_pair = [measure_pred_vs_gold(pair[0], pair[1], phonology_converter) for pair in pairs]

    average_measure_by_key = lambda key: sum([sample_results_dict[key] for sample_results_dict in measures_per_pair]) / len(pairs)
    graphemes_accuracy = average_measure_by_key('graphemes_equality')
    features_accuracy = average_measure_by_key('features_equality')
    graphemes_ed = average_measure_by_key('graphemes_ed')
    features_ed = average_measure_by_key('features_ed')

    return graphemes_accuracy, features_accuracy, graphemes_ed, features_ed

if __name__ == '__main__':
    # main()
    print(extract_from_features_predictions(join('Results', 'Outputs1426__kat_V_form_g_g_None_42', 'f.greedy.test.predictions')))