# This is a Python3 script for extracting the runs results to an Excel file, including final dev and test accuracies.
import os
from os.path import basename, isdir, join
import re
from itertools import product
from pandas import DataFrame
from util import are_substrings_in_string

def find_float_in_line(line):
    matches = re.findall("\d+\.\d+", line)
    assert len(matches) == 1, "Invalid output!"
    return float(matches[0])

def find_2_floats_in_line(line):
    a, b = re.findall("\d+\.\d+", line)
    return float(a), float(b)

def extract_accuracies_from_single_folder(folder, io_format):
    # The folder has the format: Outputs_YYYY-MM-DD hhmmss_lang_pos_training-mode_IO_analogy_seed.
    # If the output format is not graphemes ('g'), then reevaluation was made at graphemes level.
    assert io_format in {'g_g, f_f'}
    datetime, lang, pos, training_mode = basename(folder).split("_")[1:5]

    lines = open(join(folder, 'f.stats'), encoding='utf8').read().split('\n')

    dev_accuracy, test_accuracy, test_ed, test_graphemes_accuracy, test_graphemes_ed = 0, 0, 0, 0, 0
    if io_format == 'g_g':
        dev_accuracy, test_accuracy, test_ed = find_float_in_line(lines[-3]), find_float_in_line(lines[-2]), -1
        test_graphemes_accuracy, test_graphemes_ed = None, None
    elif io_format == 'f_f':
        # See write_generalized_measures in utils.py; dev_ and test_ accuracy mean here eval at features level
        dev_accuracy, test_accuracy = find_float_in_line(lines[-6]), find_float_in_line(lines[-5])

        test_features_accuracy, test_ed = find_2_floats_in_line(lines[-3])
        assert test_accuracy == test_features_accuracy

        test_graphemes_accuracy, test_graphemes_ed = find_2_floats_in_line(lines[-2])

    return datetime, lang, pos, training_mode, dev_accuracy, test_accuracy, test_ed, test_graphemes_accuracy, test_graphemes_ed


def main():
    results_folder = join('.', 'Results')
    io_format = 'f_f' # 'g_g'
    analogy_types, seeds  = ['None'], ['100', '42', '21']

    lang_pos_groups = dict(zip(range(1,16), ['kat_V', 'kat_N', 'fin_ADJ', 'swc_V', 'swc_ADJ', 'fin_V', 'sqi_V', 'hun_V',
                                             'bul_V', 'bul_ADJ', 'lav_V', 'lav_N', 'tur_V', 'tur_ADJ', 'fin_N']))
    excel_results_file = f"Test-Results {'_'.join(analogy_types)}-{'_'.join(seeds)}-{io_format}.xlsx"

    accuracies = []
    for analogy, seed in product(analogy_types, seeds):
        current_parent_folder = join(results_folder, f"{analogy}_{seed}_{io_format}")
        assert isdir(current_parent_folder), f"Folder Results doesn't exist!"

        folders = [join(current_parent_folder, f) for f in os.listdir(current_parent_folder)
                    if isdir(join(current_parent_folder, f)) and are_substrings_in_string(f, (analogy, seed, io_format))]

        for folder in folders:
            datetime, lang, pos, training_mode, dev_accuracy, test_accuracy, test_ed, \
                test_graphemes_accuracy, test_graphemes_ed = extract_accuracies_from_single_folder(folder, io_format)
            accuracies.append([datetime, analogy, seed, lang, pos, training_mode, io_format, dev_accuracy,
                               test_accuracy, test_ed, test_graphemes_accuracy, test_graphemes_ed])

    df = DataFrame(accuracies, columns=['DateTime', 'AnalogyMode', 'Seed', 'Language', 'POS', 'Form/Lemma', 'IO mode',
                                        'Dev Accuracy', 'Test Accuracy', 'Test ED', 'Test Graphemes Accuracy', 'Test Graphemes ED'])
    df = df.fillna("")
    df.to_excel(excel_results_file)

if __name__ == '__main__':
    main()
