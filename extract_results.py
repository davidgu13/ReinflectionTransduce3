# This is a Python3 script for extracting the runs results to an Excel file, including final dev and test accuracies.
from itertools import product
from os import listdir
from os.path import basename, isdir, join
import re

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
    assert io_format in {'g_g', 'f_f', 'f_p_attn'}
    datetime, lang, pos, training_mode = basename(folder).split("_")[1:5]

    lines = open(join(folder, 'f.stats'), encoding='utf8').read().split('\n')

    test_ed, test_graphemes_accuracy, test_graphemes_ed = [-1] * 3
    if io_format == 'g_g':
        dev_accuracy, test_accuracy = find_float_in_line(lines[-3]), find_float_in_line(lines[-2])

    else:  # io_format in {'f_f', 'f_p_attn'}
        # See util.write_generalized_measures; dev_ and test_ accuracy mean here eval at features level
        dev_accuracy, test_accuracy = find_float_in_line(lines[-6]), find_float_in_line(lines[-5])

        test_features_accuracy, test_ed = find_2_floats_in_line(lines[-3])
        assert test_features_accuracy == test_accuracy

        test_graphemes_accuracy, test_graphemes_ed = find_2_floats_in_line(lines[-2])

    return datetime, lang, pos, training_mode, dev_accuracy, test_accuracy, test_ed, test_graphemes_accuracy, test_graphemes_ed


def main():
    results_folder = join('.', 'Results')
    io_format = 'f_p_attn'  # 'g_g' or 'f_f'
    analogy_types, seeds = ['None'], ['100', '42', '21'] # ['42']

    excel_results_file = f"Test-Results {'_'.join(analogy_types)}-{'_'.join(seeds)}-{io_format}.xlsx"

    run_records = []
    for analogy, seed in product(analogy_types, seeds):
        parent_folder = join(results_folder, f"{analogy}_{seed}_{io_format}")
        assert isdir(parent_folder), f"Folder Results doesn't exist!"

        folders = [join(parent_folder, f) for f in listdir(parent_folder)
                   if isdir(join(parent_folder, f)) and are_substrings_in_string(f[26:], (analogy, seed, io_format))]

        for folder in folders:
            datetime, lang, pos, training_mode, dev_accuracy, test_accuracy, test_ed, \
            test_graphemes_accuracy, test_graphemes_ed = extract_accuracies_from_single_folder(folder, io_format)
            run_records.append([datetime, analogy, seed, lang, pos, training_mode, io_format, dev_accuracy,
                                test_accuracy, test_ed, test_graphemes_accuracy, test_graphemes_ed])

    df = DataFrame(run_records, columns=['DateTime', 'AnalogyMode', 'Seed', 'Language', 'POS', 'Form/Lemma', 'IO mode',
                                         'Dev Accuracy', 'Test Accuracy', 'Test ED', 'Test Graphemes Accuracy',
                                         'Test Graphemes ED'])
    df = df.fillna("")
    df.to_excel(excel_results_file)


if __name__ == '__main__':
    main()
