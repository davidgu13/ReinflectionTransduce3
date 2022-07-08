# This is a Python3 script for extracting the runs results to an Excel file, including final dev and test accuracies.
import os
import re
from itertools import product
from os.path import basename, isdir, join

def are_substrings_in_string(target_string: str, substrings: tuple) -> bool:
    return all([substring in target_string for substring in substrings])

def find_float_in_line(line):
    matches = re.findall("\d+\.\d+", line)
    assert len(matches) == 1, "Invalid output!"
    return float(matches[0])

results_folder = join('.', 'Results')
analogy_types, seeds, io_formats  = ['None'], ['100', '42', '21'], ['g_g'] # to be extended

lang_pos_groups = dict(zip(range(1,16), ['kat_V', 'kat_N', 'fin_ADJ', 'swc_V', 'swc_ADJ', 'fin_V', 'sqi_V', 'hun_V',
                                         'bul_V', 'bul_ADJ', 'lav_V', 'lav_N', 'tur_V', 'tur_ADJ', 'fin_N']))
excel_results_file = f"Test-Results {'_'.join(analogy_types)}-{'_'.join(seeds)}-{'_'.join(io_formats)}.xlsx"

def extract_accuracies_from_single_folder(folder):
    # The folder has the format: Outputs_YYYY-MM-DD hhmmss_lang_pos_training-mode_IO_analogy_seed
    datetime, lang, pos, training_mode = basename(folder).split("_")[1:5]

    lines = open(join(folder, 'f.stats'), encoding='utf8').read().split('\n')
    dev_accuracy, test_accuracy = find_float_in_line(lines[-3]), find_float_in_line(lines[-2])
    return datetime, lang, pos, training_mode, dev_accuracy, test_accuracy


def main():
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

if __name__ == '__main__':
    main()