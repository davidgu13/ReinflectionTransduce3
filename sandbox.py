from os import listdir
from os.path import join, isfile

results_folder = join('.', 'Results', 'None_42_f_f')

def get_runs_out_of_patience():
    valids_counter, invalids_counter, valid_runs, invalid_runs = 0, 0, [], []

    for sub_folder in listdir(results_folder):
        log_file = join(results_folder, sub_folder, 'f.log')
        if isfile(log_file):
            if len(open(log_file).readlines()) < 51:
                invalid_runs.append(sub_folder)
                invalids_counter += 1
            else:
                valid_runs.append(sub_folder)
                valids_counter += 1
        else:
            print(f"No f.log in {sub_folder}")

    valid_runs.sort()
    invalid_runs.sort()

    print("Valid runs:")
    for sub_folder in valid_runs:
        print(f'\t{sub_folder}')
    print(f'\t{valids_counter} valid runs.')

    print('Invalid runs:')
    for sub_folder in invalid_runs:
        print(f'\t{sub_folder}')
    print(f'\t{invalids_counter} invalid runs.')


if __name__ == '__main__':
    get_runs_out_of_patience()