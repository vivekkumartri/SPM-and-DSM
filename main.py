import sys
import yaml
import pandas as pd
from utils import *

def load_config(filename):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def save_results(dsm_result, spm_result, occurrence_matrix, dsm_result_path, spm_result_path, occurrence_matrix_path):
    dsm_result.to_csv(dsm_result_path, index=False)
    spm_result.to_csv(spm_result_path, index=False)
    occurrence_matrix.to_csv(occurrence_matrix_path, index=False)

def main():
    if len(sys.argv) > 1:
        config_file = 'config.yaml'
        command_args = sys.argv[1:]
        overridden_params = {}
        for arg in command_args:
            if '=' in arg:
                key, value = arg.split('=')
                overridden_params[key] = value
            else:
                print(f"Ignoring invalid argument: {arg}")

        config = load_config(config_file)

        # SPM parameters
        spm_params = config.get('SPM', {})
        for key, value in overridden_params.items():
            if key in spm_params:
                spm_params[key] = value

        spm_result, occurrence_matrix = SPM(spm_params)

        # DSM parameters
        dsm_params = config.get('DSM', {})
        for key, value in overridden_params.items():
            if key in dsm_params:
                dsm_params[key] = value

        ptrn_left, ptrn_right, ptrn_both_left, ptrn_both_right, dsm_result = DSM(dsm_params)

        # Fetching paths from config
        dsm_result_path = config.get('dsm_result_path')
        spm_result_path = config.get('spm_result_path')
        occurrence_matrix_path = config.get('occurrence_matrix_path')

        # Saving results
        save_results(dsm_result, spm_result, occurrence_matrix, dsm_result_path, spm_result_path, occurrence_matrix_path)

    else:
        config_file = 'config.yaml'
        config = load_config(config_file)

        # SPM parameters
        spm_params = config.get('SPM', {})

        spm_result, occurrence_matrix = SPM(spm_params)

        # DSM parameters
        dsm_params = config.get('DSM', {})

        ptrn_left, ptrn_right, ptrn_both_left, ptrn_both_right, dsm_result = DSM(dsm_params)

        # Fetching paths from config
        dsm_result_path = config.get('dsm_result_path')
        spm_result_path = config.get('spm_result_path')
        occurrence_matrix_path = config.get('occurrence_matrix_path')

        # Saving results
        save_results(dsm_result, spm_result, occurrence_matrix, dsm_result_path, spm_result_path, occurrence_matrix_path)

if __name__ == "__main__":
    main()
