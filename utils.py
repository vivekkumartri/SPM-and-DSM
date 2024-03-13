import pandas as pd
import numpy as np
from scipy import stats
import csv

def process_csv(input_file, output_file="output.csv",user_id_column="user_id",timestamp_column="timestamp",action_column="actions"):

    with open(input_file, 'r', newline='') as csvfile, open(output_file, 'w', newline='') as new_csvfile:
        reader = csv.reader(csvfile)
        writer = csv.writer(new_csvfile)

        # Get the column indices for user_id, timestamp, and action
        user_id_index, timestamp_index, action_index = None, None, None
        for i, row in enumerate(reader):
            if i == 0:  # Header row
                user_id_index = row.index(user_id_column)
                timestamp_index = row.index(timestamp_column)
                action_index = row.index(action_column)
                break

        # Read the rest of the data and store it in a dictionary
        user_actions = {}
        for row in reader:
            user_id = row[user_id_index]
            timestamp = row[timestamp_index]
            action = row[action_index]

            if user_id not in user_actions:
                user_actions[user_id] = []

            # Sort actions by timestamp in ascending order
            user_actions[user_id].append((timestamp, action))

        # Write the data to the output CSV file with the new column
        writer.writerow([user_id_column, 'actions'])  # Write header with new column
        for user_id, actions in user_actions.items():
            sorted_actions = actions  # Sort by timestamp
            concatenated_actions = ';'.join([action for timestamp, action in sorted_actions])
            writer.writerow([user_id,  concatenated_actions])


def generate_sequence_list(sentence, min_gap,max_gap, sliding_window_min=1,sliding_window_max=1):
    # Split the sentence into words
    words = sentence.split(";")

    # Generate n-grams
    ngrams = []
    for sliding_window in range(sliding_window_min,sliding_window_max+1):
      for gram_length in range(min_gap,max_gap + 1):  # Loop from 1 to n
          for i in range(0, len(words) - gram_length + 1, sliding_window):
              ngram = '--->'.join(words[i:i+gram_length])
              ngrams.append(ngram)

    return ngrams
    
    
    
def create_dict_from_df(df, identifier_column, sequence,min_gap,max_gap,sliding_window_min=1,sliding_window_max=1):
    result_dict = {}
    unique_values_set = set()  # Initialize set to store unique values
    for index, row in df.iterrows():
        key = row[identifier_column]
        values = generate_sequence_list(row[sequence],min_gap,max_gap,sliding_window_min,sliding_window_max)
        result_dict[key] = values
        unique_values_set.update(values)  # Update the set with unique values
    return result_dict, unique_values_set
    
def create_dataframe_from_dict_and_set(result_dict, unique_values_set):
    # Initialize an empty dictionary to store counts
    counts_dict = {}
    # Iterate over the set
    for value in unique_values_set:
        counts_dict[value] = {}
        # Iterate over the keys in the result_dict
        for key, values in result_dict.items():
            counts_dict[value][key] = values.count(value)

    # Create a DataFrame from the counts dictionary
    df = pd.DataFrame(counts_dict).fillna(0)
    # Transpose the DataFrame so that keys become columns and values become rows
    df = df.transpose()

    return df
    
    
def process_dataframe(df):
    # Calculate num_student
    num_student = len(df.columns)

    # Calculate I-Frequency and S-Frequency
    I_Frequency = df.sum(axis=1)
    S_Frequency = (df > 0).sum(axis=1)

    # Create a dictionary for new data
    new_data = {
        'I-Frequency': I_Frequency,
        'S-Frequency': S_Frequency
    }

    # Create a DataFrame from the new data
    new_df = pd.DataFrame(new_data)

    # Calculate I-Support by dividing I-Frequency with num_student
    new_df['I-Support (mean)'] = new_df['I-Frequency'] / num_student
    new_df['S-Support'] = new_df['S-Frequency'] / num_student

    # Calculate standard deviation of each row
    new_df['I-Support (sd)'] = df.std(axis=1)

    return new_df


def calculate_p_value(test_type, vector_a, vector_b=None, **kwargs):
    """
    Calculate the p-value for different types of t-tests.

    Parameters:
        test_type (str): Type of test to perform.
        vector_a (array-like): Data for sample A.
        vector_b (array-like, optional): Data for sample B (only required for some tests).
        **kwargs: Additional keyword arguments required for specific tests.

    Returns:
        p_value (float): The p-value obtained from the test.
    """
    if test_type == 'poisson_means_test':
        # Poisson means test
        result = stats.poisson_means_test(vector_a, vector_b, **kwargs)
    elif test_type == 'ttest_ind':
        # T-test for the means of two independent samples
        result = stats.ttest_ind(vector_a, vector_b, **kwargs)
    elif test_type == 'mannwhitneyu':
        # Mann-Whitney U rank test on two independent samples
        result = stats.mannwhitneyu(vector_a, vector_b, **kwargs)
    elif test_type == 'bws_test':
        # Baumgartner-Weiss-Schindler test on two independent samples
        result = stats.bws_test(vector_a, vector_b, **kwargs)
    elif test_type == 'ranksums':
        # Wilcoxon rank-sum statistic for two samples
        result = stats.ranksums(vector_a, vector_b, **kwargs)
    elif test_type == 'brunnermunzel':
        # Brunner-Munzel test on samples
        result = stats.brunnermunzel(vector_a, vector_b, **kwargs)
    elif test_type == 'mood':
        # Mood's test for equal scale parameters
        result = stats.mood(vector_a, vector_b, **kwargs)
    elif test_type == 'ansari':
        # Ansari-Bradley test for equal scale parameters
        result = stats.ansari(vector_a, vector_b, **kwargs)
    elif test_type == 'cramervonmises_2samp':
        # Two-sample Cramér-von Mises test for goodness of fit
        result = stats.cramervonmises_2samp(vector_a, vector_b, **kwargs)
    elif test_type == 'epps_singleton_2samp':
        # Epps-Singleton (ES) test statistic
        result = stats.epps_singleton_2samp(vector_a, vector_b, **kwargs)
    elif test_type == 'ks_2samp':
        # Two-sample Kolmogorov-Smirnov test for goodness of fit
        result = stats.ks_2samp(vector_a, vector_b, **kwargs)
    elif test_type == 'kstest':
        # One-sample or two-sample Kolmogorov-Smirnov test for goodness of fit
        result = stats.kstest(vector_a, vector_b, **kwargs)
    else:
        raise ValueError("Invalid test type.")

    # Get the p-value
    p_value = result.pvalue
    return p_value

def SPM_(path_to_csv,dataset_format, identifier_column, sequence_column,sortby="S-Support",min_gap=1,max_gap=1,sliding_window_min=1,sliding_window_max=1,S_support_thresh=0,I_support_thresh=0,timestamp_column="timestamp"):

    if dataset_format==1:
      process_csv(path_to_csv, output_file="output.csv",user_id_column=identifier_column,timestamp_column=timestamp_column,action_column=sequence_column)
      path_to_csv="output.csv"
    # Read CSV file
    data = pd.read_csv(path_to_csv)

    # Create dictionary from DataFrame
    data_seq, corpus = create_dict_from_df(data, identifier_column, sequence_column, min_gap,max_gap,sliding_window_min,sliding_window_max)

    # Create occurrence matrix
    occurence_matrix = create_dataframe_from_dict_and_set(data_seq, corpus)

    # Process occurrence matrix
    spm_result = process_dataframe(occurence_matrix)
    spm_result = spm_result.sort_values(by=sortby, ascending=False)

    return spm_result[(spm_result['S-Support'] > S_support_thresh) & (spm_result['I-Support (mean)'] > I_support_thresh)], occurence_matrix
    


def SPM(config):
    path_to_csv = config.get('path_to_csv')
    dataset_format = config.get('dataset_format')
    identifier_column = config.get('identifier_column')
    sequence_column = config.get('sequence_column')
    sortby = config.get('sortby', "S-Support")
    min_gap = config.get('min_gap', 1)
    max_gap = config.get('max_gap', 1)
    sliding_window_min = config.get('sliding_window_min', 1)
    sliding_window_max = config.get('sliding_window_max', 1)
    S_support_thresh = config.get('S_support_thresh', 0)
    I_support_thresh = config.get('I_support_thresh', 0)
    timestamp_column = config.get('timestamp_column', "timestamp")

    return SPM_(path_to_csv,dataset_format, identifier_column, sequence_column,sortby,min_gap,max_gap,sliding_window_min,sliding_window_max,S_support_thresh,I_support_thresh,timestamp_column)



def DSM(config):
    path_to_csv_left = config['path_to_csv_left']
    dataset_format = config['dataset_format']
    path_to_csv_right = config['path_to_csv_right']
    identifier_column = config['identifier_column']
    sequence_column = config['sequence_column']
    sortby = config['sortby']
    min_gap = config['min_gap']
    max_gap = config['max_gap']
    sliding_window_min = config['sliding_window_min']
    sliding_window_max = config['sliding_window_max']
    S_support_thresh = config['S_support_thresh']
    I_support_thresh = config['I_support_thresh']
    threshold_pvalue = config['threshold_pvalue']
    test_type = config['test_type']
    timestamp_column = config.get('timestamp_column', 'timestamp')

    if dataset_format == 1:
        process_csv(path_to_csv_left, output_file="output_left.csv", user_id_column=identifier_column,
                    timestamp_column=timestamp_column, action_column=sequence_column)
        path_to_csv_left = "output_left.csv"
        process_csv(path_to_csv_right, output_file="output_right.csv", user_id_column=identifier_column,
                    timestamp_column=timestamp_column, action_column=sequence_column)
        path_to_csv_left = "output_right.csv"

    ptrn_left = []
    ptrn_right = []
    ptrn_both_left = []
    ptrn_both_right = []

    spm_result_left, occurence_matrix_left = SPM_(path_to_csv_left, 0, identifier_column, sequence_column, sortby,
                                                 min_gap, max_gap, sliding_window_min, sliding_window_max,
                                                 S_support_thresh, I_support_thresh)
    spm_result_right, occurence_matrix_right = SPM_(path_to_csv_right, 0, identifier_column, sequence_column, sortby,
                                                   min_gap, max_gap, sliding_window_min, sliding_window_max,
                                                   S_support_thresh, I_support_thresh)

    result_data = []
    all_ptrn = set(spm_result_left.index)
    all_ptrn.update(spm_result_right.index)
    left_ptrn_data = set(spm_result_left.index)
    right_ptrn_data = set(spm_result_right.index)

    for ptrn in all_ptrn:
        isupport_left = occurence_matrix_left.loc[ptrn, :].values if ptrn in spm_result_left.index else np.zeros(
            occurence_matrix_left.shape[1])
        isupport_right = occurence_matrix_right.loc[ptrn, :].values if ptrn in spm_result_right.index else np.zeros(
            occurence_matrix_right.shape[1])
        p_value = calculate_p_value(test_type, isupport_left, isupport_right)
        if p_value < threshold_pvalue:
            if (ptrn in left_ptrn_data) and (ptrn in right_ptrn_data):
                if isupport_left.mean() > isupport_right.mean():
                    ptrn_both_left.append(ptrn)
                    result_data.append((ptrn, p_value, isupport_left.mean(), isupport_right.mean(), "both_left"))
                else:
                    ptrn_both_right.append(ptrn)
                    result_data.append((ptrn, p_value, isupport_left.mean(), isupport_right.mean(), "both_right"))
            else:
                if ptrn in left_ptrn_data:
                    ptrn_left.append(ptrn)
                    result_data.append((ptrn, p_value, isupport_left.mean(), np.nan, "left"))
                else:
                    ptrn_right.append(ptrn)
                    result_data.append((ptrn, p_value, np.nan, isupport_right.mean(), "right"))

    result_df = pd.DataFrame(result_data,
                             columns=['ptrn', 'ttest_value', 'isupportleft_mean', 'isupportright_mean', "Group"])
    return ptrn_left, ptrn_right, ptrn_both_left, ptrn_both_right, result_df
