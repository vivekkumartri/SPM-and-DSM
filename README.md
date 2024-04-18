# Course Project README

## Overview

This project is part of the Course Project for ET623. It focuses on implementing Sequential Pattern Mining (SPM) and Differential Pattern Mining (DSM) in Python.

## Requirements

The following libraries are required to run this code:
- pandas
- numpy
- yaml
- scipy

You can install these libraries using pip:
```bash
pip install pandas numpy PyYAML scipy
```

## Configuration

You can customize the behavior of the Sequential Pattern Mining (SPM) and Differential Pattern Mining (DSM) algorithms by modifying the `config.yaml` file.

### SPM Configuration

```yaml
SPM:
  path_to_csv: "fixed.csv"
  identifier_column: "Identifier"
  sequence_column: "Sequence"
  sortby: "S-Support"
  sliding_window_min: 1
  sliding_window_max: 4
  min_gap: 1
  max_gap: 12
  S_support_thresh: 0.4
  I_support_thresh: 0
  dataset_format: 0
```
### DSM Configuration

```yaml
DSM:
  path_to_csv_left: "fixed.csv"
  path_to_csv_right: "growth.csv"
  identifier_column: "Identifier"
  sequence_column: "Sequence"
  sortby: "S-Support"
  sliding_window_min: 1
  sliding_window_max: 1
  min_gap: 1
  max_gap: 12
  S_support_thresh: 0.4
  I_support_thresh: 0
  threshold_pvalue: 0.1
  dataset_format: 0
  test_type: "ttest_ind"
```

### Output Paths

```yaml
dsm_result_path: "/home/vivek.trivedi/ET623_project/dsm_result.csv"
spm_result_path: "/home/vivek.trivedi/ET623_project/spm_result.csv"
occurrence_matrix_path: "/home/vivek.trivedi/ET623_project/occurrence_matrix.csv"
```

## Running the Code

To run the code with the default configuration from `config.yaml`, execute the following command:

```bash
python main.py
```

Alternatively, you can specify configuration parameters at runtime by providing them after `main.py`:

```bash
python main.py <path_to_csv> <max_gap> <min_gap> <sortby> ...
```

Replace `<path_to_csv>`, `<max_gap>`, `<min_gap>`, `<sortby>`, etc., with your desired values.

## Demo

A demo notebook `demo.ipynb` is provided for a quick overview of the project. Additionally, a demo is available on Hugging Face's space for a hands-on experience. You can access the Gradio interface running on Hugging Face Space [here](https://huggingface.co/spaces/vivek9/ET_623_Project).

---

