# Thesis Data Analysis Scripts

This repository contains the Python analysis scripts used for the CPP/RNA delivery dataset in the thesis supplementary material. The literature-analysis scripts are stored in `cpp_literature_analysis/`.

## Workflow

The scripts are organized by analysis task:

- `cpp_literature_analysis/T1A.py`: peptide backbone usage, diversity, concentration, temporal trend, and longevity analysis.
- `cpp_literature_analysis/T1B.py`: peptide modification signatures and backbone-modification strategy analysis.
- `cpp_literature_analysis/T1C.py`: sequence-level backbone similarity analysis and dendrogram generation.
- `cpp_literature_analysis/T2.py`: delivery pipeline normalization and backfilling.
- `cpp_literature_analysis/T3A.py`: logistic regression analysis using sequence-derived and modification-derived features.
- `cpp_literature_analysis/T3B.py`: PCA and UMAP visualization of the feature matrix.
- `cpp_literature_analysis/T3C.py`: cross-validated logistic regression analysis and coefficient stability visualization.
- `cpp_literature_analysis/T4.py`: top-10 CPP backbone stage-specific success probability analysis.

## Input Files

The scripts expect processed local input files with the following names. To reproduce the workflow, place these files in the working directory from which the scripts are run, for example `cpp_literature_analysis/`.

- `data.xlsx`
- `data2.csv`
- `data2_with_mod_signature.csv`
- `data_backfilled.xlsx`

The data files are not included in this repository. They may contain curated literature annotations and should be shared only according to the data availability policy of the thesis/manuscript.

## Main Processing Order

1. `T1A.py` reads `data.xlsx` and generates backbone-level summary tables and figures.
2. `T1B.py` reads `data2.csv` and generates `data2_with_mod_signature.csv`.
3. `T2.py` reads `data2_with_mod_signature.csv` and generates `data_backfilled.xlsx`.
4. `T3A.py`, `T3B.py`, and `figures/T3C.py` read `data_backfilled.xlsx`.
5. `T4.py` reads `data_backfilled.xlsx` for top-10 CPP stage-specific success probability analysis.

## Notes

For stage-specific success probabilities in `T4.py`, the denominator excludes missing or not assessed entries. The numerator is the number of entries annotated as successful for the corresponding stage, and the denominator is the number of entries annotated as either success or failure.

The sequence-derived `net_charge_proxy` used in the T3 analyses is defined as:

```text
K + R + H - D - E
```

This is a simple sequence-derived proxy and does not include pH-dependent effects, terminal charges, lipid charge, or non-canonical residues unless explicitly encoded in the sequence features.
