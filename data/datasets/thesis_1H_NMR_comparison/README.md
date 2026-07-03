## Thesis 1H NMR Comparison Inputs

This directory contains the only two dataset variants used for the thesis comparison of metadata extraction on the same underlying 1H NMR measurement.

- `1H_NMR_raw_bruker_10.zip`: native Bruker export preserved from the nested `10.zip` archive. This is the raw vendor-specific input containing the FID and associated acquisition, processing, and instrument files.
- `1H_NMR_chemspectra_export.zip`: ChemSpectra-derived package with the nested raw archive removed. This variant contains only the surrounding export artifacts (`10.edit.jdx`, `10.edit.png`, `10.infer.json`, `dataset_description.txt`).

The intended comparison is therefore:

1. extraction from the raw Bruker archive
2. extraction from the ChemSpectra export artifacts without the raw archive included
