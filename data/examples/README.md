# Synthetic Format Examples

This directory contains two synthetic rows that show the expected CSV shape for local use of the released scripts. The rows are not from the RSCM/BDC dataset, are not clinical records, and were not used in the paper experiments.

Files:

- `example_inputs.csv`: minimal input/reference columns expected by the Qwen inference and TB-14 evaluation preparation workflow.
- `example_model_outputs.csv`: example generated-report rows shaped like model output after inference.

No DICOM or chest X-ray image files are provided here. To run the full workflow, use an institutionally approved local DICOM directory outside this repository and replace the synthetic rows with approved local data.
