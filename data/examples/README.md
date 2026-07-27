# Public Example Data

This directory is reserved for small public examples that demonstrate the expected input and output formats for the released scripts.

Files:

- `example_inputs.csv`: minimal synthetic input/reference columns expected by the Qwen inference and TB-14 evaluation preparation workflow.
- `example_model_outputs.csv`: synthetic generated-report rows shaped like model output after inference. These synthetic examples are English-only because they document the preprocessed evaluation format rather than the original Indonesian reporting workflow.
- `deidentified_example_inputs.csv`: two de-identified real example rows linked to the public PNG examples. These rows are provided for format testing only and were not used to reproduce the paper metrics.
- `images/`: approved de-identified chest X-ray example images exported as metadata-stripped PNG files.
- `reports/`: approved de-identified example reports stored with source identifiers removed. These files include the Indonesian report text and the prepared English text when available in the approved local metadata.
- `public_examples_manifest.csv`: manifest linking released public example IDs to image/report files.

Public examples must not contain original patient identifiers, record numbers, DICOM metadata, acquisition dates, institution-specific file paths, or free-text information that could identify an individual. DICOM files should not be committed to this repository; approved examples should be exported as metadata-stripped image files such as PNG.

To run the full workflow on the private study data, use an institutionally approved local DICOM directory outside this repository and replace the synthetic rows with approved local data.
