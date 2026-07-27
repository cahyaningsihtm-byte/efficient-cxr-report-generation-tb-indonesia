# Data Availability

The study used de-identified chest X-ray/report pairs from Cipto Mangunkusumo Hospital (RSCM), accessed through BDC IMERI Idealab.

The dataset is not publicly distributed because its use is governed by institutional privacy, ethics, and data-use requirements.

Consequently, this repository does not contain medical images, radiology reports, patient or study identifiers, data manifests, fold assignments, or case-level predictions.

The released code is designed to be applied to a local dataset only when the user has the necessary ethics approval, data-governance approval, and technical safeguards. The absence of public data should not be interpreted as authorisation to reconstruct, redistribute, or seek access to the underlying clinical records.

## Example files

The [examples](examples/) directory contains two synthetic, format-only rows showing the expected CSV structure for local inference and evaluation preparation. These examples are not drawn from the RSCM/BDC dataset, are not part of the paper experiments, and must not be used to estimate model performance.

## Ethics approval

The study protocol was reviewed and approved by the Health Research Ethics Committee of the Faculty of Medicine, Universitas Indonesia - Cipto Mangunkusumo Hospital (ethical approval no. KET-68/UN2.F1/ETIK/PPM.00.02/2026; protocol no. 25-11-1726; approved 16 January 2026).

## Study scope

The reported benchmark is single-centre and uses automated clinical-semantic labelling of Indonesian reports. It does not establish external generalisability or clinical validity. The system is intended for human-verifiable reporting support rather than autonomous diagnosis.
