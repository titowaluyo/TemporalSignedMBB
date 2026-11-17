Online Resource 1 (ESM_1.zip)

Article: Recovering signed, directed and time-varying effective connectivity against a known ground
         truth: TemporalSignedMBB from a gut-brain model of depression to fMRI and EEG
Journal: Cognitive Neurodynamics
Author:  Tito Waluyo Purboyo
Affiliation: Department of Biomedical Engineering, School of Electrical Engineering, Telkom University,
             Bandung 40257, Indonesia
Corresponding author e-mail: titowaluyo@telkomuniversity.ac.id

Caption: Complete TemporalSignedMBB code package (simulation model, analysis modules, unit tests,
entry-point scripts, neural-data scripts, result files and figures) that reproduces every figure,
table and number in the article.

Contents
  tsmbb/generator.py    simulation model (Section 2.2)
  tsmbb/preprocess.py   QC, CLR, cohort correction, demeaning, transitions
  tsmbb/te.py           conditional and bivariate Gaussian TE, LIONESS, KSG estimator (2.3, 2.8)
  tsmbb/topology.py     persistence, Wasserstein, density null, SPTS, frustration, resilience, motifs
  tsmbb/classify.py     subject-level features, models and comparators (2.8-2.10)
  tsmbb/evaluate.py     DeLong AUROC, AUPRC, edge recovery measures
  run_all.py            main MGB analysis           -> results/main.json, main_arrays.npz
  design_factor.py      design-factor experiment    -> results/design_factor.json
  neural_demo.py        NetSim and EEGLAB analyses  -> results/neural.json, neural_arrays.npz
  fig1_design.py        Fig. 1 drawn from the simulation model (600 dpi)
  make_figures.py       Figs. 1-6 (600 dpi; calls fig1_design.py) -> figures/
  make_fig7.py          Fig. 7 (600 dpi)             -> figures/
  tests/test_core.py    seven unit tests (python -m pytest -q tests)
  results/              result files read by the article, with run logs
  requirements.txt      software versions used

Reproduction (from this folder)
  pip install -r requirements.txt
  python run_all.py          # about 25 min on two CPU cores
  python design_factor.py    # about 22 min
  # neural data (public, no registration):
  #   NetSim sim3 (NumPy format): https://github.com/sakhanna/SRU_for_GCI  -> data/netsim/sim3_subject_*.npz
  #   EEGLAB tutorial data:       https://github.com/sccn/eeglab (sample_data) -> data/eeglab/eeglab_data.set/.fdt
  python neural_demo.py      # about 7 min
  python make_figures.py && python make_fig7.py

License: MIT (see LICENSE). On acceptance the package will be archived on Zenodo with a DOI.
