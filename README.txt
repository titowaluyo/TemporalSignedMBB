Online Resource 1 (ESM_1.zip) - revised version

Article: Ground-truth benchmarking of signed, directed lagged connectivity: TemporalSignedMBB from a
         simulated gut-brain model to neural data
Journal: Cognitive Neurodynamics
Author:  Tito Waluyo Purboyo
Affiliation: Department of Biomedical Engineering, School of Electrical Engineering, Telkom University,
             Bandung 40257, Indonesia
Corresponding author e-mail: titowaluyo@telkomuniversity.ac.id

Caption: Complete TemporalSignedMBB code package (simulation model, analysis modules, unit tests,
entry-point scripts of the original submission and of the revision, neural-data scripts, result files
and figures) that reproduces every figure, table and number in the revised article.

Environment
  environment.yml         Python 3.13 (conda); installs the pinned pip packages below
  requirements.txt        pinned Python packages (Python itself is not a pip dependency)
  requirements-lingam.txt extra pinned packages for the DirectLiNGAM comparator (neural_revision.py);
                          install with: pip install --no-deps -r requirements-lingam.txt
  get_data.py             downloads the two public neural datasets into data/ and checks the checksums
  data_checksums.sha256   SHA-256 of the NetSim and EEGLAB files used (check: sha256sum -c data_checksums.sha256)

Package (tsmbb/)
  generator.py    simulation model; scenario options (revision) default to the original configuration,
                  so simulate(42) reproduces the original data bit for bit
  preprocess.py   QC, CLR (choice of omitted coordinate), cohort correction (pooled or training-only)
  te.py           conditional/bivariate Gaussian TE, cluster-robust and df-corrected edge tests,
                  LIONESS, lasso-VAR and bivariate comparators, KSG estimator
  topology.py     persistence, Wasserstein, density null, SPTS and variants, exact (MILP) frustration,
                  directed cycles, signed motif null, dynamic perturbation responses
  benchmark.py    one group-level Monte Carlo replicate (all group-level measures of the revision)
  classify.py     original subject-level code;  classify2.py  revised subject-level benchmark
  neuralmass.py   Wilson-Cowan network, volume conduction, Balloon-Windkessel model
  evaluate.py     DeLong AUROC, AUPRC, edge recovery measures

Revision entry points (run in this order; times on two CPU cores)
  python mc_group.py main 1001 30 2         group-level main scenario (about 20 min)
  python mc_group.py null 2001 50 2         inferential null
  python mc_group.py <scenario> 3001 20 2   correct, initdiag, tanh, lag2, glv, t3, driver, hetero,
                                            nonadd, weak, combined (see run_revision.sh)
  python seed42_revised.py                  illustrative dataset (Table 3, Figs. 1-3)
  python motif_mc.py 2                      signed motifs over 30 replications
  python clr_phi_checks.py 2                omitted CLR coordinate; phi stability; SPTS scales
  python lioness_check.py 2                 LIONESS vs heterogeneous subject-specific truth
  python comparators_mgb.py 10 2            external comparators on the microbiome benchmark
  python neural_mass_bench.py 20 2          Wilson-Cowan benchmark
  python neural_revision.py                 NetSim comparators and EEG revisions (needs data/, see below)
  python mc_subject.py <label> <seed> <R> 2 subject-level and design replications (see run_revision.sh)
  python aggregate_group.py && python aggregate_subject.py
  python fig1_design_rev.py && python make_figures_rev.py
  python -m pytest -q tests                 thirteen unit tests

Original entry points (unchanged; reproduce the original submission)
  run_all.py, design_factor.py, neural_demo.py, make_figures.py, make_fig7.py, fig1_design.py
  Note: run_all.py contains the group-membership error corrected in the revision (non-responder group
  networks pooled all MDD subjects); see Section 2.13 of the revised article.

Neural data (public, no registration; fetched by: python get_data.py)
  NetSim sim3 (NumPy format): https://github.com/sakhanna/SRU_for_GCI -> data/netsim/sim3_subject_*.npz
  EEGLAB tutorial data:       https://github.com/sccn/eeglab (sample_data) -> data/eeglab/eeglab_data.set/.fdt

Results (results/)
  mc/<scenario>/rep_<seed>.json, mc_subject/<label>/rep_<seed>.json   one file per replicate
  summary_group.json, summary_subject.json                            summaries read by the article
  main_rev.json, motif_mc.json, clr_phi_checks.json, lioness_check.json, comparators_mgb.json,
  neuralmass.json, neural_revision.json, neural.json                   other result files
  results_orig/                                                        result files of the original submission

License: MIT (see LICENSE). On acceptance the package will be archived on Zenodo with a DOI.
