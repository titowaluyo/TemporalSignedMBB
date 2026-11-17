#!/bin/bash
# Complete revision pipeline (order used for the article). Neural data must be in data/ (see ESM_1_README.txt).
set -e
python3 mc_group.py main 1001 30 2
python3 mc_group.py null 2001 50 2
for s in correct initdiag tanh lag2 glv t3 driver hetero nonadd weak combined; do python3 mc_group.py $s 3001 20 2; done
python3 seed42_revised.py
python3 motif_mc.py 2
python3 clr_phi_checks.py 2
python3 lioness_check.py 2
python3 comparators_mgb.py 10 2
python3 neural_mass_bench.py 20 2
python3 neural_revision.py
python3 mc_subject.py cc_main 5001 8 2
python3 mc_subject.py rp_early 5401 6 2
python3 mc_subject.py cc_coupling 5101 6 2
python3 mc_subject.py cc_calib 5201 6 2
for v in 5 10 20; do for c in 1.0 0.7; do python3 mc_subject.py design_v${v}_c${c} 5301 5 2; done; done
python3 aggregate_group.py
python3 aggregate_subject.py
python3 gen_supp.py
python3 fig1_design_rev.py
python3 make_figures_rev.py
python3 -m pytest -q tests
