# data/

The two public neural datasets are not redistributed here. Download them with

    python get_data.py

which creates `data/netsim/sim3_subject_0.npz … sim3_subject_49.npz` and
`data/eeglab/eeglab_data.set`, `eeglab_data.fdt`, and verifies every file against
`data_checksums.sha256`. Only `neural_demo.py` and `neural_revision.py` need these files.
