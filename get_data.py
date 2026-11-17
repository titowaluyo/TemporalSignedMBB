"""Download the two public neural datasets used by neural_demo.py and neural_revision.py and verify
them against data_checksums.sha256.  Usage (from this folder):  python get_data.py

Sources (no registration; fixed commits so that the files cannot change):
  NetSim simulation 3, NumPy format: https://github.com/sakhanna/SRU_for_GCI  (data/netsim)
  EEGLAB tutorial recording:         https://github.com/sccn/eeglab           (sample_data)
"""
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
NETSIM = ("https://raw.githubusercontent.com/sakhanna/SRU_for_GCI/"
          "7fe4b24880a947056d7753cf51f69d5e4f0ee5fb/data/netsim/")
EEGLAB = ("https://raw.githubusercontent.com/sccn/eeglab/"
          "1f8488b81eaec671319938278ab59bace267d341/sample_data/")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    wanted = {}
    with open(os.path.join(HERE, "data_checksums.sha256")) as f:
        for line in f:
            if line.strip():
                digest, rel = line.split()
                wanted[rel] = digest
    bad = 0
    for i, (rel, digest) in enumerate(sorted(wanted.items()), 1):
        dest = os.path.join(HERE, *rel.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        name = rel.split("/")[-1]
        if not (os.path.exists(dest) and sha256(dest) == digest):
            url = (NETSIM if "/netsim/" in rel else EEGLAB) + name
            print(f"[{i}/{len(wanted)}] downloading {rel}", flush=True)
            tmp = dest + ".part"
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as out:
                out.write(r.read())
            os.replace(tmp, dest)
        if sha256(dest) != digest:
            print(f"CHECKSUM MISMATCH: {rel}")
            bad += 1
    if bad:
        sys.exit(f"{bad} file(s) failed the SHA-256 check")
    print(f"All {len(wanted)} data files are present and match data_checksums.sha256")


if __name__ == "__main__":
    main()
