from glob import glob
from pathlib import Path
import subprocess


def calc_distances(genus: str, out_parent: str | Path) -> None:
    """
    Parent method for RAxML distance calculation

    Args:
        genus: The current genus
        out_parent: Parent data directory
    """
    out_dir = f"{out_parent}/no_interpolation/raw_sorted_files/{genus}/RAxML"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for f in glob(f"{out_parent}/no_interpolation/raw_sorted_files/{genus}/alignments/*.fasta"):
        run_raxml(f, out_dir)
        

def run_raxml(aln_f: str, out_dir: str | Path) -> None:
    """
    Runs RAxML for GTRGAMMA distance calculation outside the script.

    Args:
        aln_f: The alignment file to be used as input for RAxML
        out_dir: Parent output directory for RAxML directory
    """
    raxml_cmd = f'raxmlHPC -f x -s "{aln_f}" -m GTRGAMMA -n "{Path(aln_f).stem}" -p 12345 -T 2' \
                f' -w "$(realpath "{out_dir}")"'
    subprocess.call(raxml_cmd, shell=True)
    