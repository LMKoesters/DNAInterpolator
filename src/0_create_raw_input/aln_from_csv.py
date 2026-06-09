from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
from pathlib import Path


def aln_from_csv(input_csv: str | Path, genus: str, out_parent: str | Path) -> None:
    """
    Method for generation of alignment files from input csv

    Args:
        input_csv: The input csv with genetic data from which to generate alignment files
        genus: The current genus
        out_parent: The parent data directory
    """
    recs = pd.read_csv(input_csv, header=0, dtype={"gene_id": str})

    out_dir = f"{out_parent}/no_interpolation/raw_sorted_files/{genus}/alignments"
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    for locus, grouped_recs in recs.groupby("gene_id"):
        alignment = []
        for _, rec in grouped_recs.iterrows():
            total_record = SeqRecord(
                Seq(rec["seq"]),
                id=rec["individual_id"],
                description="",
            )
        
            alignment.append(total_record)
    
        # write fasta entries
        alignment_file = f"{out_dir}/{locus}.fasta"
        with open(alignment_file, "w+") as conc_aln_f:
            for seq in alignment:
                SeqIO.write(seq, conc_aln_f, "fasta")
    
