#!/bin/bash

IN_DIR=$1
OUT_DIR=$2

for filename in "$IN_DIR"/*; do
    if [[ ! "$filename" =~ ".reduced" ]]; then
        snp_out=$(basename -- "${filename%.*}.vcf")

        # snp_snites
        if [ ! -f "$OUT_DIR/$snp_out" ]; then
            snp-sites -v -o "$OUT_DIR/$snp_out" "$filename"
        fi    
    fi
done
