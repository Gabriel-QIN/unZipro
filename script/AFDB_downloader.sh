#!/bin/bash
# Download PDB files from a given PDB ID list, skip if exists

echo "Please input your filename of the PDB list for downloading."
cat "$1" | while read pdb
do
    outfile="PDB_structures/${pdb}.pdb"

    if [ -f "$outfile" ]; then
        echo "File $outfile already exists, skipping."
    else
        echo "Downloading $pdb ..."
        wget -c "https://alphafold.ebi.ac.uk/files/AF-${pdb}-F1-model_v4.pdb" -O "$outfile"
    fi
done