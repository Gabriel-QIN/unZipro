#!/bin/bash
#Download PDB file from given PDB id list.

echo  "please input your filename of PDB list for downloading."
cat $1|while read pdb
do
	# wget -c https://files.rcsb.org/pub/pdb/data/structures/all/pdb/pdb$pdb.ent.gz
	wget -c 	https://files.rcsb.org/download/$pdb.pdb
	done
