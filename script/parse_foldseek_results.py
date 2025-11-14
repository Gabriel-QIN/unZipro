import os
import pandas as pd

# https://alphafold.ebi.ac.uk/files/AF-A0A2Y9F5D4-F1-model_v4.pdb

def parse_mmseqs(m8_path, savepath, train_path, test_path, train_size=0.5, max_train_size=100, test_ratio=0.2, include_af2=True):
    m8_columns = "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,prob,evalue,bits,qlen,tlen,qaln,taln,tca,tseq,taxid,taxname".split(',')
    saved_cols = ['PDB_ID']
    All_PDB_list = []
    df = pd.read_csv(m8_path,sep='\t')
    df.columns = m8_columns
    df['PDB_ID'] = [a[:4]+a[22:].split(' ')[0].split('-')[0] for a in df['target'].tolist()]
    num_pdbs_before = len(df)
    num_pdbs = len(df)
    if train_size <= 1:
        train_size_pdb = int(len(df) * train_size)
        if train_size_pdb >= max_train_size * (1+test_ratio):
            train_size_pdb = max_train_size
        else:
            train_size_pdb = train_size_pdb
    else:
        train_size_pdb = int(train_size)
    sorted_df = df.sort_values(by='prob', ascending=False)
    final_df = sorted_df.copy()

    num_test_pdb = int(train_size_pdb * test_ratio)
    num_train_pdb = int(train_size_pdb)
    
    test_df = final_df.head(int(num_test_pdb))[saved_cols]
    remaining_df = final_df.iloc[:-num_test_pdb]
    train_df = remaining_df.tail(num_train_pdb)[saved_cols].copy()
    overlap = set(train_df['PDB_ID']).intersection(set(test_df['PDB_ID']))
    if overlap:
        print(f"Warning: Train/Test overlap detected, removing {len(overlap)} PDB(s) from train set!")
        train_df = train_df[~train_df['PDB_ID'].isin(overlap)]
    assert len(set(train_df['PDB_ID']).intersection(set(test_df['PDB_ID']))) == 0, f"Train/Test overlap detected: {overlap}"
    All_PDB_list.extend(train_df['PDB_ID'].tolist())
    All_PDB_list.extend(test_df['PDB_ID'].tolist())

    if len(train_df) < max_train_size:
        include_af2_train = True
        fill_train_numer = int(max_train_size - len(train_df))
    else:
        include_af2_train = False
    if len(test_df) < max_train_size * test_ratio:
        include_af2_test = True
        fill_test_number = int(max_train_size * test_ratio - len(test_df))
    else:
        include_af2_test = False
    num_pdbs = len(final_df)

    pdb_ids_af = []
    num_af_pdbs = 0
    num_af_pdbs_before = 0
    if include_af2_test or include_af2_train:
        try:
            af_m8_path = m8_path.replace('pdb100', 'afdb50')
            df_af = pd.read_csv(af_m8_path,sep='\t')
            df_af.columns = m8_columns
            sorted_df_af = df_af.sort_values(by='prob', ascending=False)
            num_af_pdbs_before = len(sorted_df_af)
            sorted_df_af = sorted_df_af.copy()
            num_af_pdbs = len(sorted_df_af)
            sorted_df_af['PDB_ID'] = [a.split(' ')[0].replace('AF-', '').replace('-F1-model_v4', '') for a in df_af['target'].tolist()]

            final_df_af = sorted_df_af.copy()
            if include_af2_test:
                test_af = final_df_af.head(int(fill_test_number))[saved_cols]
                remaining_af = final_df_af.iloc[:-fill_test_number]
                train_af = remaining_af.tail(fill_train_numer)[saved_cols].copy()
                All_PDB_list.extend(train_af['PDB_ID'].tolist())
                All_PDB_list.extend(test_af['PDB_ID'].tolist())
            else:
                train_af = final_df_af.tail(fill_train_numer)[saved_cols].copy()
                All_PDB_list.extend(train_af['PDB_ID'].tolist())
            num_af_pdbs = len(final_df_af)
        except Exception as e:
            print(f"Error in parsing AFDB m8 file {af_m8_path}: {e}")
            include_af2_train = False
            include_af2_test = False
            pass
            
    if include_af2_train:
        all_train_df = pd.concat([train_df, train_af], ignore_index=True)
    else:
        all_train_df = pd.concat([train_df], ignore_index=True)
    if include_af2_test:
        all_test_df = pd.concat([test_df, test_af], ignore_index=True)
    else:
        all_test_df = pd.concat([test_df], ignore_index=True)
    
    all_train_df.to_csv(train_path,sep='\t',index=False,header=False)
    all_test_df.to_csv(test_path,sep='\t',index=False,header=False)
    return all_train_df, all_test_df, All_PDB_list, num_pdbs, num_af_pdbs

def write_ids(datalist, savepath):
    with open(savepath, 'w') as f:
        for line in datalist:
            f.write(line)
            f.write('\n')

def write_list(datalist, savepath):
    with open(savepath, 'w') as f:
        for line in datalist:
            for x in line:
                f.write(f'{x}')
                f.write('\t')
            f.write('\n')

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Parse Foldseek results")
    parser.add_argument('--pdb', type=str, default="data/benchmark/Mutation/ProteinGym_pdbs.txt", help="PDB list (file path or comma-separated IDs)")
    parser.add_argument('--foldseek_dir', type=str, default="data/Foldseek_search", help="Result directory")
    parser.add_argument('--outdir', type=str, default="data/finetuned/", help="Output directory")
    parser.add_argument('--train_size', type=float, default=100, help="Train size")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if os.path.isfile(args.pdb):
        with open(args.pdb) as f:
            pdb_list = [line.strip() for line in f if line.strip()]
    else:
        pdb_list = [p.strip() for p in args.pdb.split(",") if p.strip()]

    all_homolog_info = []
    all_pdbids = []
    for pdb_name in pdb_list:
        try:
            m8_path = os.path.join(args.foldseek_dir, f'{pdb_name}/alis_pdb100.m8')
            savepath = os.path.join(args.outdir, f'{pdb_name}_pdb100.csv')
            pdb_dir = os.path.join(args.outdir, f'{pdb_name}/')
            os.makedirs(pdb_dir, exist_ok=True)
            train_path = os.path.join(pdb_dir, f'train.csv')
            test_path = os.path.join(pdb_dir, f'test.csv')
            train, test, homolog_PDB_list, num_pdbs, num_af_pdbs = parse_mmseqs(m8_path, savepath, train_path, test_path, train_size=args.train_size, include_af2=True)
            all_pdbids.extend(homolog_PDB_list)
            all_homolog_info.append([pdb_name, num_pdbs, num_af_pdbs])
            if len(train) < 10 or len(test) < 10:
                raise ValueError(f"Too little homologs for {pdb_name} Train size {len(train)} | Test size {len(test)}!")
            if len(train) < 100 or len(test) < 20:
                print(f'Warning! [{pdb_name}]: Train size {len(train)} | Test size {len(test)}')
        except Exception as e:
            # all_homolog_info.append([pdb_name, num_pdbs, num_af_pdbs])
            print(f'Error in {pdb_name}: {e}')

    write_list(all_homolog_info, os.path.join(args.outdir, 'Homolog_info.txt'))
    write_ids(all_pdbids, os.path.join(args.outdir, 'PDB_IDs.txt'))
