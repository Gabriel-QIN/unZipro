mapped = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
          'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
          'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
          'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X':20}
res_map = dict(zip(mapped.values(), mapped.keys()))
mapped_20 = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
          'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
          'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
          'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
# 3-letter code to 1-letter code
three2one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU':'E', 'PHE': 'F',
             'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
             'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
             'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', 'UNK': 'X'}
one2three = dict(zip(three2one.values(), three2one.keys()))

atom2id = {'N':0, 'CA':1, 'C':2, 'O':3, 'CB':4, 'H':5}
id2atom = ['N', 'CA', 'C', 'O', 'CB', 'H']
