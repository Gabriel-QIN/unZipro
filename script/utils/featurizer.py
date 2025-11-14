import os
import torch
import numpy as np
import pandas as pd
from .commons import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_protein_coords(file, file_format='pdb'):
    """
    Read a protein structure and output backbone coordinates.

    Parameters
    ----------
    file : str
        Path to the PDB or CSV file.
    file_format : str
        'pdb' or 'csv'.

    Returns
    -------
    coords : np.ndarray
        Array of shape (num_residues, 6, 3) containing backbone atom coordinates
        (N, CA, C, O, CB, H). Missing atoms are filled with zeros.
    resnames : list
        List of residue names for each residue.
    iaa2org : list
        List of original chain+residue identifiers.
    """
    atom2id = {'N':0, 'CA':1, 'C':2, 'O':3, 'CB':4, 'H':5}
    id2atom = ['N', 'CA', 'C', 'O', 'CB', 'H']

    # Read PDB
    if file_format == 'pdb':
        lines = open(file).read().splitlines()
        # Count residues
        naa = 0
        org2iaa = {}
        for l in lines:
            header, atomtype, resname, chain, iaa_org = l[0:4], l[12:16].strip(), l[17:20], l[21:22], l[22:27]
            if not (header == "ATOM" and atomtype == 'CA'):
                continue
            org2iaa[(chain+iaa_org)] = naa
            naa += 1

        coords = np.zeros((naa, len(atom2id), 3), dtype=float)
        exists = np.zeros((naa, len(atom2id)), dtype=bool)
        resnames = ['NAN']*naa
        iaa2org = ['A0000']*naa

        for l in lines:
            header, atomtype, resname, chain, iaa_org = l[0:4], l[12:16].strip(), l[17:20], l[21:22], l[22:27]
            if not header == "ATOM": continue
            if atomtype not in atom2id: continue
            org_key = chain+iaa_org
            iaa = org2iaa.get(org_key)
            if iaa is None: continue
            id_atom = atom2id[atomtype]
            coords[iaa][id_atom] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
            exists[iaa][id_atom] = True
            resnames[iaa] = resname
            iaa2org[iaa] = org_key

    elif file_format == 'csv':
        pdbdf = pd.read_csv(file, sep='\t')
        org_index_list = []
        naa = pdbdf["residue_number"].nunique()
        new_iaa = 0
        org2iaa = {}
        for _, row in pdbdf.iterrows():
            atomtype, resname, chain, res_iaa_org = row[3], row[4], row[5], row[6]
            org_key = f"{chain} {res_iaa_org}"
            if org_key not in org2iaa:
                org_index_list.append(res_iaa_org)
                org2iaa[org_key] = new_iaa
                new_iaa += 1

        coords = np.zeros((naa, len(atom2id), 3), dtype=float)
        exists = np.zeros((naa, len(atom2id)), dtype=bool)
        resnames = ['NAN']*naa
        iaa2org = ['A0000']*naa
        all_atom_set = set()

        for _, row in pdbdf.iterrows():
            atomtype, resname, chain, res_iaa_org = row[3], row[4], row[5], row[6]
            if atomtype not in atom2id: continue
            org_key = f"{chain} {res_iaa_org}"
            iaa = org2iaa.get(org_key)
            tmp_id = f"{org_key} {atomtype}"
            if tmp_id in all_atom_set: continue
            if iaa is None: continue
            id_atom = atom2id[atomtype]
            coords[iaa][id_atom] = [float(row[7]), float(row[8]), float(row[9])]
            exists[iaa][id_atom] = True
            resnames[iaa] = resname
            iaa2org[iaa] = org_key
            all_atom_set.add(tmp_id)

    return coords, resnames, iaa2org

def calc_distmat(coord, atomtype='CB', device='cuda'):
    """
    Get distance matrix of atomtype using torch, supports GPU.
    
    Args:
        atomtype: str, atom type to calculate (e.g., 'CB')
        device: 'cuda' or 'cpu'
    Returns:
        distance matrix of shape (naa, naa) as torch.Tensor
    """
    # get coordinates of the atom type, shape [naa, 3]
    points = coord[:, atom2id[atomtype], :].to(device)  # move to GPU if needed

    # compute pairwise differences with broadcasting, shape [naa, naa, 3]
    diff = points.unsqueeze(0) - points.unsqueeze(1)

    # compute squared distances and sum over last axis
    dist_squared = torch.sum(diff ** 2, dim=2)

    # sqrt to get Euclidean distances
    distmat = torch.sqrt(dist_squared)
    return distmat

def get_nearestN(coords, N, atomtype='CB', rm_self=True):
    """
    Compute the N nearest neighbors for each residue (or atom) in a protein structure.

    Args:
        coords (np.ndarray): A 3D coordinate tensor of shape (n_res, n_atom, 3),
                             where n_res is the number of residues.
        N (int): Number of nearest neighbors to retrieve.
        atomtype (str): The atom type used for distance calculation (e.g., 'CA', 'CB').
        rm_self (bool): If True, remove the self (i,i) pair from the neighbor list.

    Returns:
        np.ndarray: A matrix of shape (n_res, N) containing the indices of the
                    N nearest neighbors for each residue.
    """
    # get a point matrix with the shape [naa, 3] which is coordinates of atomtype (e.g. CA)
    points = coords[:,atom2id[atomtype],:]
    distmat = np.sqrt( np.sum((points[np.newaxis,:,:] - points[:,np.newaxis,:])**2, axis=2) )
    N = N + 1 if rm_self else N
    # args_topN_unsorted is unsorted indices for topN values in distmat, shape(naa, N)
    args_topN_unsorted = np.argpartition(distmat, N)[:,:N]
    args_topN_sorted = np.ndarray((distmat.shape[0], N), dtype=int)
    for i in range(distmat.shape[0]):
        # Kth minimal distance in i_th column; args_topN_sorted is sorted indices for args_topN_unsorted
        vals = distmat[i][args_topN_unsorted[i]]
        indices = np.argsort(vals)
        args_topN_sorted[i] = args_topN_unsorted[i][indices]
    if rm_self:
        args_topN_sorted = args_topN_sorted[:,1:]
    return args_topN_sorted

def zmat2xyz(bond, angle, dihedral, one, two, three):
    """
    Convert Z-matrix coordinates to Cartesian coordinates.

    Args:
        bond (float): bond length
        angle (float): bond angle (in radians)
        dihedral (float): dihedral angle (in radians)
        one, two, three: np.ndarray of shape (3,) representing coordinates of reference atoms

    Returns:
        newvec (np.ndarray): Cartesian coordinates (x, y, z, 1)
    """
    oldvec = np.ones(4, dtype=float)
    oldvec[0] = bond * np.sin(angle) * np.sin(dihedral)
    oldvec[1] = bond * np.sin(angle) * np.cos(dihedral)
    oldvec[2] = bond * np.cos(angle)
    mat = viewat(three, two, one)
    newvec = np.zeros(4, dtype=float)
    for i in range(4):
        for j in range(4):
            newvec[i] += mat[i][j] * oldvec[j]
    return newvec

def viewat(p1, p2, p3):
    """
    Create a 4x4 view (camera) matrix.

    Parameters:
    p1 : np.array, shape (3,) - Camera position (eye)
    p2 : np.array, shape (3,) - Target point (look-at point)
    p3 : np.array, shape (3,) - Reference point to determine the up direction

    Returns:
    mat : np.array, shape (4,4) - 4x4 transformation matrix
    """
    # Compute vectors from p1 to p2 and p1 to p3
    p12 = p2 - p1
    p13 = p3 - p1
    # normalize;
    z = p12 / np.linalg.norm(p12)
    # cross product #
    x = np.cross(p13, p12)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    # transpation matrix
    mat = np.zeros((4, 4), dtype=float)
    for i in range(3):
        mat[i][0] = x[i]
        mat[i][1] = y[i]
        mat[i][2] = z[i]
        mat[i][3] = p1[i]
    mat[3][3] = 1.0
    return mat

def xyz2dihedral(p1, p2, p3, p4):
    """
    Calculate the dihedral (torsion) angle defined by four points in 3D space.

    Parameters:
    p1, p2, p3, p4 : np.array, shape (3,) - Coordinates of the four atoms/residues

    Returns:
    angle : float - Dihedral angle in degrees, range [-180, 180]
    """
    eps = 0.0000001
    # bond vector
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3
    # perpendicular vector #
    perp123 = np.cross(v1, v2)
    perp234 = np.cross(v2, v3)
    perp123 /= np.linalg.norm(perp123)
    perp234 /= np.linalg.norm(perp234)
    # scalar product #
    scp = np.dot(perp123, perp234)
    scp = scp - eps if (1-eps < scp < 1+eps) else scp
    scp = scp + eps if (-1-eps < scp < -1+eps) else scp
    # absolute angle #
    angle = np.rad2deg( np.arccos(scp) )
    return angle if np.dot(v1, perp234) > 0 else -angle

def add_virtual_H(coords, exists, force=True, length_NH=1.00, angle_C_N_H=np.deg2rad(123.0), dhdrl_CA_C_N_H=np.deg2rad(0.0)):
    """
    Add virtual hydrogen atoms (H) to a protein backbone.

    Parameters:
    coords : np.array, shape (N,6,3) 
        Atomic coordinates for N residues, each with 6 backbone atoms.
    exists : np.array, shape (N,6) 
        Boolean array indicating if the atom already exists.
    force : bool, default=True
        If True, overwrite existing H atoms; if False, skip residues where H exists.
    length_NH : float, default=1.00
        N-H bond length in Angstroms.
    angle_C_N_H : float, default=123 deg
        C-N-H bond angle in radians.
    dhdrl_CA_C_N_H : float, default=0.0
        Dihedral angle (CA-C-N-H) in radians for defining H placement plane.

    Returns:
    coords : np.array
        Updated coordinates with H atoms added.
    exists : np.array
        Updated existence array marking H atoms as True.
    """
    for iaa in range(1,len(coords)):
            # Skip if H already exists and we are not forcing overwrite
            if ((exists[iaa][atom2id['H']] == True) and (force==False)):
                continue
            nh = zmat2xyz(length_NH,
                          angle_C_N_H,
                          dhdrl_CA_C_N_H, # plane rad=0
                          coords[iaa-1][atom2id['CA']], # previous CA
                          coords[iaa-1][atom2id['C']], # previous C
                          coords[iaa][atom2id['N']])  # structure starts from N
            coords[iaa][atom2id['H']][0] = nh[0]
            coords[iaa][atom2id['H']][1] = nh[1]
            coords[iaa][atom2id['H']][2] = nh[2]
            exists[iaa][atom2id['H']] = True
    return coords, exists

def add_virtual_O(coords, exists, force=True, length_CO=1.24, angle_N_C_O=np.deg2rad(122.7), dhdrl_CA_N_C_O=np.deg2rad(0.0)):
    """
    Add virtual carbonyl oxygen (O) atoms to a protein backbone.

    Parameters:
    coords : np.array, shape (N,6,3)
        Atomic coordinates for N residues, each with 6 backbone atoms.
    exists : np.array, shape (N,6)
        Boolean array indicating if the atom already exists.
    force : bool, default=True
        If True, overwrite existing O atoms; if False, skip residues where O exists.
    length_CO : float, default=1.24
        C=O bond length in Angstroms.
    angle_N_C_O : float, default=122.7 deg
        N-C-O bond angle in radians.
    dhdrl_CA_N_C_O : float, default=0.0
        Dihedral angle (CA-N-C-O) in radians for defining O placement plane.

    Returns:
    coords : np.array
        Updated coordinates with O atoms added.
    exists : np.array
        Updated existence array marking O atoms as True.
    """
    for iaa in range(len(coords)-1):
        if ((exists[iaa][atom2id['O']] == True) and (force==False)):
            continue
        # print(iaa, coords[iaa])
        # print(iaa+1, coords[iaa+1])
        # bond, angle, dihedral_angle: 1.24, 2.1415189921970423, 0.0; deg2rad: x * pi / 180
        co = zmat2xyz(length_CO,
                        angle_N_C_O,
                        dhdrl_CA_N_C_O,
                        coords[iaa+1][atom2id['CA']], # next CA
                        coords[iaa+1][atom2id['N']],  # next N
                        coords[iaa][atom2id['C']])
        coords[iaa][atom2id['O']][0] = co[0]
        coords[iaa][atom2id['O']][1] = co[1]
        coords[iaa][atom2id['O']][2] = co[2]
        exists[iaa][atom2id['O']] = True
    return coords, exists

def add_virtual_CB(coords, exists, force=True, length_CC=1.54, angle_N_CA_CB=np.deg2rad(110.6), angle_CB_CA_C=np.deg2rad(110.6), dhdrl_C_N_CA_CB=np.deg2rad(-124.4), dhdrl_N_C_CA_CB=np.deg2rad(121.5)):
    """
    Add virtual beta-carbon (CB) atoms to protein backbone.

    Parameters:
    coords : np.array, shape (N,6,3)
        Atomic coordinates for N residues, each with backbone atoms.
    exists : np.array, shape (N,6)
        Boolean array indicating if the atom already exists.
    force : bool, default=True
        If True, overwrite existing CB atoms; if False, skip residues where CB exists.
    length_CC : float, default=1.54
        CA-CB bond length in Angstroms.
    angle_N_CA_CB : float, default=110.6 deg
        Bond angle N-CA-CB in radians.
    angle_CB_CA_C : float, default=110.6 deg
        Bond angle CB-CA-C in radians.
    dhdrl_C_N_CA_CB : float, default=-124.4 deg
        Dihedral angle C-N-CA-CB in radians.
    dhdrl_N_C_CA_CB : float, default=121.5 deg
        Dihedral angle N-C-CA-CB in radians.

    Returns:
    coords : np.array
        Updated coordinates with CB atoms added.
    exists : np.array
        Updated existence array marking CB atoms as True.
    """
    for iaa in range(len(coords)):
        if ((exists[iaa][atom2id['CB']] == True) and (force==False)):
            continue
        cb1 = zmat2xyz(length_CC,
                        angle_N_CA_CB,
                        dhdrl_C_N_CA_CB,
                        coords[iaa][atom2id['C']],
                        coords[iaa][atom2id['N']],
                        coords[iaa][atom2id['CA']])
        cb2 = zmat2xyz(length_CC,
                        angle_CB_CA_C,
                        dhdrl_N_C_CA_CB,
                        coords[iaa][atom2id['N']],
                        coords[iaa][atom2id['C']],
                        coords[iaa][atom2id['CA']])
        cb = (cb1 + cb2)/2.0
        coords[iaa][atom2id['CB']][0] = cb[0]
        coords[iaa][atom2id['CB']][1] = cb[1]
        coords[iaa][atom2id['CB']][2] = cb[2]
        exists[iaa][atom2id['CB']] = True
    return coords, exists

def calc_dihedral(coord, exists, atom2id=atom2id):
    """
    Calculate backbone dihedral angles (psi, phi, omega) for a protein.

    Parameters
    ----------
    coord : np.array, shape (N, natom, 3)
        3D coordinates of all atoms for N residues.
    exists : np.array, shape (N, natom)
        Boolean mask indicating which atoms exist.
    atom2id : dict
        Mapping from atom names (e.g., 'N', 'CA', 'C') to indices in coord array.

    Returns
    -------
    dihedral : np.array, shape (N, 3)
        Dihedral angles in degrees for each residue:
        [psi, phi, omega]
    """
    N = coord.shape[0]
    dihedral = np.zeros((N, 3), dtype=float)

    for iaa in range(N):
            if (iaa > 0) and (exists[iaa-1][atom2id['C']] == True):
                # psi ψ
                dihedral[iaa][0] = xyz2dihedral(coord[iaa-1][atom2id['C']],
                                                     coord[iaa][atom2id['N']],
                                                     coord[iaa][atom2id['CA']],
                                                     coord[iaa][atom2id['C']])
            if (iaa < N-1) and (exists[iaa+1][atom2id['N']] == True):
                # phi Φ
                dihedral[iaa][1] = xyz2dihedral(coord[iaa][atom2id['N']],
                                                     coord[iaa][atom2id['CA']],
                                                     coord[iaa][atom2id['C']],
                                                     coord[iaa+1][atom2id['N']])
            if (iaa < N-1) and (exists[iaa+1][atom2id['CA']] == True):
                # omega. nearly always 180 Ω
                dihedral[iaa][2] = xyz2dihedral(coord[iaa][atom2id['CA']],
                                                     coord[iaa][atom2id['C']],
                                                     coord[iaa+1][atom2id['N']],
                                                     coord[iaa+1][atom2id['CA']])
    return dihedral

def get_node_features(coords, exists, atom2id):
    """
    Compute node features for a protein backbone based on dihedral angles.

    Parameters
    ----------
    coords : np.array, shape (N, natom, 3)
        3D coordinates of all atoms for N residues.
    exists : np.array, shape (N, natom)
        Boolean mask indicating which atoms exist.
    atom2id : dict
        Mapping from atom names (e.g., 'N', 'CA', 'C') to indices in coord array.

    Returns
    -------
    node : np.array, shape (N, 6)
        Node features for each residue:
        [sin(psi), cos(psi), sin(phi), cos(phi), sin(omega), cos(omega)]
    """
    N = coords.shape[0]
    dihedral = calc_dihedral(coords, exists, atom2id)  # (N,3)
    node = np.zeros((N, 6), dtype=float)
    # psi, phi, omega
    sins = np.sin(np.deg2rad(dihedral))
    coss = np.cos(np.deg2rad(dihedral))
    node[:, 0::2] = sins # indices 0, 2, 4 = sin (psi, phi, omega)
    node[:, 1::2] = coss # indices 1, 3, 5 = cos (psi, phi, omega)
    node[0, 0:2] = 0
    node[-1, 2:] = 0 
    return node

def get_edge_features(coords, nneighbor=20, atomtype='CB', rm_self=True, dist_chbreak=2.0, dist_mean=6.4, dist_var=2.4):
    """
    Compute edge features and adjacency matrix for a protein structure given backbone coordinates.

    Parameters
    ----------
    coords : np.array, shape (N, natom, 3)
        3D coordinates of all atoms for N residues.
    nneighbor : int
        Number of nearest neighbors to consider for each residue.
    atomtype : str
        Atom type used to compute distances for neighbors (default 'CB').
    rm_self : bool
        Whether to remove self-edges (default True).
    dist_chbreak : float
        Distance cutoff (angstrom) to detect chain breaks (default 2.0 Å).
    dist_mean : float
        Mean distance for normalization (default 6.4 Å).
    dist_var : float
        Variance for normalization (default 2.4 Å).

    Returns
    -------
    edgemat : np.array, shape (N, N, 36)
        Edge features between residue pairs (distance-based, normalized).
    adjmat : np.array, shape (N, N, 1)
        Boolean adjacency matrix indicating which residues are neighbors.
    mask : np.array, shape (N,)
        Boolean mask indicating residues not in chain breaks.
    """
    N, natom, _ = coords.shape
    mask = np.ones((len(coords)), dtype=bool)
    for iaa in range(len(coords)):
        # d1: N - CA distance; d2: CA - C distance
        d1 = np.sqrt(np.sum((coords[iaa,0,:] - coords[iaa,1,:])**2, axis=0))
        d2 = np.sqrt(np.sum((coords[iaa,1,:] - coords[iaa,2,:])**2, axis=0))
        # hypara.dist_chbreak: 2.0 angstrom, distance cutoff for chain break
        # if d1 > hypara.dist_chbreak or d2 > hypara.dist_chbreak:
        if d1 > 2 or d2 > 2:
            mask[iaa] = 0
    for iaa in range(len(coords)-1):
        # d3: C - next N distance. Ensure n * (N-CA-C-) have no breaks longer than 2 angstrom.
        d3 = np.sqrt(np.sum((coords[iaa,2,:] - coords[iaa+1,0,:])**2, axis=0))
        if d3 > dist_chbreak:
            # broken residues (n_th) and (n_th + 1) should not be masked
            mask[iaa], mask[iaa+1] = 0, 0
    # edge features. edgemat shape: (naa, naa, 36); adjmat shape: (naa, naa, 1)
    edgemat = np.zeros((len(coords), len(coords), 36), dtype=float)
    adjmat = np.zeros((len(coords), len(coords), 1), dtype=bool)
    # nneighbor = 20; top k distance neighbors. nn is neighbor indices to index neighbor distance
    nn = get_nearestN(coords, nneighbor, atomtype='CB', rm_self=True)
    for iaa in range(len(coords)):
        adjmat[iaa, nn[iaa]] = True
        if(mask[iaa] == False): continue
        # dist_mean:        float = 6.4 # distance mean for normalization
        # dist_var:         float = 2.4 # distance variance for normalization
        for i in nn[iaa]:
            edgemat[iaa, i] = np.sqrt(
                np.sum((coords[iaa,:,np.newaxis,:] - coords[i,np.newaxis,:,:])**2, axis=2)
            ).reshape(-1)
            # normalization.
            edgemat[iaa, i] = (edgemat[iaa, i] - dist_mean) / dist_var
    return edgemat, adjmat, mask

def get_graph_features(coords, exists, resname, atom2id=atom2id, nneighbor=20, rm_self=True, atomtype='CB', dist_chbreak=2.0, dist_mean=6.4, dist_var=2.4):
    """
    Extract graph features (node features, edge features, adjacency, labels, masks) for a protein structure.

    Parameters
    ----------
    coords : np.array, shape (N, natom, 3)
        3D coordinates of atoms for N residues.
    exists : np.array, shape (N, natom)
        Boolean mask indicating if an atom exists.
    resname : list of str
        Residue names (three-letter codes) for N residues.
    atom2id : dict
        Mapping from atom names to indices.
    nneighbor : int
        Number of nearest neighbors for each residue.
    rm_self : bool
        Whether to remove self-edges.
    atomtype : str
        Atom type used to calculate nearest neighbors ('CB' by default).
    dist_chbreak : float
        Distance cutoff to detect chain breaks.
    dist_mean : float
        Mean distance for edge feature normalization.
    dist_var : float
        Variance for edge feature normalization.

    Returns
    -------
    node : torch.FloatTensor, shape (N, 6)
        Node features (sin/cos of dihedral angles ψ, φ, ω).
    edgemat : torch.FloatTensor, shape (N, N, 36)
        Edge features (pairwise distances between atoms, normalized).
    adjmat : torch.BoolTensor, shape (N, N, 1)
        Adjacency matrix indicating which residues are neighbors.
    label : torch.LongTensor, shape (N,)
        Residue label indices (0-19, unknown=20).
    mask : torch.BoolTensor, shape (N,)
        Mask for valid residues (not chain breaks and known residues).
    aa1 : np.array of str, shape (N,)
        One-letter residue codes corresponding to resname.
    """

    N = coords.shape[0]

    # node features (dihedral)
    node = get_node_features(coords, exists, atom2id)  # (N,6)
    
    # edge fea (atom-atom distances)
    edgemat, adjmat, mask = get_edge_features(coords, nneighbor=nneighbor, atomtype=atomtype, rm_self=rm_self, dist_chbreak=dist_chbreak, dist_mean=dist_mean, dist_var=dist_var)

    # label
    aa1 = np.array([three2one.get(x,'X') for x in resname])
    label = np.array([mapped.get(x,20) for x in aa1])
    mask &= (label != 20)
    node = torch.FloatTensor(node)
    edgemat = torch.FloatTensor(edgemat)
    adjmat = torch.BoolTensor(adjmat)
    mask = torch.BoolTensor(mask)
    label = torch.LongTensor(label)
    return node, edgemat, adjmat, label, mask, aa1

def replace_nan(x, value=0.0):
    return torch.nan_to_num(x, nan=value)

def add_margin(node, edgemat, adjmat, label, mask, nneighbor):
    """
    Add padding (margins) to graph features to prevent boundary issues
    in graph neural network computations.
    """
    # node
    # input, pad(left,right,up,down), mode, value
    node = torch.nn.functional.pad(node, (0,0,1,1), 'constant', 0)
    edgemat = torch.nn.functional.pad(edgemat, (0,0,1,1,1,1), 'constant', 0)
    adjmat = torch.nn.functional.pad(adjmat, (0,0,1,1,1,1), 'constant', False)
    adjmat[0,0:nneighbor,0] = True
    adjmat[-1,0:nneighbor,0] = True
    label = torch.nn.functional.pad(label, (1,1), 'constant', 20)
    mask = torch.nn.functional.pad(mask, (1,1), 'constant', False)
    return node, edgemat, adjmat, label, mask

def protein_to_features(file, file_format='pdb', nneighbor=20, device='cpu'):
    """
    Convert protein backbone structure to protein graph based on k-nearest neighbors.
    """
    coords, resnames, iaa2org = get_protein_coords(file, file_format=file_format)
    exists = np.ones((len(coords), len(atom2id)), dtype=bool)
    exists[:,atom2id['CB']] = False
    exists[:,atom2id['H']] = False
    coords, exists = add_virtual_H(coords, exists)
    coords, exists = add_virtual_O(coords, exists)
    coords, exists = add_virtual_CB(coords, exists)
    coords[0, 5] = coords[0, 0] # H. first H atom is omitted in addO. Use coord of N in the first residue instead.
    coords[-1, 4] = coords[-1, 3] # CB. Last CB atom is omitted in addCB. Use coord of O in final residue instead.
    node, edgemat, adjmat, label, mask, aa1 = get_graph_features(coords, exists, resnames, atom2id=atom2id, nneighbor=nneighbor)
    node, edgemat, adjmat, label, mask = add_margin(node, edgemat, adjmat, label, mask, nneighbor)
    node, edgemat, adjmat = map(replace_nan, [node, edgemat, adjmat])
    return node.squeeze(), edgemat.squeeze(), adjmat.squeeze(), label.squeeze(), mask.squeeze(), aa1

def _process_pdb(pdb_path, nneighbor):
    try:
        coords, resnames, iaa2org = get_protein_coords(pdb_path)
        exists = np.ones((len(coords), len(atom2id)), dtype=bool)
        exists[:, atom2id['CB']] = False
        exists[:, atom2id['H']] = False
        coords, exists = add_virtual_H(coords, exists)
        coords, exists = add_virtual_O(coords, exists)
        coords, exists = add_virtual_CB(coords, exists)
        coords[0, 5] = coords[0, 0]
        coords[-1, 4] = coords[-1, 3]

        node, edgemat, adjmat, label, mask, aa1 = get_graph_features(
            coords, exists, resnames, atom2id=atom2id, nneighbor=nneighbor
        )
        node, edgemat, adjmat = map(replace_nan, [node, edgemat, adjmat])
        naa = coords.shape[0]
        if naa > nneighbor + 1:
            return pdb_path
        else:
            return None
    except Exception as e:
        print(f'Warning: skip {pdb_path}! ({e})')
        return None

def check_pdb(datalist, nneighbor=20, num_workers=8):
    checklist = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_pdb, pdb_path, nneighbor): pdb_path for pdb_path in datalist}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                checklist.append(result)
    return checklist
    
# def check_pdb(datalist, nneighbor=20):
#     checklist = []
#     for pdb_path in datalist:
#         try:
#             coords, resnames, iaa2org = get_protein_coords(pdb_path)
#             exists = np.ones((len(coords), len(atom2id)), dtype=bool)
#             exists[:,atom2id['CB']] = False
#             exists[:,atom2id['H']] = False
#             coords, exists = add_virtual_H(coords, exists)
#             coords, exists = add_virtual_O(coords, exists)
#             coords, exists = add_virtual_CB(coords, exists)
#             coords[0, 5] = coords[0, 0] # H. first H atom is omitted in addO. Use coord of N in the first residue instead.
#             coords[-1, 4] = coords[-1, 3] # CB. Last CB atom is omitted in addCB. Use coord of O in final residue instead.
#             node, edgemat, adjmat, label, mask, aa1 = get_graph_features(coords, exists, resnames, atom2id=atom2id, nneighbor=nneighbor)
#             node, edgemat, adjmat = map(replace_nan, [node, edgemat, adjmat])
#             naa = coords.shape[0]
#             if naa > nneighbor+1:
#                 checklist.append(pdb_path)
#             else:
#                 continue
#         except:
#             print(f'Error loading {pdb_path}!')
#             continue
#     return checklist
