
import os
import csv
import math
import numpy as np
import nrrd

NRRD_PATH = 'annotation_25_dg_septal_temporal.nrrd'
SWC_DIR = 'Neurons_DG_ION_ext'
OUTPUT_CSV = 'neuron_sublayer_lengths.csv'

NEW_IDS = {
    'DG_MO_SEPTAL': 700000001,
    'DG_MO_TEMPORAL': 700000002,
    'DG_PO_SEPTAL': 700000003,
    'DG_PO_TEMPORAL': 700000004,
    'DG_SG_SEPTAL': 700000005,
    'DG_SG_TEMPORAL': 700000006,
}

print('Loading annotation volume...')
volume, header = nrrd.read(NRRD_PATH)
voxel_size = np.array([float(np.linalg.norm(d)) for d in header['space directions']])
origin = np.array(header['space origin'])


def world_to_index(coord):
    idx = np.round((np.array(coord) - origin) / voxel_size).astype(int)
    return idx


def get_region(coord):
    idx = world_to_index(coord)
    if np.any(idx < 0) or np.any(idx >= volume.shape):
        return None
    return int(volume[tuple(idx)])


def segment_length(a, b):
    return math.dist(a, b)


def process_swc(path):
    nodes = {}
    soma = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            nid = int(parts[0])
            ntype = int(parts[1])
            x, y, z = map(float, parts[2:5])
            parent = int(parts[6])
            nodes[nid] = (x, y, z, parent, ntype)
            if parent == -1 and soma is None:
                soma = (x, y, z)
    if soma is None:
        return None
    soma_region = get_region(soma)
    if soma_region not in NEW_IDS.values():
        return None
    lengths = {k: 0.0 for k in NEW_IDS}
    for nid, (x, y, z, parent, ntype) in nodes.items():
        if parent == -1 or parent not in nodes:
            continue
        # only accumulate axon segments (type 2)
        if ntype != 2:
            continue
        px, py, pz, _, _ = nodes[parent]
        length = segment_length((x, y, z), (px, py, pz))
        mid = ((x + px) / 2.0, (y + py) / 2.0, (z + pz) / 2.0)
        region = get_region(mid)
        for key, val in NEW_IDS.items():
            if region == val:
                lengths[key] += length
                break
    return soma_region, lengths


def main():
    rows = []
    for fname in sorted(os.listdir(SWC_DIR)):
        if not fname.endswith('.swc'):
            continue
        res = process_swc(os.path.join(SWC_DIR, fname))
        if res is None:
            continue
        soma_region, lengths = res
        row = {
            'neuron': fname,
            'soma_region': soma_region,
        }
        for k in NEW_IDS:
            row[k] = lengths[k]
        rows.append(row)
        print('Processed', fname)

    with open(OUTPUT_CSV, 'w', newline='') as f:
        fieldnames = ['neuron', 'soma_region'] + list(NEW_IDS.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print('Saved', OUTPUT_CSV)


if __name__ == '__main__':
    main()
