import numpy as np
import nrrd

NRRD_PATH = 'annotation_25.nrrd'
OUTPUT_PATH = 'annotation_25_dg_septal_temporal.nrrd'

DG_MO = 10703
DG_PO = 10704
DG_SG = 632

NEW_IDS = {
    'DG_MO_SEPTAL': 700000001,
   'DG_MO_TEMPORAL': 700000002,
    'DG_PO_SEPTAL': 700000003,
    'DG_PO_TEMPORAL': 700000004,
    'DG_SG_SEPTAL': 700000005,
    'DG_SG_TEMPORAL': 700000006,
}

vol, header = nrrd.read(NRRD_PATH)

# Axis 1 corresponds to posterior->anterior in the provided volume
AXIS = 1

for region_id, septal_id, temporal_id in [
    (DG_MO, NEW_IDS['DG_MO_SEPTAL'], NEW_IDS['DG_MO_TEMPORAL']),
    (DG_PO, NEW_IDS['DG_PO_SEPTAL'], NEW_IDS['DG_PO_TEMPORAL']),
    (DG_SG, NEW_IDS['DG_SG_SEPTAL'], NEW_IDS['DG_SG_TEMPORAL']),
]:
    coords = np.argwhere(vol == region_id)
    if coords.size == 0:
        print(f'Region id {region_id} not found, skipping.')
        continue
    min_pos = coords[:, AXIS].min()
    max_pos = coords[:, AXIS].max()
    split_index = (min_pos + max_pos) // 2
    print(f'Region {region_id}: axis range {min_pos}-{max_pos}, split at {split_index}')

    idx = np.arange(vol.shape[AXIS])
    slicer = [None, None, None]
    slicer[AXIS] = slice(None)
    axis_indices = idx[tuple(slicer)]

    mask = vol == region_id
    mask_septal = mask & (axis_indices <= split_index)
    mask_temporal = mask & (axis_indices > split_index)

    vol[mask_septal] = septal_id
    vol[mask_temporal] = temporal_id

nrrd.write(OUTPUT_PATH, vol, header)
print('Saved', OUTPUT_PATH)
