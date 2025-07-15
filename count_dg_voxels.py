import csv
import nrrd
import numpy as np

CSV_PATH = 'Mouse.csv'
NRRD_PATH = 'annotation_25.nrrd'

# Gather dentate gyrus related region IDs
ids = {}
with open(CSV_PATH, newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if len(row) < 3:
            continue
        try:
            region_id = int(row[1])
        except ValueError:
            continue
        name = row[-1].strip()
        acronym = row[2].strip()
        if 'dentate gyrus' in name.lower():
            ids[region_id] = name

# Load the annotation volume
volume, _ = nrrd.read(NRRD_PATH)

total = 0
subregion_counts = {}
for region_id, region_name in ids.items():
    count = int(np.sum(volume == region_id))
    subregion_counts[region_id] = count
    total += count

print(f"Total DG voxels: {total}")
for region_id, count in subregion_counts.items():
    print(f"{region_id}\t{ids[region_id]}\t{count}")
