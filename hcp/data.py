"""Prepare DK68 streamline counts and deterministic subject-specific masks."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


COHORTS = ('F31_35', 'M22_25', 'M26_30')
EXPECTED_COUNTS = {'F31_35': 233, 'M22_25': 143, 'M26_30': 213}
REGIONS = ('bankssts caudalanteriorcingulate caudalmiddlefrontal cuneus entorhinal '
           'fusiform inferiorparietal inferiortemporal isthmuscingulate lateraloccipital '
           'lateralorbitofrontal lingual medialorbitofrontal middletemporal parahippocampal '
           'paracentral parsopercularis parsorbitalis parstriangularis pericalcarine '
           'postcentral posteriorcingulate precentral precuneus rostralanteriorcingulate '
           'rostralmiddlefrontal superiorfrontal superiorparietal superiortemporal '
           'supramarginal frontalpole temporalpole transversetemporal insula').split()
ROI_LABELS = np.array([f'ctx-{hemi}-{region}' for hemi in ('lh', 'rh') for region in REGIONS])


def read_npz(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_networks(path):
    """Read the public scs_desikan.mat or the documented DK68 NPZ schema."""
    path = Path(path)
    if path.suffix.lower() == '.mat':
        from scipy.io import loadmat
        raw = loadmat(path)
        if 'scs' not in raw or 'subject_list' not in raw:
            raise ValueError('Expected scs and subject_list in scs_desikan.mat')
        tensor = np.asarray(raw['scs'])
        if tensor.ndim != 3 or tensor.shape[:2] != (87, 87):
            raise ValueError('Expected an 87 x 87 x participants streamline-count tensor')
        upper = np.triu(tensor[19:87, 19:87, :].transpose(2, 0, 1), 1)
        networks = upper + upper.transpose(0, 2, 1)
        ids = raw['subject_list'].reshape(-1).astype(np.int64).astype(str)
        roi = ROI_LABELS.copy()
    else:
        raw = read_npz(path)
        networks, ids, roi = raw['networks'], raw['subject_ids'].astype(str), raw['roi_labels'].astype(str)
    if networks.shape != (len(ids), 68, 68) or len(set(ids)) != len(ids):
        raise ValueError('Expected unique subjects and participant x 68 x 68 networks')
    if not np.array_equal(roi, ROI_LABELS):
        raise ValueError('ROI order differs from the cortical Desikan--Killiany source order')
    if not np.isfinite(networks).all() or np.any(networks < 0):
        raise ValueError('Counts must be finite and nonnegative')
    if not np.array_equal(networks, networks.transpose(0, 2, 1)) or np.any(np.diagonal(networks, axis1=1, axis2=2)):
        raise ValueError('Networks must be symmetric with zero diagonal')
    if not np.array_equal(networks, np.rint(networks)):
        raise ValueError('Expected untransformed streamline counts, not normalized weights')
    return networks.astype(np.float64), ids, roi


def make_masks(subject_ids, *, seed=12345, edges=2278, validation_edges=228, test_edges=228):
    validation = np.zeros((len(subject_ids), edges), dtype=bool)
    test = np.zeros_like(validation)
    for row, subject in enumerate(subject_ids):
        row_seed = int(hashlib.sha256(f'{seed}:{subject}'.encode()).hexdigest()[:16], 16)
        order = np.random.default_rng(row_seed).permutation(edges)
        validation[row, order[:validation_edges]] = True
        test[row, order[validation_edges:validation_edges + test_edges]] = True
    return {'train': ~(validation | test), 'validation': validation, 'test': test}


def load_people(path, subject_ids=None):
    frame = pd.read_csv(path, dtype={'Subject': str}, low_memory=False)
    required = {'Subject', 'Age', 'Gender'}
    if not required.issubset(frame.columns) or frame.Subject.isna().any() or frame.Subject.duplicated().any():
        raise ValueError('Metadata require unique Subject, Age and Gender columns')
    frame = frame.set_index('Subject')
    if subject_ids is not None:
        missing = set(subject_ids) - set(frame.index)
        if missing:
            raise ValueError(f'Metadata missing for {len(missing)} network subjects')
        frame = frame.loc[subject_ids]
    return frame


def cohort_labels(people):
    return (people.Gender.astype(str) + people.Age.astype(str).str.replace('-', '_', regex=False)
            .str.replace('+', 'plus', regex=False)).to_numpy(dtype=str)


def prepare(networks_path, people_path, output, *, masks_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    networks, ids, roi = load_networks(networks_path)
    people = load_people(people_path, ids)
    masks = make_masks(ids) if masks_path is None else read_npz(masks_path)
    masks = {key: masks[key] for key in ('train', 'validation', 'test')}
    for key, expected in [('train', 1822), ('validation', 228), ('test', 228)]:
        mask = masks[key]
        if mask.dtype != bool or mask.shape != (len(ids), 2278) or not np.all(mask.sum(1) == expected):
            raise ValueError(f'Invalid {key} mask')
    if not np.all(np.stack(list(masks.values())).sum(0) == 1):
        raise ValueError('Training, validation and test masks must partition every edge')
    ii, jj = np.triu_indices(68, 1)
    cohorts = cohort_labels(people)
    np.savez_compressed(output / 'networks.npz', count=networks[:, ii, jj], subject_ids=ids,
                        roi_labels=roi, cohort=cohorts, edge_i=ii, edge_j=jj, **masks)
    people.to_csv(output / 'people.csv', index=True)
    write_json(output / 'preparation.json', {
        'network_sha256': file_hash(networks_path), 'metadata_sha256': file_hash(people_path),
        'mask_sha256': None if masks_path is None else file_hash(masks_path),
        'mask_seed': 12345 if masks_path is None else None, 'cohort_counts': {str(k): int(v) for k, v in pd.Series(cohorts).value_counts().items()},
        'conversion': '87-region count channel: cortical slice 19:87; upper triangle plus transpose',
        'subject_order': 'source network row order', 'contains_participant_data': True,
    })
    return output


def observations(count, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Training scale must be positive and finite')
    z = count > 0
    y = np.zeros(count.shape, dtype=np.float64)
    y[z] = np.log(count[z] / scale)
    return z.astype(np.int32), y


def load_cohort(prepared, cohort):
    data = read_npz(Path(prepared) / 'networks.npz')
    rows = np.flatnonzero(data['cohort'] == cohort)
    if not len(rows):
        raise ValueError(f'No participants in {cohort}')
    count, train = data['count'][rows], data['train'][rows]
    positive = count[train & (count > 0)]
    if not len(positive):
        raise ValueError('No positive training edges')
    scale = float(np.median(positive))
    z, y = observations(count, scale)
    return dict(z=z, y=y, train=train, validation=data['validation'][rows], test=data['test'][rows],
                rows=rows, subject_ids=data['subject_ids'][rows], roi_labels=data['roi_labels'],
                training_scale=scale, cohort=cohort)


def target_rows(data, source):
    if source == 'M22_25':
        cohorts = ('M26_30', 'M31_35', 'M36plus')
    elif source == 'F31_35':
        cohorts = ('F22_25', 'F26_30', 'M22_25', 'M26_30', 'M31_35', 'M36plus')
    else:
        raise ValueError('The manuscript transfer sources are M22_25 and F31_35')
    return np.flatnonzero(np.isin(data['cohort'], cohorts))
