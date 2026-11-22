#!/usr/bin/env python3
"""Compile the small surface-code memory into an exact logical MAP table."""

import numpy as np


def logical_map(image):
    if image['checks'] > 16:
        raise ValueError('the full probability table is limited to 16 detectors')
    size = 1 << image['checks']
    addresses = np.arange(2*size, dtype=np.int32)
    distribution = np.zeros(2*size, dtype=np.float64)
    distribution[0] = 1.0
    for endpoints, logical, p in image['faults']:
        mask = logical*size + sum(1 << i for i in endpoints)
        distribution = (1-p)*distribution + p*distribution[addresses ^ mask]
    if not np.isclose(distribution.sum(), 1.0) or not np.all(distribution >= 0):
        raise RuntimeError('invalid joint distribution')
    return distribution[size:] > distribution[:size]


def emit_map(image, path):
    table = logical_map(image)
    entries = min(len(table), 1 << 14)
    words = np.zeros(entries, dtype=np.uint8)
    for bank, section in enumerate(table.reshape(-1, entries)):
        words |= section.astype(np.uint8) << bank
    path.write_text(''.join(f'{word:x}\n' for word in words))
