import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import sample
from growth_reference import GrowthModel
from surface_model import generate


def vectors(folder, count):
    image = json.loads((folder/'graph.json').read_text())
    model = GrowthModel(folder/'growth_graph.txt')
    rng = np.random.default_rng(715012)
    collected = [np.zeros((1, image['checks']), bool)]
    for p in (.001, image.get('growth_p', image['p']), .008):
        physical = generate(image['distance'], image['basis'], image['rounds'], p)
        collected.append(sample(physical, count, rng)[0])
    cases = np.vstack(collected)
    out = model.decode(cases, image['growth_cycles'])
    with (folder/'growth_vectors.txt').open('w') as stream:
        print(len(cases), file=stream)
        for syndrome, (label, status, cycles) in zip(cases, out, strict=True):
            print(int(status == 0), int(label), int(cycles), file=stream)
            for row in syndrome.reshape(-1, image['width']):
                print(f'{sum(1<<int(i) for i in row.nonzero()[0]):x}', file=stream)


if __name__ == '__main__':
    vectors(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 50)
