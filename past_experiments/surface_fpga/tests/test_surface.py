import itertools
import unittest

import numpy as np

from exact import logical_map
from host import Inbox, REPLY, frame, pack, parse_reply, unpack
from reference import Predecoder, sample
from rescue import ResidualCheck
from surface_model import generate


class TestSurface(unittest.TestCase):
    def test_single_faults_in_every_profile(self):
        for d in (3, 7, 15):
            for basis in ('x', 'z'):
                with self.subTest(distance=d, basis=basis):
                    image = generate(d, basis, d, .001)
                    if d == 3:
                        image['frontend'] = 'map'
                    model = Predecoder(image)
                    self.assertEqual(image['fault_distance'], d)
                    for start in range(0, len(image['faults']), 256):
                        faults = image['faults'][start:start+256]
                        words = np.zeros((len(faults), image['checks']), dtype=np.bool_)
                        for k, (edges, _, _) in enumerate(faults):
                            words[k, edges] = True
                        decoded = model.decode(words)
                        self.assertTrue(decoded.accepted.all())
                        np.testing.assert_array_equal(decoded.logical, [x[1] for x in faults])

    def test_boundary_choices_preserved_on_defer(self):
        image = generate(15, 'z', 15, .001)
        model = Predecoder(image)
        syndrome, _ = sample(image, 512, np.random.default_rng(71715))
        completed = model.decode(syndrome)
        missed = ~completed.accepted
        self.assertTrue(missed.any())
        self.assertTrue(np.all(completed.clocks[missed] == 78))
        self.assertFalse(completed.logical[missed].any())
        self.assertFalse((completed.residual & ~syndrome).any())
        boundary = model.boundary.reshape(-1)
        self.assertTrue((completed.residual[missed] & boundary).any())

    def test_map_against_enumeration(self):
        model = {'checks': 3, 'faults': [([0], 1, .01), ([0, 1], 0, .02),
                                         ([1, 2], 0, .03), ([2], 0, .035)]}
        prob = np.zeros((2, 8))
        for active in itertools.product((False, True), repeat=len(model['faults'])):
            syndrome, logical, rate = 0, 0, 1.0
            for use, (edges, label, p) in zip(active, model['faults'], strict=True):
                rate *= p if use else 1-p
                if use:
                    syndrome ^= sum(1 << r for r in edges)
                    logical ^= label
            prob[logical, syndrome] += rate
        np.testing.assert_array_equal(logical_map(model), prob[1] > prob[0])

    def test_rescues_do_not_change_existing_fpga_answers(self):
        for distance in (7, 15):
            image = generate(distance, 'z', distance, .003)
            raw, _ = sample(image, 2048, np.random.default_rng(617915))
            baseline = Predecoder(image).decode(raw)
            updated = Predecoder({**image, 'frontend': 'rescue'}).decode(raw)
            take = updated.rescued
            self.assertTrue(take.any())
            self.assertFalse((take & baseline.accepted).any())
            np.testing.assert_array_equal(updated.accepted, baseline.accepted | take)
            np.testing.assert_array_equal(updated.clocks, baseline.clocks)
            np.testing.assert_array_equal(updated.logical[~take], baseline.logical[~take])
            np.testing.assert_array_equal(updated.residual[~take], baseline.residual[~take])

    def test_patch_span_and_cut_projection(self):
        image = generate(7, 'z', 7, .001)
        patch = ResidualCheck(image)
        i = patch.plan.tiles[0][0][0]
        bits = np.zeros((3, image['checks']), bool)
        bits[:, i] = True
        bits[0, 2*patch.width+i] = True
        bits[1, 3*patch.width+i] = True
        np.testing.assert_array_equal(patch.fit(bits), (True, False, False))
        for endpoints, label, _ in image['faults']:
            if len(endpoints) == 1:
                word = np.zeros((1, image['checks']), bool)
                word[0, endpoints] = True
                self.assertEqual(int(patch.cut_count(word, np.array([label]))[0]), 1)


class TestProtocol(unittest.TestCase):
    def test_escaped_frame_and_fragments(self):
        body = bytes.fromhex('927d7e1234')
        wire = frame(body)
        inbox = Inbox()
        corrupt = bytearray(wire)
        corrupt[1] ^= 1
        self.assertEqual(list(inbox.feed(corrupt)), [])
        actual = []
        for b in wire:
            actual.extend(inbox.feed([b]))
        self.assertEqual(actual, [body])

    def test_pack_and_reply_padding(self):
        raw = bytes.fromhex('01040b0f')
        words = unpack(raw, 4, 4)
        self.assertEqual(pack(words, 4), raw)
        header = REPLY.pack(0x92, 2, 13, 3, 50, 0, 4)
        token, partial = parse_reply(header+raw, 13, 4, 4, 50)
        self.assertEqual(token, 2)
        self.assertEqual(partial.residual, raw)
        self.assertRaises(ValueError, parse_reply, header+bytes.fromhex('ff040b0f'), 13, 4, 4, 50)
        self.assertRaises(ValueError, parse_reply, header+raw, 10, 4, 4, 50)


if __name__ == '__main__':
    unittest.main()
