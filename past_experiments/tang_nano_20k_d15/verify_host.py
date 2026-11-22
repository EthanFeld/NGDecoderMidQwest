"""Exercise the stock host, packet checks and fallback without a board."""

import struct
from tempfile import TemporaryDirectory

import numpy as np

from geometry import ROOT
from reference import Reference, build_reference
from run import UartFront, smoke
from transport import (ACCEPT_0, BAD_PACKET, REPLY_BYTES, REPLY_MAGIC, REQUEST_MAGIC,
                       crc32c, parse_reply, project, telescope)


def reply_frame(tag, code):
    header = struct.pack("<HH", REPLY_MAGIC, code << 8 | tag)
    return header + struct.pack("<I", crc32c(header))


class Loopback:
    def __init__(self, engine):
        self.engine = engine
        self.writes = 0
        self.reply = b""

    def write(self, packet):
        self.writes += 1
        assert len(packet) == 232
        magic, header = struct.unpack_from("<HH", packet)
        assert magic == REQUEST_MAGIC and header >> 8 == 1
        if crc32c(packet[:-4]) == int.from_bytes(packet[-4:], "little"):
            projected = np.frombuffer(packet[4:-4], dtype=np.uint8)[None, :]
            codes, _ = self.engine.decode(projected)
            code = int(codes[0])
        else:
            code = BAD_PACKET
        self.reply = reply_frame(header & 255, code)
        return len(packet)

    def read(self, size):
        assert size == REPLY_BYTES
        data, self.reply = self.reply, b""
        return data


class BrokenReply(Loopback):
    def __init__(self, engine, mutate):
        super().__init__(engine)
        self.mutate = mutate

    def read(self, size):
        return self.mutate(super().read(size))


def test_reply():
    assert crc32c(b"123456789") == 0xE3069283
    frame = reply_frame(17, ACCEPT_0)
    assert parse_reply(frame, 17).logical == 0
    for bit in range(8 * len(frame)):
        damaged = bytearray(frame)
        damaged[bit // 8] ^= 1 << (bit % 8)
        try:
            parse_reply(damaged, 17)
        except ValueError:
            continue
        raise AssertionError(f"Reply accepted after bit {bit} was flipped.")
    print("CRC test vector and all 64 single-bit reply corruptions passed")


def test_fallback(engine):
    rows = np.load(ROOT / "examples/shots.npy", allow_pickle=False)
    snapshot = rows.copy()
    ids = np.arange(500, 500 + len(rows), dtype=np.uint64)
    codes, _ = engine.decode(project(rows))
    accepted = codes != 0
    fallback_bits = (ids % 2).astype(np.uint8)

    def fallback(deferred, deferred_ids):
        np.testing.assert_array_equal(deferred, snapshot[~accepted])
        np.testing.assert_array_equal(deferred_ids, ids[~accepted])
        return fallback_bits[~accepted]

    front = UartFront(Loopback(engine))
    front.tag = 240
    logical, keep = telescope(rows, front, fallback, shot_ids=ids)
    assert front.fault is None and front.tag == 48
    assert int(keep.sum()) == 29
    np.testing.assert_array_equal(keep, accepted)
    np.testing.assert_array_equal(logical, np.where(accepted, codes - 1, fallback_bits))
    np.testing.assert_array_equal(rows, snapshot)
    print("64-shot host loopback, tag wraparound and deferred rows and IDs passed")

    mutations = (lambda frame: frame[:-1],
                 lambda _: reply_frame(1, ACCEPT_0),
                 lambda frame: frame[:-1] + bytes([frame[-1] ^ 1]))
    for mutate in mutations:
        link = BrokenReply(engine, mutate)
        failed = UartFront(link)

        def use_cpu(deferred, deferred_ids):
            np.testing.assert_array_equal(deferred, snapshot)
            np.testing.assert_array_equal(deferred_ids, ids)
            return fallback_bits

        logical, keep = telescope(rows, failed, use_cpu, shot_ids=ids)
        assert failed.fault is not None and link.writes == 1 and not keep.any()
        np.testing.assert_array_equal(logical, fallback_bits)
    print("Short, mis-tagged and damaged replies routed all affected shots to fallback")


if __name__ == "__main__":
    test_reply()
    with TemporaryDirectory() as scratch, Reference(build_reference(scratch)) as engine:
        smoke(UartFront(Loopback(engine)))
        test_fallback(engine)
