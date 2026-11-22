"""Detector projection, packet framing, CRC, and fallback routing."""

from dataclasses import dataclass
import json
from pathlib import Path
import struct

import numpy as np


REQUEST_MAGIC = 0xD15A
REPLY_MAGIC = 0xD15B
REPLY_BYTES = 8
DEFER, ACCEPT_0, ACCEPT_1, BAD_PACKET = range(4)
GEOMETRY = json.loads((Path(__file__).resolve().parent / "data/geometry.json").read_text())
_Z_CHECKS = np.asarray(GEOMETRY["planes"]).reshape(-1, 8).T
_Z_BYTE = _Z_CHECKS // 8
_Z_SHIFT = (_Z_CHECKS % 8).astype(np.uint8)
RAW_BYTES = GEOMETRY["detector_count"] // 8
PROJECTED_BYTES = _Z_CHECKS.shape[1]


def _crc_table_entry(value):
    for _ in range(8):
        value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
    return value


_CRC_TABLE = tuple(map(_crc_table_entry, range(256)))


def crc32c(data):
    value = 0xFFFFFFFF
    for byte in data:
        value = (value >> 8) ^ _CRC_TABLE[(value ^ byte) & 0xFF]
    return value ^ 0xFFFFFFFF


def project(packed):
    rows = np.asarray(packed)
    if rows.ndim != 2 or rows.shape[1] != RAW_BYTES or rows.dtype != np.uint8:
        raise ValueError(f"Expected a batch of {RAW_BYTES}-byte Stim detector rows with dtype uint8.")
    out = np.zeros((len(rows), PROJECTED_BYTES), dtype=np.uint8)
    for lane in range(8):
        out |= ((rows[:, _Z_BYTE[lane]] >> _Z_SHIFT[lane]) & 1) << lane
    return out


def request(projected, tag):
    if len(projected) != PROJECTED_BYTES or not 0 <= tag <= 255:
        raise ValueError("A request takes a 224-byte checked syndrome and an 8-bit tag.")
    body = struct.pack("<HH", REQUEST_MAGIC, 0x100 | tag) + bytes(projected)
    return body + struct.pack("<I", crc32c(body))


@dataclass(frozen=True)
class Reply:
    tag: int
    code: int

    @property
    def logical(self):
        return self.code - ACCEPT_0 if self.code in (ACCEPT_0, ACCEPT_1) else None


def parse_reply(frame, expected_tag):
    if len(frame) != REPLY_BYTES:
        raise ValueError("Incomplete FPGA reply.")
    magic, status, checksum = struct.unpack("<HHI", frame)
    if magic != REPLY_MAGIC or status >> 10 or crc32c(frame[:4]) != checksum:
        raise ValueError("Invalid FPGA reply.")
    if (status & 0xFF) != expected_tag:
        raise ValueError("FPGA reply tag differs from the in-flight request.")
    return Reply(expected_tag, status >> 8)


def telescope(rows, front, fallback, *, shot_ids=None):
    """Give the front end projected bytes and the fallback untouched Stim rows."""
    rows = np.asarray(rows)
    projected = project(rows)
    if shot_ids is not None:
        shot_ids = np.asarray(shot_ids)
        if shot_ids.shape != (len(rows),) or shot_ids.dtype != np.uint64:
            raise ValueError("Expected one uint64 shot ID per row.")
    status = np.asarray(front(projected))
    if status.shape != (len(rows),) or np.any(~np.isin(status, [0, 1, 2, 3])):
        raise ValueError("The FPGA front end must return one status code per row.")
    accepted = (status == ACCEPT_0) | (status == ACCEPT_1)
    logical = np.zeros(len(rows), dtype=np.uint8)
    logical[accepted] = status[accepted] - ACCEPT_0
    if (~accepted).any():
        ids = None if shot_ids is None else shot_ids[~accepted]
        decisions = np.asarray(fallback(rows[~accepted], ids))
        if decisions.shape != (int((~accepted).sum()),) or np.any(~np.isin(decisions, [0, 1])):
            raise ValueError("The fallback must return one logical bit per deferred row.")
        logical[~accepted] = decisions
    return logical, accepted
