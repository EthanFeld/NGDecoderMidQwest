#!/usr/bin/env python3
"""Run the Tang Nano surface-code telescope and decode defers on the CPU."""

import argparse
from binascii import crc_hqx
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import time

import numpy as np

from .reference import Predecoder, matching, sample
from .growth_reference import GrowthModel
from .surface_model import generate, seal

FLAG = 0x7e
ESC = 0x7d
REQUEST = struct.Struct('<BHI')
REPLY = struct.Struct('<BHIBHBB')
COMMAND = {'tilt': 0x45, 'flat': 0x46}


@dataclass(frozen=True)
class Result:
    logical: bool
    cycles: int
    residual: bytes | None


def pack(syndrome, width):
    return np.packbits(np.asarray(syndrome).reshape(-1, width), axis=1, bitorder='little').tobytes()


def unpack(raw, width, count):
    depth = (width+7)//8
    if len(raw) != count*depth:
        raise ValueError('wrong wire syndrome length')
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8).reshape(count, depth), axis=1, bitorder='little')
    if bits[:, width:].any():
        raise ValueError('nonzero padding in wire syndrome')
    return bits[:, :width].reshape(-1).astype(np.bool_)


def frame(body):
    out = bytearray([FLAG])
    for b in body + crc_hqx(body, 0xffff).to_bytes(2, 'big'):
        if b in (FLAG, ESC):
            out.extend((ESC, b ^ 0x20))
        else:
            out.append(b)
    out.append(FLAG)
    return bytes(out)


class Inbox:
    def __init__(self):
        self.partial = bytearray()
        self.active = False
        self.escape = False

    def feed(self, octets):
        for b in octets:
            if b == FLAG:
                body = None
                if self.active and not self.escape and len(self.partial) >= 2 and crc_hqx(self.partial, 0xffff) == 0:
                    body = bytes(self.partial[:-2])
                self.active, self.escape = True, False
                self.partial.clear()
                if body is not None:
                    yield body
            elif self.active:
                if self.escape:
                    self.partial.append(b ^ 0x20)
                    self.escape = False
                elif b == ESC:
                    self.escape = True
                else:
                    self.partial.append(b)
                if len(self.partial) > 256:
                    self.active, self.escape = False, False
                    self.partial.clear()


def parse_reply(body, tag, width, count, max_clocks):
    if len(body) < REPLY.size:
        raise ValueError('short reply from FPGA')
    opcode, token, image, status, cycles, bit, nbytes = REPLY.unpack_from(body)
    wanted = count*((width+7)//8) if status == 3 else 0
    if (opcode != 0x92 or image != tag or status not in (1, 3) or bit not in (0, 1) or
            not 0 < cycles <= max_clocks or nbytes != wanted or len(body) != REPLY.size+nbytes):
        raise ValueError('invalid reply or mismatched bitstream')
    residual = body[REPLY.size:] if wanted else None
    if residual is not None and not unpack(residual, width, count).any():
        raise ValueError('zero residual on FPGA defer')
    return token, Result(bool(bit), cycles, residual)


class Link:
    def __init__(self, port, image, max_clocks):
        self.port = port
        self.tag = seal(image)
        self.width = image['width']
        self.rows = image['rounds']+1
        self.max_clocks = max_clocks
        self.token = 0xffff
        self.inbox = Inbox()

    def run(self, syndrome, mode):
        packet = pack(syndrome, self.width)
        if len(packet) != ((self.width+7)//8)*self.rows:
            raise ValueError('input does not match the loaded profile')
        self.token = (self.token+1) & 0xffff
        wire = frame(REQUEST.pack(COMMAND[mode], self.token, self.tag) + packet)
        if self.port.write(wire) != len(wire):
            raise OSError('short serial write')
        deadline = time.monotonic()+1.0
        while time.monotonic() < deadline:
            for response in self.inbox.feed(self.port.read(max(self.port.in_waiting, 1))):
                if len(response) >= 3 and int.from_bytes(response[1:3], 'little') == self.token:
                    _, result = parse_reply(response, self.tag, self.width, self.rows, self.max_clocks)
                    return result
        self.inbox = Inbox()
        raise TimeoutError('FPGA did not reply. Check the profile and serial port.')


def run(image, args, connection):
    physical = image if args.noise_p is None else generate(
        image['distance'], image['basis'], image['rounds'], args.noise_p)
    local, cpu = Predecoder(image), matching(physical)
    # PyMatching builds its native search graph on the first decode.
    warmup = np.zeros(image['checks'], dtype=np.bool_)
    warmup[0] = True
    cpu.decode(warmup)
    width, rows = image['width'], image['rounds']+1
    budget = image.get('growth_cycles', 0)
    max_cycles = local.plan.max_clocks + (budget+1 if budget else 0)
    link = Link(connection, image, max_cycles) if connection else None
    growth = None
    if budget and link is None and args.mode == 'tilt':
        topology = args.graph.with_suffix('.growth.txt')
        if not topology.exists():
            topology = args.graph.with_name('growth_graph.txt')
        growth = GrowthModel(topology)
    if args.syndrome:
        syndromes = [unpack(bytes.fromhex(args.syndrome), width, rows)]
        labels = [None]
    else:
        syndromes, labels = sample(physical, args.shots, np.random.default_rng(args.seed))
    for i, (syndrome, expected) in enumerate(zip(syndromes, labels, strict=True)):
        error = None
        start = time.perf_counter_ns()
        if link:
            try:
                prediction = link.run(syndrome, args.mode)
            except (OSError, ValueError) as exc:
                error = str(exc)
                prediction = None
        else:
            soft = local.decode(syndrome[None, :], tilt=args.mode == 'tilt')
            if growth is not None:
                growth.recover(soft, syndrome[None, :], budget)
            prediction = Result(bool(soft.logical[0]), int(soft.clocks[0]),
                                None if soft.accepted[0] else pack(soft.residual[0], width))
        link_us = (time.perf_counter_ns()-start)//1000 if link else None
        residual = None
        if prediction and prediction.residual:
            residual = unpack(prediction.residual, width, rows)
            if np.any(residual & ~syndrome):
                error = 'FPGA residual has detectors outside the input'
                prediction = None
                residual = None
        cpu_time = None
        if prediction is None or residual is not None:
            start = time.perf_counter_ns()
            if residual is not None and args.tail == 'residual':
                logical = bool(cpu.decode(residual)[0]) ^ prediction.logical
            else:
                logical = bool(cpu.decode(syndrome)[0])
            cpu_time = (time.perf_counter_ns()-start)/1000
        else:
            logical = prediction.logical
        print(json.dumps({
            'shot': i, 'result': int(logical), 'expected': None if expected is None else int(expected),
            'fpga_cycles': prediction.cycles if prediction else None,
            'deferred': residual is not None, 'residual_weight': int(residual.sum()) if residual is not None else 0,
            'cpu_us': cpu_time, 'link_us': link_us, 'error': error
        }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('--port', help='leave unset for a software demonstration')
    parser.add_argument('--baud', type=int, default=1_000_000, help='UART baud rate for the loaded image')
    parser.add_argument('--mode', choices=COMMAND, default='tilt')
    parser.add_argument('--tail', choices=('original', 'residual'), default='original')
    parser.add_argument('--shots', type=int, default=10)
    parser.add_argument('--seed', type=int, default=37515)
    parser.add_argument('--noise-p', type=float, help='set CPU weights and software sampling at a different noise rate')
    parser.add_argument('--syndrome', help='use one row-padded syndrome in hexadecimal')
    args = parser.parse_args()
    if args.shots <= 0:
        parser.error('positive shot count required')
    image = json.loads(args.graph.read_text())
    if args.port:
        import serial
        with serial.Serial(args.port, args.baud, timeout=.025, write_timeout=.4) as port:
            port.reset_input_buffer()
            run(image, args, port)
    else:
        run(image, args, None)


if __name__ == '__main__':
    main()
