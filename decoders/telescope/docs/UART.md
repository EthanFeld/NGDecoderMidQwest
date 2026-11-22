# FPGA UART protocol

The Tang Nano 20K endpoint uses 1,000,000 baud, 8N1, with one outstanding request.
Body integers are little-endian. The [host](../host.py) and
[RTL endpoint](../rtl/link.sv) implement the protocol.

## Framing

Append CRC-16/CCITT-FALSE to the body, high byte first: polynomial `0x1021`,
initial value `0xffff`, no reflection, final XOR zero. Escape `0x7e` and `0x7d`
by sending `0x7d` followed by the byte XOR `0x20`. Surround the escaped body
and CRC with `0x7e`.

## Request body

| Offset | Bytes | Field |
| ---: | ---: | --- |
| 0 | 1 | Opcode `0x45` for the telescope path |
| 1 | 2 | Host token |
| 3 | 4 | Compiled image tag |
| 7 | 224 | Packed detector slices |

The default d15 profile has 16 slices of 112 detector bits. Each slice occupies
14 bytes, oldest slice first. Detector `i` occupies bit `i % 8` of byte `i // 8`
within its slice. Coordinates follow the profile's `coordinates` array.

## Reply body

| Offset | Bytes | Field |
| ---: | ---: | --- |
| 0 | 1 | Opcode `0x92` |
| 1 | 2 | Echoed token |
| 3 | 4 | Compiled image tag |
| 7 | 1 | Status: `0x01` accept, `0x03` defer |
| 8 | 2 | FPGA decode cycles |
| 10 | 1 | Logical bit or deferred logical frame |
| 11 | 1 | Residual byte count: zero on accept |
| 12 | Variable | Packed residual on defer |

The host validates CRC, tag, token, status, payload length and padding. A deferred
residual must be nonempty and a subset of the input. A valid FPGA accept supplies
the prediction; defers and communication failures enter the host tensor backend.
Core time in microseconds is `cycles / 81` for the default 81 MHz image.
