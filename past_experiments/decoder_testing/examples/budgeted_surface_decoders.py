"""Edit these constants to budget both decoder factories; fingerprinted by qecbench."""

from surface_decoders import LUTDecoder, RelayBPConfig, RelayBPDecoder, ResourceLimits

# Supply a measured operation rate for your platform before setting a latency budget.
LIMITS = ResourceLimits(
    max_memory_mib=256,
    latency_budget_us=100_000,
    operations_per_second=100_000_000,
    max_build_operations=200_000_000,
)
RELAY = RelayBPConfig(legs=8, iterations=30, chunk_shots=32)


def relay_bp(context):
    return RelayBPDecoder(context, RELAY, LIMITS)


def lut(context):
    return LUTDecoder(context, limits=LIMITS)
