import numpy as np


class MutatingDecoder:
    def decode_batch(self, detection_events):
        detection_events[0, 0] = 0


def mutating(context):
    return MutatingDecoder()


class WrongShapeDecoder:
    def decode_batch(self, detection_events):
        return np.zeros((len(detection_events), 2), dtype=np.uint8)


def wrong_shape(context):
    return WrongShapeDecoder()
