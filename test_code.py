"""Main pytest script"""

from src.tests.test_encoder import test_type_encoder, test_shape_encoder
from src.tests.test_decoder import test_type_decoder, test_shape_decoder
from src.tests.test_clip import test_type_clip, test_shape_clip

test_type_encoder()
test_shape_encoder()

test_type_decoder()
test_shape_decoder()

test_type_clip()
test_shape_clip()
