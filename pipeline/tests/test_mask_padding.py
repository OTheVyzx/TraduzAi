from inpainter.mask_builder import glyph_padding


def test_glyph_padding_has_minimum_absolute_pad():
    assert glyph_padding(None) == 3
    assert glyph_padding(16) == 3
    assert glyph_padding(80) == 4
