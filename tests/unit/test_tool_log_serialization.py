"""Provider LOB support must not chase arbitrary reader chains indefinitely."""

import pytest

from memorizz.memagent.utils.tool_log import _to_jsonable


@pytest.mark.parametrize("content", ["stored text", b"stored text"])
def test_lob_content_is_read_inside_nested_tool_results(content):
    class Lob:
        def read(self):
            return content

    assert _to_jsonable({"rows": [(Lob(),)]}) == {"rows": [["stored text"]]}


@pytest.mark.parametrize("returns_self", [True, False])
def test_non_text_reader_is_not_read_recursively(returns_self):
    reads = []

    class Reader:
        def read(self):
            reads.append(self)
            if len(reads) > 1:
                pytest.fail("The serializer followed a non-text reader chain")
            return self if returns_self else Reader()

        def __str__(self):
            return "opaque reader"

    value = Reader()
    assert _to_jsonable(value) == "opaque reader"
    assert reads == [value]
