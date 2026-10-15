"""Unit tests (CPU only). The same checks run at the start of every GPU job."""
from rag_audit import selftest


def test_selftest():
    selftest.main()
