"""The server's tests: the library's autouse fixtures (core/tests/library_fixtures.py, on this
suite's pythonpath), the stand-in models (core/tests/stand_ins.py) and the server's own
(serving.py)."""

from library_fixtures import no_driver_asked, one_to_one_clock  # noqa: F401
from stand_ins import cpu_models  # noqa: F401
