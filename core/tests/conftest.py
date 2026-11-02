"""The library's tests: their autouse fixtures are library_fixtures.py's, which the CLI's tests
(apps/seedvr2x/tests/conftest.py) take as well. A fixture is registered by being imported into a
conftest, hence the imports and their noqa."""

from library_fixtures import no_driver_asked, one_to_one_clock  # noqa: F401
