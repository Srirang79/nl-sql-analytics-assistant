import pytest

from tests.fixture_data import build_fixture_db


@pytest.fixture(scope="session")
def fixture_db(tmp_path_factory):
    return build_fixture_db(tmp_path_factory.mktemp("db") / "fixture.db")
