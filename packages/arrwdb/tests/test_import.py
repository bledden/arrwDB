"""Top-level import smoke tests for the arrwdb package."""


def test_version_exposed():
    import arrwdb

    assert hasattr(arrwdb, "__version__")
    assert isinstance(arrwdb.__version__, str)
    assert arrwdb.__version__


def test_client_classes_exposed():
    from arrwdb import (
        ArrwDBClient,
        ArrwDBException,
        AuthenticationError,
        NotFoundError,
        RateLimitError,
        ServerError,
        ValidationError,
    )

    assert ArrwDBClient is not None
    # Exception hierarchy
    for exc in (
        NotFoundError,
        ValidationError,
        AuthenticationError,
        RateLimitError,
        ServerError,
    ):
        assert issubclass(exc, ArrwDBException)


def test_integrations_namespace_exists():
    """The integrations namespace loads without any optional deps."""
    import arrwdb.integrations  # noqa: F401
