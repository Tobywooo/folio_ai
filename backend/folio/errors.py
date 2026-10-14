"""Exception types the HTTP layer maps to status codes."""


class FolioError(Exception):
    """Base class. The message is safe to show to the user."""

    status_code = 500


class ConfigError(FolioError):
    status_code = 500


class ValidationError(FolioError):
    status_code = 400


class NotFoundError(FolioError):
    status_code = 404


class IngestionError(FolioError):
    """A source could not be extracted, embedded, or indexed."""

    status_code = 422


class ModelAPIError(FolioError):
    """The LLM or embedding API failed or returned something unusable."""

    status_code = 502
