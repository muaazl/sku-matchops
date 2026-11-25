"""Framework-agnostic domain exceptions mapped to HTTP or gRPC status codes at transport boundaries."""


class ServiceValidationError(ValueError):
    """Raised when input arguments violate domain validation rules (HTTP 400 / gRPC INVALID_ARGUMENT)."""


class UnauthorizedPortalError(PermissionError):
    """Raised when an external portal rejects caller authentication credentials (HTTP 401)."""


class ResourceNotFoundError(KeyError):
    """Raised when a requested resource does not exist (HTTP 404 / gRPC NOT_FOUND)."""


class ResourceConflictError(RuntimeError):
    """Raised when an operation conflicts with the resource's current state (HTTP 409)."""


class PayloadTooLargeError(ValueError):
    """Raised when an uploaded payload exceeds maximum configured bounds (HTTP 413)."""


class UpstreamGatewayError(RuntimeError):
    """Raised when an external upstream service fails unexpectedly (HTTP 502)."""


class ServiceUnavailableError(RuntimeError):
    """Raised when a required broker or backing subsystem is unreachable (HTTP 503)."""
