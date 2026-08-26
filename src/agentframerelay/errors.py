"""Public exceptions raised by AgentFrameRelay's framework-neutral core."""


class AgentFrameRelayError(Exception):
    """Base class for AgentFrameRelay errors."""


class ToolError(AgentFrameRelayError):
    """Base class for errors raised while validating or executing a tool."""


class ToolInputError(ToolError):
    """Raised when supplied tool arguments do not match its signature or types."""


class ToolExecutionError(ToolError):
    """Raised when the underlying tool function raises an exception."""


class AsyncToolError(ToolError):
    """Raised when an asynchronous tool is invoked from an active event loop."""
