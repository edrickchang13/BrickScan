"""
Structured logging configuration for production and development.

In development: Colored human-readable output to console
In production: JSON-formatted logs with request context
"""

import logging
import json
import sys
from datetime import datetime
from typing import Optional


class JsonFormatter(logging.Formatter):
    """
    JSON log formatter for production environments.
    Outputs structured logs that can be parsed by log aggregation services.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
        - record: LogRecord from logging system

        Returns:
        - JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include optional context fields
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id

        if hasattr(record, "endpoint"):
            log_data["endpoint"] = record.endpoint

        if hasattr(record, "method"):
            log_data["method"] = record.method

        if hasattr(record, "status_code"):
            log_data["status_code"] = record.status_code

        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms

        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            log_data["exception_type"] = record.exc_info[0].__name__

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for development.
    Makes logs easier to read during development.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.

        Args:
        - record: LogRecord from logging system

        Returns:
        - Colored log string
        """
        color = self.COLORS.get(record.levelname, self.RESET)
        log_format = f"{color}[%(levelname)-8s]{self.RESET} %(asctime)s - %(name)s - %(message)s"

        # Format the base message
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logging(
    debug: bool = False,
    log_level: str = "INFO",
    include_modules: Optional[list[str]] = None,
) -> None:
    """
    Configure application-wide logging.

    Args:
    - debug: Enable debug mode with colored output and DEBUG level
    - log_level: Minimum log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    - include_modules: List of module names to enable detailed logging for
                       (e.g., ["sqlalchemy", "httpx"]). Default: empty list
    """
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Determine actual log level
    if debug:
        actual_level = logging.DEBUG
        formatter_class = ColoredFormatter
    else:
        actual_level = getattr(logging, log_level.upper(), logging.INFO)
        formatter_class = JsonFormatter

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(actual_level)

    # Set formatter
    formatter = formatter_class()
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger.setLevel(actual_level)
    root_logger.addHandler(console_handler)

    # Configure app-specific loggers
    app_logger = logging.getLogger("app")
    app_logger.setLevel(actual_level)

    # Configure module-specific loggers if in debug mode
    if debug:
        if include_modules is None:
            include_modules = []

        for module_name in include_modules:
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(logging.DEBUG)

    # Suppress noisy third-party loggers in production
    if not debug:
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Log startup info
    app_logger.info(
        f"Logging configured: level={log_level}, debug={debug}, format={'json' if not debug else 'colored'}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
    - name: Module name (typically __name__)

    Returns:
    - Logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for adding request-scoped context to logs.

    Useful for correlating logs from the same request across services.

    Example:
    ```python
    async def handle_request(request_id: str, user_id: str):
        with LogContext(request_id=request_id, user_id=user_id):
            logger.info("Processing request")  # Will include request_id and user_id
    ```
    """

    def __init__(self, **context):
        """
        Initialize context with key-value pairs.

        Args:
        - **context: Context fields (request_id, user_id, endpoint, etc.)
        """
        self.context = context
        self._previous_context = {}

    def __enter__(self):
        """Enter context manager"""
        # Store previous context values
        root_logger = logging.getLogger()
        for key in self.context:
            self._previous_context[key] = getattr(root_logger, f"_{key}", None)
            setattr(root_logger, f"_{key}", self.context[key])

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        # Restore previous context
        root_logger = logging.getLogger()
        for key, value in self._previous_context.items():
            if value is None:
                delattr(root_logger, f"_{key}")
            else:
                setattr(root_logger, f"_{key}", value)

        return False


# Convenience functions for common logging patterns
def log_request(request_id: str, method: str, endpoint: str, **kwargs):
    """Log incoming request"""
    logger = get_logger("app.request")
    logger.info(
        f"{method} {endpoint}",
        extra={
            "request_id": request_id,
            "endpoint": endpoint,
            "method": method,
            **kwargs,
        },
    )


def log_response(request_id: str, status_code: int, duration_ms: float, **kwargs):
    """Log outgoing response"""
    logger = get_logger("app.response")
    level = logging.INFO if 200 <= status_code < 400 else logging.WARNING
    logger.log(
        level,
        f"Response {status_code} ({duration_ms}ms)",
        extra={
            "request_id": request_id,
            "status_code": status_code,
            "duration_ms": duration_ms,
            **kwargs,
        },
    )


def log_error(error: Exception, context: str = "", **kwargs):
    """Log error with context"""
    logger = get_logger("app.error")
    logger.exception(
        f"Error in {context}: {str(error)}",
        extra=kwargs,
    )
