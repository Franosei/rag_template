import logging
import sys
from pathlib import Path
from rich.logging import RichHandler
import json
from datetime import datetime

def setup_logging(level: str = "INFO", format_type: str = "pretty"):
    """Configure global logging"""
    
    if format_type == "pretty":
        # Development: Rich pretty-printed logs
        handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=True
        )
        formatter = logging.Formatter("%(message)s")
    else:
        # Production: JSON logs for parsing
        handler = logging.StreamHandler(sys.stdout)
        formatter = JSONFormatter()
    
    handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=True  # Override any existing config
    )
    
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for production"""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Include extra fields passed via extra={...}
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", 
                          "funcName", "levelname", "levelno", "lineno", 
                          "module", "msecs", "message", "pathname", 
                          "process", "processName", "relativeCreated", 
                          "thread", "threadName", "exc_info", "exc_text", 
                          "stack_info"]:
                log_data[key] = value
        
        return json.dumps(log_data)