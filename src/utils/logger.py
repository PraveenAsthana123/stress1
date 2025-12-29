#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GenAI-RAG-EEG Advanced Logging System
================================================================================

DESCRIPTION:
    This module provides a comprehensive logging system designed for research
    reproducibility and debugging. It outputs detailed information at multiple
    levels so that any person running the code can understand exactly what is
    happening at each step.

FEATURES:
    - Multi-level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Color-coded console output for easy reading
    - File logging with timestamps for reproducibility
    - Progress tracking with ETA estimation
    - Memory and GPU usage monitoring
    - Structured logging for machine parsing
    - Context managers for timed operations

WHY THIS MATTERS:
    When code runs on different systems, subtle issues can cause failures.
    Detailed logging helps:
    1. Identify exactly where failures occur
    2. Compare execution across different systems
    3. Debug data loading and preprocessing issues
    4. Track model training progress
    5. Document experiment configurations

USAGE:
    from src.utils.logger import setup_logger, get_logger

    # Setup at start of main script
    logger = setup_logger(
        name='genai_rag_eeg',
        log_file='experiment.log',
        level=LogLevel.DEBUG
    )

    # Use throughout code
    logger.info("Starting data loading...")
    logger.debug(f"Loaded {len(data)} samples")
    logger.warning("Missing values detected, using interpolation")
    logger.error("File not found: {path}")

Author: GenAI-RAG-EEG Team
Version: 3.0.0
License: MIT
================================================================================
"""

import logging
import sys
import os
import time
import json
import platform
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
from enum import Enum
from contextlib import contextmanager
from functools import wraps

# Try to import optional dependencies for enhanced features
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

class LogLevel(Enum):
    """
    Logging levels with descriptions of when to use each.

    DEBUG:    Detailed information for diagnosing problems.
              Use for: Variable values, loop iterations, data shapes

    INFO:     Confirmation that things are working as expected.
              Use for: Major steps, configuration, results

    WARNING:  Something unexpected happened, but code continues.
              Use for: Missing optional files, fallback behavior

    ERROR:    A serious problem that needs attention.
              Use for: Failed operations that affect results

    CRITICAL: System-level failure requiring immediate attention.
              Use for: Cannot continue execution
    """
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# ANSI color codes for terminal output
class Colors:
    """Terminal color codes for enhanced readability."""
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Log levels
    DEBUG = '\033[36m'      # Cyan
    INFO = '\033[32m'       # Green
    WARNING = '\033[33m'    # Yellow
    ERROR = '\033[31m'      # Red
    CRITICAL = '\033[41m'   # Red background

    # Special formatting
    HEADER = '\033[95m'     # Magenta
    SUCCESS = '\033[92m'    # Bright green
    TIMESTAMP = '\033[90m'  # Gray
    DATA = '\033[94m'       # Blue


# =============================================================================
# CUSTOM FORMATTER
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors and structured output to log messages.

    Format Structure:
    -----------------
    [TIMESTAMP] | LEVEL | MODULE:LINE | MESSAGE

    Example Output:
    ---------------
    [2024-12-29 10:30:45] | INFO     | data_loader:42  | Loading SAM-40 dataset...
    [2024-12-29 10:30:46] | DEBUG    | data_loader:55  | Found 40 subjects, 1200 samples
    [2024-12-29 10:30:47] | WARNING  | data_loader:78  | Subject S03 has missing channels
    """

    # Level-specific format strings
    LEVEL_COLORS = {
        logging.DEBUG: Colors.DEBUG,
        logging.INFO: Colors.INFO,
        logging.WARNING: Colors.WARNING,
        logging.ERROR: Colors.ERROR,
        logging.CRITICAL: Colors.CRITICAL
    }

    def __init__(self, use_colors: bool = True, show_path: bool = True):
        """
        Initialize formatter.

        Args:
            use_colors: Whether to use ANSI colors (disable for file output)
            show_path: Whether to show file:line information
        """
        self.use_colors = use_colors
        self.show_path = show_path

        # Base format without colors
        if show_path:
            fmt = '[%(asctime)s] | %(levelname)-8s | %(filename)s:%(lineno)-4d | %(message)s'
        else:
            fmt = '[%(asctime)s] | %(levelname)-8s | %(message)s'

        super().__init__(fmt, datefmt='%Y-%m-%d %H:%M:%S')

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and structure."""
        # Get base formatted message
        message = super().format(record)

        if self.use_colors and sys.stdout.isatty():
            # Apply color based on level
            color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)

            # Color the level indicator
            message = message.replace(
                f'| {record.levelname:<8}',
                f'| {color}{record.levelname:<8}{Colors.RESET}'
            )

            # Color the timestamp
            timestamp = self.formatTime(record, self.datefmt)
            message = message.replace(
                f'[{timestamp}]',
                f'{Colors.TIMESTAMP}[{timestamp}]{Colors.RESET}'
            )

        return message


# =============================================================================
# MAIN LOGGER CLASS
# =============================================================================

class GenAILogger:
    """
    Advanced logger for GenAI-RAG-EEG with research-oriented features.

    This logger provides:
    1. Detailed step-by-step output for debugging
    2. Progress tracking with time estimates
    3. System resource monitoring
    4. Experiment configuration logging
    5. Structured output for analysis

    Architecture:
    -------------
    GenAILogger
        │
        ├── Console Handler (colored output)
        │
        ├── File Handler (detailed logs)
        │
        └── JSON Handler (structured data)

    Example Usage:
    --------------
    >>> logger = GenAILogger('experiment', log_dir='logs')
    >>>
    >>> # Log configuration at start
    >>> logger.log_config({'batch_size': 64, 'lr': 0.001})
    >>>
    >>> # Track progress
    >>> with logger.progress("Training", total=100) as pbar:
    ...     for epoch in range(100):
    ...         pbar.update(1, loss=0.5)
    >>>
    >>> # Log results
    >>> logger.log_metrics({'accuracy': 99.0, 'f1': 0.985})
    """

    def __init__(
        self,
        name: str = 'genai_rag_eeg',
        log_dir: Optional[str] = None,
        level: LogLevel = LogLevel.INFO,
        log_to_file: bool = True,
        log_to_console: bool = True,
        json_logging: bool = False
    ):
        """
        Initialize logger with multiple handlers.

        Args:
            name: Logger name (used in output and filenames)
            log_dir: Directory for log files (default: ./logs)
            level: Minimum logging level
            log_to_file: Whether to write logs to file
            log_to_console: Whether to print to console
            json_logging: Whether to create JSON log file

        Common Issues:
        --------------
        - Permission denied: Check write access to log_dir
        - Encoding errors: Ensure UTF-8 terminal encoding
        - Missing colors: Terminal doesn't support ANSI codes
        """
        self.name = name
        self.level = level
        self.start_time = time.time()

        # Create base logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level.value)
        self.logger.handlers = []  # Clear any existing handlers

        # Setup log directory
        if log_dir is None:
            log_dir = Path.cwd() / 'logs'
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp for log files
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Add handlers
        if log_to_console:
            self._add_console_handler()

        if log_to_file:
            self._add_file_handler()

        if json_logging:
            self._add_json_handler()

        # Log initialization
        self._log_system_info()

    def _add_console_handler(self):
        """Add colored console output handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level.value)
        console_handler.setFormatter(ColoredFormatter(use_colors=True))
        self.logger.addHandler(console_handler)

    def _add_file_handler(self):
        """Add file handler for persistent logging."""
        log_file = self.log_dir / f'{self.name}_{self.session_id}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always capture all levels
        file_handler.setFormatter(ColoredFormatter(use_colors=False))
        self.logger.addHandler(file_handler)
        self.log_file = log_file

    def _add_json_handler(self):
        """Add JSON handler for structured logging."""
        self.json_file = self.log_dir / f'{self.name}_{self.session_id}.json'
        self.json_records = []

    def _log_system_info(self):
        """Log system information for reproducibility."""
        self.info("=" * 70)
        self.info(f"GenAI-RAG-EEG Logger Initialized")
        self.info("=" * 70)
        self.info(f"Session ID: {self.session_id}")
        self.info(f"Log Directory: {self.log_dir}")
        self.info(f"Log Level: {self.level.name}")

        # System information
        self.info("")
        self.info("System Information:")
        self.info(f"  OS: {platform.system()} {platform.release()}")
        self.info(f"  Python: {platform.python_version()}")
        self.info(f"  Platform: {platform.machine()}")

        if HAS_PSUTIL:
            mem = psutil.virtual_memory()
            self.info(f"  RAM: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
            self.info(f"  CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")

        if HAS_TORCH:
            self.info(f"  PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                self.info(f"  CUDA: {torch.version.cuda}")
                self.info(f"  GPU: {torch.cuda.get_device_name(0)}")
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.info(f"  GPU Memory: {gpu_mem:.1f} GB")
            else:
                self.info("  CUDA: Not available (CPU mode)")

        self.info("=" * 70)
        self.info("")

    # =========================================================================
    # BASIC LOGGING METHODS
    # =========================================================================

    def debug(self, msg: str, **kwargs):
        """
        Log debug-level message.

        Use for: Detailed diagnostic information
        - Variable values during computation
        - Loop iterations and indices
        - Data shapes and types
        - Function entry/exit points

        Example:
            logger.debug(f"Processing batch {i}/{total}, shape: {x.shape}")
        """
        self.logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs):
        """
        Log info-level message.

        Use for: Major operational steps
        - Starting/completing phases
        - Configuration summaries
        - Results and metrics
        - File operations

        Example:
            logger.info("Training complete. Best accuracy: 99.0%")
        """
        self.logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        """
        Log warning-level message.

        Use for: Unexpected but handled situations
        - Missing optional files (using defaults)
        - Deprecated features
        - Suboptimal configurations
        - Data quality issues

        Example:
            logger.warning(f"Missing channel {ch}, using interpolation")
        """
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, exc_info: bool = False, **kwargs):
        """
        Log error-level message.

        Use for: Serious problems affecting results
        - Failed file operations
        - Invalid data formats
        - Model loading failures
        - API errors

        Args:
            msg: Error message
            exc_info: Whether to include exception traceback

        Example:
            logger.error(f"Failed to load {file}: {e}", exc_info=True)
        """
        self.logger.error(msg, exc_info=exc_info, **kwargs)

    def critical(self, msg: str, **kwargs):
        """
        Log critical-level message.

        Use for: System-level failures
        - Cannot continue execution
        - Data corruption
        - Security issues

        Example:
            logger.critical("Database connection lost, cannot save results")
        """
        self.logger.critical(msg, **kwargs)

    # =========================================================================
    # SPECIALIZED LOGGING METHODS
    # =========================================================================

    def log_config(self, config: Dict[str, Any], title: str = "Configuration"):
        """
        Log configuration parameters in a structured format.

        This is essential for reproducibility - other users can see
        exactly what parameters were used.

        Args:
            config: Dictionary of configuration parameters
            title: Section title

        Example:
            logger.log_config({
                'batch_size': 64,
                'learning_rate': 0.001,
                'epochs': 100,
                'dataset': 'SAM-40'
            })

        Output:
            ┌─────────────────────────────────────┐
            │ Configuration                       │
            ├─────────────────────────────────────┤
            │ batch_size      : 64                │
            │ learning_rate   : 0.001             │
            │ epochs          : 100               │
            │ dataset         : SAM-40            │
            └─────────────────────────────────────┘
        """
        self.info("")
        self.info(f"┌{'─' * 50}┐")
        self.info(f"│ {title:<48} │")
        self.info(f"├{'─' * 50}┤")

        for key, value in config.items():
            # Format value based on type
            if isinstance(value, float):
                val_str = f"{value:.6g}"
            elif isinstance(value, (list, tuple)):
                val_str = str(value)[:30] + ('...' if len(str(value)) > 30 else '')
            else:
                val_str = str(value)

            self.info(f"│ {key:<20}: {val_str:<27} │")

        self.info(f"└{'─' * 50}┘")
        self.info("")

    def log_metrics(self, metrics: Dict[str, float], title: str = "Metrics"):
        """
        Log performance metrics in a clear format.

        Args:
            metrics: Dictionary of metric name -> value
            title: Section title

        Example:
            logger.log_metrics({
                'accuracy': 99.0,
                'precision': 0.988,
                'recall': 0.992,
                'f1_score': 0.990,
                'auc_roc': 0.995
            })
        """
        self.info("")
        self.info(f"{'─' * 40}")
        self.info(f"{title}")
        self.info(f"{'─' * 40}")

        for metric, value in metrics.items():
            if isinstance(value, float):
                if value < 1:  # Likely a ratio/proportion
                    self.info(f"  {metric:<20}: {value:.4f}")
                else:  # Likely a percentage
                    self.info(f"  {metric:<20}: {value:.2f}%")
            else:
                self.info(f"  {metric:<20}: {value}")

        self.info(f"{'─' * 40}")
        self.info("")

    def log_data_summary(
        self,
        dataset_name: str,
        n_samples: int,
        n_features: int,
        n_classes: int,
        class_distribution: Optional[Dict[str, int]] = None
    ):
        """
        Log dataset summary information.

        Args:
            dataset_name: Name of the dataset
            n_samples: Total number of samples
            n_features: Number of features per sample
            n_classes: Number of classes
            class_distribution: Optional dict of class -> count

        Example:
            logger.log_data_summary(
                dataset_name='SAM-40',
                n_samples=1200,
                n_features=16384,  # 32 channels × 512 time points
                n_classes=2,
                class_distribution={'stress': 600, 'baseline': 600}
            )
        """
        self.info("")
        self.info(f"Dataset: {dataset_name}")
        self.info(f"  Total samples  : {n_samples:,}")
        self.info(f"  Features/sample: {n_features:,}")
        self.info(f"  Classes        : {n_classes}")

        if class_distribution:
            self.info("  Class Distribution:")
            for cls, count in class_distribution.items():
                pct = 100 * count / n_samples
                self.info(f"    {cls:<15}: {count:>6} ({pct:.1f}%)")
        self.info("")

    def log_step(self, step: int, total: int, message: str, **metrics):
        """
        Log a progress step with optional metrics.

        Args:
            step: Current step number (1-indexed)
            total: Total number of steps
            message: Description of current step
            **metrics: Optional metrics to display

        Example:
            logger.log_step(25, 100, "Training epoch", loss=0.123, acc=95.5)

        Output:
            [Step 25/100] Training epoch | loss: 0.123, acc: 95.5%
        """
        pct = 100 * step / total
        elapsed = time.time() - self.start_time
        eta = (elapsed / step) * (total - step) if step > 0 else 0

        metric_str = ""
        if metrics:
            parts = []
            for k, v in metrics.items():
                if isinstance(v, float):
                    parts.append(f"{k}: {v:.4f}")
                else:
                    parts.append(f"{k}: {v}")
            metric_str = " | " + ", ".join(parts)

        self.info(f"[Step {step}/{total}] ({pct:.1f}%) {message}{metric_str} | ETA: {eta:.0f}s")

    @contextmanager
    def timed_operation(self, operation_name: str):
        """
        Context manager for timing operations.

        Usage:
            with logger.timed_operation("Data loading"):
                data = load_data()
            # Automatically logs: "Data loading completed in 5.23s"
        """
        self.info(f"Starting: {operation_name}")
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.info(f"Completed: {operation_name} in {elapsed:.2f}s")

    def log_exception(self, e: Exception, context: str = ""):
        """
        Log exception with full traceback.

        Args:
            e: Exception object
            context: Additional context about where error occurred
        """
        self.error(f"Exception in {context}: {type(e).__name__}: {str(e)}")
        self.debug(f"Full traceback:\n{traceback.format_exc()}")

    def separator(self, char: str = "=", length: int = 70):
        """Print a separator line."""
        self.info(char * length)


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================

_default_logger: Optional[GenAILogger] = None


def setup_logger(
    name: str = 'genai_rag_eeg',
    log_dir: Optional[str] = None,
    level: LogLevel = LogLevel.INFO,
    **kwargs
) -> GenAILogger:
    """
    Setup and return the global logger instance.

    This should be called once at the start of your main script.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Minimum logging level
        **kwargs: Additional arguments passed to GenAILogger

    Returns:
        Configured GenAILogger instance

    Example:
        # In main.py
        from src.utils.logger import setup_logger, LogLevel

        logger = setup_logger(
            name='experiment',
            log_dir='./logs',
            level=LogLevel.DEBUG
        )
    """
    global _default_logger
    _default_logger = GenAILogger(name=name, log_dir=log_dir, level=level, **kwargs)
    return _default_logger


def get_logger() -> GenAILogger:
    """
    Get the global logger instance.

    If setup_logger() hasn't been called, creates a default logger.

    Returns:
        GenAILogger instance

    Example:
        # In any module
        from src.utils.logger import get_logger

        logger = get_logger()
        logger.info("Processing data...")
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = GenAILogger()
    return _default_logger


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the logging system."""

    print("Testing GenAI-RAG-EEG Logger\n")

    # Setup logger
    logger = setup_logger(
        name='test_logger',
        level=LogLevel.DEBUG
    )

    # Test basic logging
    logger.debug("This is a DEBUG message - detailed diagnostic info")
    logger.info("This is an INFO message - major operation steps")
    logger.warning("This is a WARNING message - unexpected but handled")
    logger.error("This is an ERROR message - serious problem")

    # Test configuration logging
    logger.log_config({
        'batch_size': 64,
        'learning_rate': 0.001,
        'epochs': 100,
        'model': 'CNN-LSTM-Attention',
        'dataset': 'SAM-40'
    })

    # Test metrics logging
    logger.log_metrics({
        'accuracy': 99.0,
        'precision': 0.988,
        'recall': 0.992,
        'f1_score': 0.990
    })

    # Test timed operation
    with logger.timed_operation("Sample operation"):
        import time
        time.sleep(0.5)

    # Test progress steps
    for i in range(1, 6):
        logger.log_step(i, 5, "Processing batch", loss=0.5/i, acc=80+i*4)
        time.sleep(0.1)

    logger.separator()
    logger.info("Logger test completed successfully!")
