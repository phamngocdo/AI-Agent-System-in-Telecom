from abc import ABC, abstractmethod
from pathlib import Path


class BaseCleaner(ABC):
    """Abstract base class for all text and document cleaners.

    This class provides a common interface and scaffolding for document
    cleaners, standardizing initialization and the main execution pipeline.
    Subclasses must implement the ``run`` method with document-specific logic.

    Attributes:
        input_path (Path): Path to the input directory or file.
        output_path (Path): Path to the output directory or file.
    """

    def __init__(self, input_dir: str, output_file: str) -> None:
        """Initialize the cleaner with input and output paths.

        Args:
            input_dir: Directory containing input raw files.
            output_file: File or directory path for the cleaned output.
        """
        self.input_path = Path(input_dir)
        self.output_path = Path(output_file)

    @abstractmethod
    def run(self) -> None:
        """Execute the full cleaning pipeline.

        Subclasses must handle reading inputs, applying cleaning operations,
        and writing results to ``output_path``.
        """
        pass