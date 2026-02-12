"""Abstract base class for formal languages."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Detailed validation result for a string."""

    is_valid: bool
    error_message: str | None = None
    error_position: int | None = None


class FormalLanguage(ABC):
    """Abstract base class for formal languages."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the language."""
        pass

    @property
    @abstractmethod
    def vocab(self) -> list[str]:
        """List of valid characters in the language (excluding special tokens)."""
        pass

    @abstractmethod
    def generate(self, min_n: int = 1, max_n: int = 10) -> str:
        """Generate a random valid string from the language.

        Args:
            min_n: Minimum parameter for generation (e.g., min length parameter)
            max_n: Maximum parameter for generation

        Returns:
            A valid string from the language
        """
        pass

    @abstractmethod
    def validate(self, s: str) -> bool:
        """Check if a string is in the language.

        Args:
            s: String to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def validate_detailed(self, s: str) -> ValidationResult:
        """Validate a string with detailed error information.

        Args:
            s: String to validate

        Returns:
            ValidationResult with validity and error details
        """
        pass

    def generate_batch(self, n: int, min_n: int = 1, max_n: int = 10) -> list[str]:
        """Generate multiple valid strings.

        Args:
            n: Number of strings to generate
            min_n: Minimum parameter for generation
            max_n: Maximum parameter for generation

        Returns:
            List of valid strings
        """
        return [self.generate(min_n, max_n) for _ in range(n)]
