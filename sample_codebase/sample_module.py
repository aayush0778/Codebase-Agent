"""
Sample codebase for testing the RAG pipeline.
This file contains a small set of functions and classes to verify
that the AST chunker, indexer, and query engine work end-to-end.
"""

import math
from typing import List, Optional


class Calculator:
    """A simple calculator that supports basic arithmetic operations."""

    def __init__(self):
        self.history: List[str] = []

    def add(self, a: float, b: float) -> float:
        """Add two numbers and return the result."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a and return the result."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers and return the result."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def divide(self, a: float, b: float) -> Optional[float]:
        """Divide a by b. Returns None if b is zero."""
        if b == 0:
            self.history.append(f"{a} / {b} = ERROR (division by zero)")
            return None
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result

    def get_history(self) -> List[str]:
        """Return the history of all operations performed."""
        return self.history.copy()

    def clear_history(self):
        """Clear the operation history."""
        self.history.clear()


class StatisticsEngine:
    """Computes basic statistical measures on a list of numbers."""

    def __init__(self, data: List[float]):
        if not data:
            raise ValueError("Data list cannot be empty.")
        self.data = data

    def mean(self) -> float:
        """Calculate the arithmetic mean of the data."""
        return sum(self.data) / len(self.data)

    def median(self) -> float:
        """Calculate the median value of the data."""
        sorted_data = sorted(self.data)
        n = len(sorted_data)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_data[mid - 1] + sorted_data[mid]) / 2
        return sorted_data[mid]

    def variance(self) -> float:
        """Calculate the population variance of the data."""
        avg = self.mean()
        return sum((x - avg) ** 2 for x in self.data) / len(self.data)

    def std_dev(self) -> float:
        """Calculate the population standard deviation."""
        return math.sqrt(self.variance())

    def min_max(self) -> tuple:
        """Return the (min, max) tuple of the data."""
        return (min(self.data), max(self.data))


def fibonacci(n: int) -> List[int]:
    """Generate the first n Fibonacci numbers.

    Args:
        n: How many Fibonacci numbers to generate.

    Returns:
        A list of the first n Fibonacci numbers.
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
    sequence = [0, 1]
    for _ in range(2, n):
        sequence.append(sequence[-1] + sequence[-2])
    return sequence


def is_prime(n: int) -> bool:
    """Check if a number is prime.

    Args:
        n: The integer to check.

    Returns:
        True if n is prime, False otherwise.
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


async def fetch_data(source: str) -> dict:
    """Simulate fetching data from a local source asynchronously.

    Args:
        source: The data source identifier.

    Returns:
        A dict with the fetched data.
    """
    # Simulated async data fetch (no actual network call)
    import asyncio
    await asyncio.sleep(0.1)
    return {
        "source": source,
        "status": "ok",
        "records": 42,
    }


def binary_search(arr: List[int], target: int) -> int:
    """Perform binary search on a sorted list.

    Args:
        arr: A sorted list of integers.
        target: The value to search for.

    Returns:
        The index of the target if found, otherwise -1.
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
