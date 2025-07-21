class ExampleClass:
    """Example class demonstrating basic functionality."""

    def __init__(self, name: str = "default"):
        """Initialize with an optional name parameter."""
        self.name = name

    def example_method(self) -> str:
        """Return a formatted example string."""
        return f"example_method executed by {self.name}"

    def get_name(self) -> str:
        """Get the name of this instance."""
        return self.name
