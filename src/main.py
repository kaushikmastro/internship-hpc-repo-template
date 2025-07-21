from example_func import ExampleClass


def main():
    """Main function to demonstrate ExampleClass functionality."""
    # Create an instance of ExampleClass
    example_instance = ExampleClass("MainApp")

    # Execute the method and print the result
    result = example_instance.example_method()
    print(f"Result: {result}")

    # Also demonstrate getting the name
    name = example_instance.get_name()
    print(f"Instance name: {name}")

    return result


if __name__ == "__main__":
    main()
