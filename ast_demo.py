import ast
import textwrap

def show_tree(code:str) -> None:
    """
    Parses the given code into an AST and prints its structure.

    Args:
        code (str): The source code to parse.
    """
    # Dedent the code to remove any leading whitespace
    dedented_code = textwrap.dedent(code)
    
    # Parse the code into an AST
    tree = ast.parse(dedented_code)
    
    # Print the AST structure
    print(ast.dump(tree, indent=4))

if __name__ == "__main__":
    # Example code to demonstrate the AST parsing
    example_code = """
    def greet(name):
        print(f"Hello, {name}!")
    
    greet("World")
    """
    
    show_tree(example_code)    