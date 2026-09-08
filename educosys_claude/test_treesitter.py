from tree_sitter import Language, Parser
from tree_sitter_languages import get_language, get_parser



source_code = """
class CherryVirtualMachine:
    def __init__(self, bytecode):
        self.bytecode = bytecode
        self.stack = []
        self.ip = 0

    def execute(self):
        while self.ip < len(self.bytecode):
            op, arg = self.bytecode[self.ip]
            self.ip += 1

            if op == "LOAD_STREAM":
                self.stack.append(self.initialize_stream(arg))
            elif op == "TRANSFORM_DATA":
                transform_func = arg
                data = self.stack.pop()
                self.stack.append(transform_func(data))
            elif op == "EMIT_SIGNAL":
                self.flush_to_hardware(self.stack.pop())
"""


parser = get_parser("python")

tree = parser.parse(source_code.encode())

def print_tree(node, indent=0):
    print(" " * indent + f"{node.type} {node.start_point} -> {node.end_point}")

    for child in node.children:
        print_tree(child, indent + 2)


print_tree(tree.root_node)