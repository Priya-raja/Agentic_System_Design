from educosys_claude.context.indexers.code_parser import (
    get_source_files,
    parse_file,
)
from pathlib import Path

# Sibling project: up one level from your repo, then into sample_project
sample_project = Path(__file__).resolve().parent.parent / "sample_project"

# 1. Find all supported files in the project
files = get_source_files(str(sample_project))
print(f"Found {len(files)} files\n")

# 2. Parse each file into chunks
for filepath in files:
    chunks = parse_file(filepath)
    for chunk in chunks:
        print("=" * 60)
        print(f"File      : {chunk.source}")
        print(f"Name      : {chunk.name}")
        print(f"Type      : {chunk.type}")
        print(f"Lines     : {chunk.start_line} - {chunk.end_line}")
        print("Content:")
        print(chunk.content)