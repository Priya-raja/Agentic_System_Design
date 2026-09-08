from dataclasses import dataclass
from pathlib import Path
from tree_sitter_languages import get_parser

from educosys_claude.observability.logger import get_logger


logger = get_logger(__name__)


# Maps file extension -> Tree-sitter language.
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sh": "bash",
}


# Files where AST parsing isn't useful for our first version.
TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
}

ALL_EXTENSIONS = set(EXTENSION_TO_LANGUAGE) | TEXT_EXTENSIONS


CHUNK_SIZE = 50
CHUNK_OVERLAP = 10


# AST nodes we consider meaningful units of code.
BLOCK_NODE_TYPES = {
    "function_definition",
    "function_declaration",
    "method_definition",
    "arrow_function",
    "class_definition",
    "class_declaration",
    "method_declaration",
    "constructor_declaration",
    "interface_declaration",
    "function_item",
    "func_declaration",
}


@dataclass
class ParsedChunk:
    name: str
    type: str
    content: str
    source: str
    start_line: int
    end_line: int


def parse_file(filepath: str) -> list[ParsedChunk]:
    """Parse one file into chunks."""
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext in TEXT_EXTENSIONS:
        lines = path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()

        return _sliding_window(lines, filepath)

    language_name = EXTENSION_TO_LANGUAGE.get(ext)

    if not language_name:
        raise ValueError(f"Unsupported file type: {ext}")

    source = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    return _parse_with_treesitter(
        source,
        filepath,
        language_name,
    )


def _parse_with_treesitter(
    source: str,
    filepath: str,
    language_name: str,
) -> list[ParsedChunk]:
    """Parse source with Tree-sitter and extract meaningful blocks."""

    logger.info(
        "Parsing %s file: %s",
        language_name,
        filepath,
    )

    parser = get_parser(language_name)
    tree = parser.parse(source.encode("utf-8"))

    chunks: list[ParsedChunk] = []

    _walk(
        tree.root_node,
        source,
        filepath,
        chunks,
    )

    # Some files contain no functions/classes.
    if not chunks:
        logger.warning(
            "No AST blocks found in %s, falling back to sliding window",
            filepath,
        )

        return _sliding_window(
            source.splitlines(),
            filepath,
        )

    logger.info(
        "Parsed %d chunks from %s",
        len(chunks),
        filepath,
    )

    return chunks


def _walk(
    node,
    source: str,
    filepath: str,
    chunks: list[ParsedChunk],
) -> None:
    """Walk the AST and extract meaningful top-level blocks."""

    if node.type in BLOCK_NODE_TYPES:
        name = _extract_name(node, source)

        content = source[
            node.start_byte:node.end_byte
        ]

        chunk_type = (
            "class"
            if "class" in node.type
            else "function"
        )

        chunks.append(
            ParsedChunk(
                name=name,
                type=chunk_type,
                content=content,
                source=str(Path(filepath).resolve()),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            )
        )

        logger.debug(
            "Found %s '%s' lines %d-%d",
            chunk_type,
            name,
            node.start_point[0] + 1,
            node.end_point[0] + 1,
        )

        # Don't separately index nested methods/functions
        # in this first version.
        return

    for child in node.children:
        _walk(
            child,
            source,
            filepath,
            chunks,
        )


def _extract_name(node, source: str) -> str:
    """Extract the identifier/name from an AST node."""

    for child in node.children:
        if child.type in {
            "identifier",
            "name",
            "property_identifier",
        }:
            return source[
                child.start_byte:child.end_byte
            ]

    return node.type


def _sliding_window(
    lines: list[str],
    filepath: str,
) -> list[ParsedChunk]:
    """Split text into overlapping line-based chunks."""

    if not lines:
        raise ValueError(f"Empty file: {filepath}")

    chunks: list[ParsedChunk] = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    for i, start in enumerate(
        range(0, len(lines), step)
    ):
        end = min(
            start + CHUNK_SIZE,
            len(lines),
        )

        text = "\n".join(
            lines[start:end]
        ).strip()

        if text:
            chunks.append(
                ParsedChunk(
                    name=f"chunk_{i}",
                    type="block",
                    content=text,
                    source=str(Path(filepath).resolve()),
                    start_line=start + 1,
                    end_line=end,
                )
            )

        if end == len(lines):
            break

    logger.info(
        "Parsed %d chunks from %s",
        len(chunks),
        filepath,
    )

    return chunks


def get_source_files(
    repo_path: str,
    skip_dirs: list[str] | None = None,
) -> list[str]:
    """Find all supported source files recursively."""

    skip = set(
        skip_dirs
        or [
            ".venv",
            "venv",
            "__pycache__",
            ".git",
            "node_modules",
            "dist",
            "build",
        ]
    )

    files = [
        str(path)
        for path in Path(repo_path).rglob("*")
        if path.suffix.lower() in ALL_EXTENSIONS
        and not any(
            part in skip
            for part in path.parts
        )
    ]

    logger.info(
        "Found %d source files in %s",
        len(files),
        repo_path,
    )

    return files