import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)
TEXT_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".csv",
    ".inc",
    ".data",
    ".dev",
    ".xml",
    ".log",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
}


def load_pdfs(pdf_dir: Path) -> tuple[list[tuple[str, str]], list[str]]:
    pages: list[tuple[str, str]] = []
    skipped_files: list[str] = []
    if not pdf_dir.exists():
        return pages, skipped_files

    for path in sorted(pdf_dir.glob("*.pdf")):
        try:
            reader = PdfReader(str(path))
        except (PdfReadError, ValueError) as exc:
            logger.warning("Skipping unreadable PDF '%s': %s", path.name, exc)
            skipped_files.append(path.name)
            continue

        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                source = f"{path.name}#page-{idx}"
                pages.append((source, text))
    return pages, skipped_files


def load_tutorial_files(tutorials_dir: Path) -> tuple[list[tuple[str, str]], list[str]]:
    docs: list[tuple[str, str]] = []
    skipped_files: list[str] = []
    if not tutorials_dir.exists():
        return docs, skipped_files

    for path in sorted(tutorials_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("~$"):
            continue

        suffix = path.suffix.lower()
        rel = str(path.relative_to(tutorials_dir))

        if suffix == ".pdf":
            try:
                reader = PdfReader(str(path))
            except (PdfReadError, ValueError) as exc:
                logger.warning("Skipping unreadable tutorial PDF '%s': %s", path.name, exc)
                skipped_files.append(rel)
                continue

            for idx, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    docs.append((f"tutorials/{rel}#page-{idx}", text))
            continue

        if suffix not in TEXT_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="latin-1")
            except Exception as exc:
                logger.warning("Skipping unreadable tutorial file '%s': %s", path.name, exc)
                skipped_files.append(rel)
                continue
        except Exception as exc:
            logger.warning("Skipping unreadable tutorial file '%s': %s", path.name, exc)
            skipped_files.append(rel)
            continue

        cleaned = text.strip()
        if cleaned:
            docs.append((f"tutorials/{rel}", cleaned))
    return docs, skipped_files


def load_documents(
    pdf_dir: Path,
    tutorials_dir: Path,
) -> tuple[list[tuple[str, str]], list[str]]:
    pdf_docs, pdf_skipped = load_pdfs(pdf_dir)
    tutorial_docs, tutorial_skipped = load_tutorial_files(tutorials_dir)
    return pdf_docs + tutorial_docs, pdf_skipped + tutorial_skipped


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = max(chunk_size - chunk_overlap, 1)
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks
