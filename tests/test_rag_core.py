from pathlib import Path

from rag_project.core import RepositoryIndex


def test_python_chunking_and_metadata(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    file_path = repo / "sample.py"
    file_path.write_text(
        """
class Service:
    def run(self):
        return \"ok\"

def helper(value):
    return value * 2
""".strip()
    )

    index = RepositoryIndex()
    changes = index.ingest_repository(str(repo))

    assert str(file_path) in changes["added"]
    chunk_types = {chunk.metadata["chunk_type"] for chunk in index.chunks}
    assert "class" in chunk_types
    assert "function" in chunk_types
    for chunk in index.chunks:
        assert chunk.metadata["file_path"].endswith("sample.py")
        assert "last_modified" in chunk.metadata


def test_hybrid_query_finds_exact_function_name(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "alpha.py").write_text("def highly_specific_function_name():\n    return 1\n")
    (repo / "beta.py").write_text("def generic():\n    return 2\n")

    index = RepositoryIndex()
    index.ingest_repository(str(repo))

    result = index.query("Where is highly_specific_function_name defined?", top_k=1)[0]
    assert "highly_specific_function_name" in result["text"]


def test_stale_detection_updates_changed_files(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "stale.py"
    source.write_text("def old():\n    return 1\n")

    index = RepositoryIndex()
    index.ingest_repository(str(repo))

    source.write_text("def fresh():\n    return 2\n")
    changes = index.ingest_repository(str(repo))

    assert str(source) in changes["updated"]
    assert any("fresh" in chunk.text for chunk in index.chunks)
