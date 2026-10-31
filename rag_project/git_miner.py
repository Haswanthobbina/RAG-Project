"""Git history, metadata miner, and incremental staleness detector."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import git
except ImportError:
    git = None  # type: ignore[assignment]


class GitMiner:
    """Extracts commit histories, authors, PR tags, and tracks repository diffs."""

    def __init__(self, repo_path: str | Path) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.repo: Optional[Any] = None
        self.is_git_repo = False
        self._last_indexed_commit: Optional[str] = None

        if git is not None:
            try:
                self.repo = git.Repo(self.repo_path, search_parent_directories=True)
                self.is_git_repo = True
                if self.repo.head.is_valid():
                    self._last_indexed_commit = self.repo.head.commit.hexsha
            except Exception:
                self.repo = None
                self.is_git_repo = False

    def get_file_metadata(self, file_path: str | Path) -> Dict[str, str]:
        """Fetches last modified commit, author, message, and timestamp."""
        path_obj = Path(file_path).resolve()
        default_meta = {
            "last_commit_hash": "untracked",
            "author": "local_developer",
            "author_email": "dev@local",
            "commit_message": "working tree update",
            "pr_reference": "N/A",
            "last_modified": (
                path_obj.stat().st_mtime_ns if path_obj.exists() else 0
            ),
        }

        if not self.is_git_repo or self.repo is None:
            return {k: str(v) for k, v in default_meta.items()}

        try:
            # Get path relative to the git repository root
            git_root = Path(self.repo.working_tree_dir).resolve()
            rel_path = path_obj.relative_to(git_root).as_posix()

            commits = list(self.repo.iter_commits(paths=rel_path, max_count=1))
            if commits:
                c = commits[0]
                message_first_line = c.message.strip().splitlines()[0] if c.message else ""
                
                # Extract PR reference if available (e.g. #123)
                pr_ref = "N/A"
                if "(#" in message_first_line:
                    start = message_first_line.find("(#") + 1
                    end = message_first_line.find(")", start)
                    if end != -1:
                        pr_ref = message_first_line[start:end]

                return {
                    "last_commit_hash": c.hexsha[:8],
                    "author": c.author.name or "unknown",
                    "author_email": c.author.email or "",
                    "commit_message": message_first_line,
                    "pr_reference": pr_ref,
                    "last_modified": c.committed_datetime.isoformat(),
                }
        except Exception:
            pass

        return {k: str(v) for k, v in default_meta.items()}

    def detect_changed_files(self, since_commit: Optional[str] = None) -> List[str]:
        """Identifies files modified or added since a target commit hash."""
        if not self.is_git_repo or self.repo is None:
            return []

        target_commit = since_commit or self._last_indexed_commit
        if not target_commit:
            return []

        try:
            current_head = self.repo.head.commit
            if current_head.hexsha == target_commit:
                return []

            diff_index = current_head.diff(target_commit)
            changed = set()
            for d in diff_index:
                if d.a_path:
                    changed.add(d.a_path)
                if d.b_path:
                    changed.add(d.b_path)
            return sorted(list(changed))
        except Exception:
            return []

    def update_last_commit(self) -> None:
        """Saves current HEAD commit as the last indexed checkpoint."""
        if self.is_git_repo and self.repo is not None:
            try:
                self._last_indexed_commit = self.repo.head.commit.hexsha
            except Exception:
                pass
