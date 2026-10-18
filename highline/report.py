"""Petit collecteur de lignes de rapport (affiche et conserve)."""
from pathlib import Path


class Report:
    def __init__(self, quiet=False):
        self.lines = []
        self.quiet = quiet

    def say(self, *a):
        s = " ".join(str(x) for x in a)
        if not self.quiet:
            print(s, flush=True)
        self.lines.append(s)

    def save(self, path):
        Path(path).write_text("\n".join(self.lines), encoding="utf-8")
