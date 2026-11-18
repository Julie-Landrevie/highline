"""Audit T3 : aucune donnee, aucun modele ni aucune constante externes dans le code versionne.

Regles (ecrites avant de lancer) :
- aucun fichier de donnees ou de modele (.pkl .npy .npz .parquet .h5 .pt .pth .onnx .joblib) hors des dossiers de donnees ignores ;
- toute URL d'un fichier .py pointe vers les donnees ouvertes SkillCorner (GitHub SkillCorner/opendata, Hugging Face SkillCorner/opendata-bodypose) ;
- aucune reference a un fournisseur ou a un fichier de reference externe, aucun chargement de poids pre-entraines.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IGNORED_DIRS = {"data", "cache", "out", "hp_data", ".git", ".github", "highline.egg-info", "__pycache__", ".venv", "venv", "build", "dist", ".pytest_cache"}
DATA_EXT = {".pkl", ".npy", ".npz", ".parquet", ".h5", ".hdf5", ".pt", ".pth", ".onnx", ".joblib", ".feather"}
ALLOWED_URL_PREFIXES = (
    "https://raw.githubusercontent.com/SkillCorner/opendata",
    "https://media.githubusercontent.com/media/SkillCorner/opendata",
    "https://huggingface.co/datasets/SkillCorner/opendata-bodypose",
)
# Seule exception documentee : polices libres (OFL) pour la mise en page des visuels, telechargees uniquement si HIGHLINE_DOWNLOAD_FONTS=1
OPTIONAL_FONT_URL = ("highline/render3d.py", "https://raw.githubusercontent.com/google/fonts/main/ofl/")
FORBIDDEN = ("karun", "statsbomb", "wyscout", "fbref", "transfermarkt", "understat", "opta", "from_pretrained", "torch.load", "pickle.load")


def _files(root):
    for p in Path(root).rglob("*"):
        if p.is_file() and not (set(p.relative_to(root).parts[:-1]) & IGNORED_DIRS):
            yield p


def audit(root=ROOT, self_path=Path(__file__).resolve()):
    """Retourne la liste des problemes trouves (vide = conforme)."""
    bad = []
    for p in _files(root):
        rel = p.relative_to(root)
        if p.suffix.lower() in DATA_EXT:
            bad.append(f"{rel} : fichier de donnees ou de modele versionne")
        if p.suffix != ".py" or p.resolve() == self_path:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for url in re.findall(r"https?://[^\s'\"<>)\]]+", text):
            if url == OPTIONAL_FONT_URL[1] and rel.as_posix() == OPTIONAL_FONT_URL[0] and 'HIGHLINE_DOWNLOAD_FONTS' in text:
                continue
            if not url.startswith(ALLOWED_URL_PREFIXES):
                bad.append(f"{rel} : URL hors donnees ouvertes SkillCorner : {url}")
        low = text.lower()
        bad += [f"{rel} : mot interdit '{w}'" for w in FORBIDDEN if re.search(r"\b" + re.escape(w), low)]
    return bad


def test_aucune_donnee_ou_constante_externe():
    assert audit() == []


def test_l_audit_detecte_un_manquement(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "grille_xt.npy").write_bytes(b"0")
    (tmp_path / "a.py").write_text("import requests\nrequests.get('https://example.com/x')\n# grille de Karun Singh\n", encoding="utf-8")
    bad = audit(tmp_path, self_path=tmp_path / "tests" / "x.py")
    assert len(bad) == 3
