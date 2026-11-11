"""La verification --check detecte un ecart et accepte une copie identique."""
from highline.pipeline import check_against


def _w(d, name, txt):
    d.mkdir(exist_ok=True)
    (d / name).write_text(txt, encoding="utf-8")


def test_check_accepte_les_memes_nombres(tmp_path):
    for n in ("table_1.md", "table_1_en.md"):
        _w(tmp_path / "a", n, "| bilan | -0.90 [-1.81 ; -0.02] |")
        _w(tmp_path / "b", n, "| bilan | -0.91 [-1.80 ; -0.02] |")
    ok, bad = check_against(tmp_path / "a", tmp_path / "b")
    assert ok and not bad


def test_check_detecte_un_ecart_et_un_fichier_absent(tmp_path):
    _w(tmp_path / "a", "table_1.md", "| bilan | -0.90 |")
    _w(tmp_path / "b", "table_1.md", "| bilan | +1.43 |")
    ok, bad = check_against(tmp_path / "a", tmp_path / "b")
    assert not ok and any("recalcule" in b for b in bad) and any("absent" in b for b in bad)
