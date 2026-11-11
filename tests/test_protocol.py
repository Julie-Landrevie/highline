"""Le protocole publie (docs/PROTOCOLE.json) doit rester identique aux constantes du code."""
import json, pathlib
from highline import config, checks

P = json.loads((pathlib.Path(__file__).parent.parent / "docs" / "PROTOCOLE.json").read_text(encoding="utf-8"))["constants"]


def test_constantes_du_protocole_egales_au_code():
    assert P["step_frames"] == config.STEP
    assert P["horizon_frames"] == config.HORIZON
    assert P["hsr_ms"] == config.HSR_MS
    assert P["min_cell"] == config.MIN_CELL
    assert P["min_events"] == config.MIN_EVENTS
    assert P["r90_factor"] == config.R90
    assert P["gap_s"] == list(config.SIG_T)
    assert P["sigma_m"] == list(config.SIG_V)
    assert P["goal_reference"] == config.GOAL_REFERENCE
    assert P["fp_max"] == checks.FP_MAX
