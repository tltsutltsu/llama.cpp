import pytest
from utils import *

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_min_p_schedule_normalized_resolution():
    """Normalized schedule positions should be resolved to absolute in the response."""
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 100,
        "min_p_schedule": [[0.0, 0.0], [1.0, 0.3]],
        "min_p_schedule_normalized": True,
        "min_p_interpolation": "linear",
    })
    assert res.status_code == 200
    settings = res.body["generation_settings"]
    sched = settings["min_p_schedule"]
    assert len(sched) == 2
    assert abs(sched[0][0] - 0.0) < 0.01
    assert abs(sched[0][1] - 0.0) < 0.01
    assert abs(sched[1][0] - 99.0) < 0.01  # 1.0 * (100-1) = 99
    assert abs(sched[1][1] - 0.3) < 0.01
    assert settings["min_p_interpolation"] == "linear"
    # Never emit the normalized flag
    assert "min_p_schedule_normalized" not in settings


def test_min_p_schedule_normalized_error_no_n_predict():
    """Normalized schedule without n_predict should fail."""
    global server
    server.n_predict = -1
    server.extra_args = ["--predict", "-1"]
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "min_p_schedule": [[0.0, 0.0], [1.0, 0.3]],
        "min_p_schedule_normalized": True,
    })
    assert res.status_code == 400


def test_min_p_schedule_invalid_interp():
    """Invalid interp string should fail."""
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 10,
        "min_p_schedule": [[0.0, 0.0], [5.0, 0.3]],
        "min_p_interpolation": "linera",
    })
    assert res.status_code == 400


def test_min_p_schedule_empty_clears():
    """Empty schedule [] clears the inherited server default schedule."""
    global server
    server.extra_args = ["--min-p-schedule", "0:0.05,100:0.3"]
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 10,
        "min_p_schedule": [],
    })
    assert res.status_code == 200
    settings = res.body["generation_settings"]
    assert "min_p_schedule" not in settings


def test_min_p_schedule_empty_with_normalized_flag_ok():
    """{min_p_schedule: [], min_p_schedule_normalized: true} must not 400 — validation runs
    against the effective non-empty schedule only; empty state means nothing to validate."""
    global server
    server.n_predict = -1
    server.extra_args = ["--predict", "-1"]
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "min_p_schedule": [],
        "min_p_schedule_normalized": True,
    })
    assert res.status_code == 200
    settings = res.body["generation_settings"]
    assert "min_p_schedule" not in settings


def test_min_p_schedule_inactive_when_not_in_samplers():
    """Schedule should be cleared when min_p is not in the sampler sequence."""
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 10,
        "min_p_schedule": [[0.0, 0.05], [100.0, 0.3]],
        "samplers": ["top_k", "top_p"],  # min_p intentionally omitted
    })
    assert res.status_code == 200
    settings = res.body["generation_settings"]
    assert "min_p_schedule" not in settings


def test_min_p_schedule_cleared_by_mirostat_same_request():
    """Schedule + mirostat in same request: sanitizer clears schedule. Validation must not 400
    on stale normalization state (state-reset invariant)."""
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 100,
        "min_p_schedule": [[0.0, 0.1], [1.0, 0.2]],
        "min_p_schedule_normalized": True,
        "mirostat": 1,
    })
    assert res.status_code == 200
    settings = res.body["generation_settings"]
    assert "min_p_schedule" not in settings
    assert "min_p_interpolation" not in settings


def test_min_p_schedule_mirostat_no_n_predict_no_400():
    """Schedule + mirostat without n_predict must NOT 400 — sanitizer runs before validator,
    clearing the schedule so the validator has nothing to check."""
    global server
    server.n_predict = -1
    server.extra_args = ["--predict", "-1"]
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "min_p_schedule": [[0.0, 0.1]],
        "mirostat": 1,
        "min_p_schedule_normalized": True,
    })
    assert res.status_code == 200


# ---------- D3 precedence matrix ----------

@pytest.fixture
def server_with_default_schedule():
    global server
    server.extra_args = ["--min-p-schedule", "0:0,100:0.3"]
    server.start()
    return server


def test_d3_empty_body_uses_inherited(server_with_default_schedule):
    """Request with neither min_p nor min_p_schedule → inherited schedule runs."""
    res = server_with_default_schedule.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 10,
    })
    assert res.status_code == 200
    settings = res.body["generation_settings"]
    assert "min_p_schedule" in settings
    assert len(settings["min_p_schedule"]) == 2


def test_d3_scalar_only_clears_schedule(server_with_default_schedule):
    """Request with min_p scalar only (no min_p_schedule key) → inherited schedule cleared."""
    res = server_with_default_schedule.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 10,
        "min_p": 0.1,
    })
    assert res.status_code == 200
    settings = res.body["generation_settings"]
    assert "min_p_schedule" not in settings
    assert abs(settings["min_p"] - 0.1) < 0.001


def test_d3_both_schedule_wins(server_with_default_schedule):
    """Request with both min_p and min_p_schedule → request schedule wins, scalar ignored."""
    res = server_with_default_schedule.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 10,
        "min_p": 0.1,
        "min_p_schedule": [[0.0, 0.2]],
    })
    assert res.status_code == 200
    settings = res.body["generation_settings"]
    assert "min_p_schedule" in settings
    assert len(settings["min_p_schedule"]) == 1
    assert abs(settings["min_p_schedule"][0][1] - 0.2) < 0.01


def test_d3_explicit_empty_clears(server_with_default_schedule):
    """Explicit empty min_p_schedule: [] clears inherited schedule."""
    res = server_with_default_schedule.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 10,
        "min_p_schedule": [],
    })
    assert res.status_code == 200
    settings = res.body["generation_settings"]
    assert "min_p_schedule" not in settings


# ---------- /props regression ----------

def test_props_echoes_resolved_absolute_positions():
    """GET /props with normalized startup schedule echoes resolved absolute positions.
    Depends on get_props() setting tparams.n_predict = params.n_predict."""
    global server
    server.extra_args = ["--min-p-schedule-normalized", "0:0,1:0.3", "--predict", "100"]
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    params = res.body["default_generation_settings"]["params"]
    assert "min_p_schedule" in params
    sched = params["min_p_schedule"]
    assert len(sched) == 2
    assert abs(sched[0][0] - 0.0) < 0.01
    assert abs(sched[1][0] - 99.0) < 0.01  # resolved absolute
    assert abs(sched[1][1] - 0.3) < 0.01
    assert params["min_p_interpolation"]
    assert "min_p_schedule_normalized" not in params


def test_props_echoes_temp_schedule_regression():
    """Regression: /props still echoes resolved temp-schedule positions (existing feature)."""
    global server
    server.extra_args = ["--temp-schedule-normalized", "0:1.0,1:0.5", "--predict", "100"]
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    params = res.body["default_generation_settings"]["params"]
    # Temp schedule is pre-scaled at parse time, not via the display helper,
    # but the end result on /props should still be resolved absolute positions.
    assert "temperature_schedule" in params
    sched = params["temperature_schedule"]
    assert abs(sched[1][0] - 99.0) < 0.01


# ---------- Triple-inheritance ----------

@pytest.fixture
def server_with_normalized_default():
    global server
    server.extra_args = ["--min-p-schedule-normalized", "0:0,1:0.3", "--predict", "100"]
    server.start()
    return server


def test_inherit_request_schedule_without_flag_is_absolute(server_with_normalized_default):
    """Request schedule replacement: flag does NOT inherit across schedule replacement."""
    res = server_with_normalized_default.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 100,
        "min_p_schedule": [[0.0, 0.1], [50.0, 0.2]],
    })
    assert res.status_code == 200
    sched = res.body["generation_settings"]["min_p_schedule"]
    assert len(sched) == 2
    assert abs(sched[0][0] - 0.0) < 0.01
    assert abs(sched[1][0] - 50.0) < 0.01  # absolute — flag not inherited


def test_inherit_request_schedule_with_flag_normalized(server_with_normalized_default):
    """Request schedule + flag = true → normalized, resolved."""
    res = server_with_normalized_default.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 100,
        "min_p_schedule": [[0.0, 0.1], [1.0, 0.2]],
        "min_p_schedule_normalized": True,
    })
    assert res.status_code == 200
    sched = res.body["generation_settings"]["min_p_schedule"]
    assert abs(sched[1][0] - 99.0) < 0.01


def test_inherit_lone_flag_true_ignored(server_with_normalized_default):
    """Lone flag (no schedule key) → flag ignored, inherited default pair runs."""
    res = server_with_normalized_default.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 100,
        "min_p_schedule_normalized": True,
    })
    assert res.status_code == 200
    sched = res.body["generation_settings"]["min_p_schedule"]
    assert len(sched) == 2
    assert abs(sched[1][0] - 99.0) < 0.01


def test_inherit_lone_flag_false_ignored(server_with_normalized_default):
    """Lone flag=false also ignored — inherited normalized schedule still runs."""
    res = server_with_normalized_default.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 100,
        "min_p_schedule_normalized": False,
    })
    assert res.status_code == 200
    sched = res.body["generation_settings"]["min_p_schedule"]
    assert len(sched) == 2
    assert abs(sched[1][0] - 99.0) < 0.01


def test_inherit_resolves_against_request_n_predict(server_with_normalized_default):
    """Request n_predict=50 with inherited normalized schedule → resolves against 50, not 100."""
    res = server_with_normalized_default.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 50,
    })
    assert res.status_code == 200
    sched = res.body["generation_settings"]["min_p_schedule"]
    assert abs(sched[1][0] - 49.0) < 0.01  # resolved against request's n_predict


def test_inherit_n_predict_zero_fails(server_with_normalized_default):
    """Request n_predict=0 with inherited normalized schedule → 400 (normalized needs >0)."""
    res = server_with_normalized_default.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 0,
    })
    assert res.status_code == 400
