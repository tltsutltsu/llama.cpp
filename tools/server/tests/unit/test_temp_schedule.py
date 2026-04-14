import pytest
from utils import *

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_temp_schedule_normalized_resolution():
    """Normalized schedule positions should be resolved to absolute in the response."""
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 100,
        "temperature_schedule": [[0.0, 1.0], [1.0, 0.5]],
        "temperature_schedule_normalized": True,
        "temperature_interpolation": "linear",
    })
    assert res.status_code == 200
    settings = res.body["generation_settings"]
    sched = settings["temperature_schedule"]
    assert len(sched) == 2
    assert abs(sched[0][0] - 0.0) < 0.01
    assert abs(sched[0][1] - 1.0) < 0.01
    assert abs(sched[1][0] - 99.0) < 0.01  # 1.0 * (100-1) = 99
    assert abs(sched[1][1] - 0.5) < 0.01
    assert settings["temperature_interpolation"] == "linear"


def test_temp_schedule_normalized_error_no_n_predict():
    """Normalized schedule without n_predict should fail."""
    global server
    server.n_predict = -1
    server.extra_args = ["--predict", "-1"]
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "temperature_schedule": [[0.0, 1.0], [1.0, 0.5]],
        "temperature_schedule_normalized": True,
        # n_predict inherits server default of -1
    })
    assert res.status_code == 400


def test_temp_schedule_inherited_not_reinterpreted():
    """Server started with absolute schedule; request sends only normalized flag.
    The inherited absolute schedule should NOT be re-multiplied."""
    global server
    server.extra_args = ["--temp-schedule", "0:1.0,100:0.5"]
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 10,
        "temperature_schedule_normalized": True,
        # No temperature_schedule in request — inherits from server
    })
    assert res.status_code == 200
    settings = res.body["generation_settings"]
    sched = settings["temperature_schedule"]
    assert len(sched) == 2
    # Should be original absolute values, not re-multiplied
    assert abs(sched[0][0] - 0.0) < 0.01
    assert abs(sched[0][1] - 1.0) < 0.01
    assert abs(sched[1][0] - 100.0) < 0.01
    assert abs(sched[1][1] - 0.5) < 0.01


def test_temp_schedule_inactive_when_not_in_samplers():
    """Schedule should be cleared when temperature is not in the sampler sequence."""
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 10,
        "temperature_schedule": [[0.0, 1.0], [100.0, 0.5]],
        "samplers": ["top_k", "top_p"],  # temperature intentionally omitted
    })
    assert res.status_code == 200
    settings = res.body["generation_settings"]
    assert "temperature_schedule" not in settings
