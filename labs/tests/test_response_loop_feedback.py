import numpy as np

from labs.response_loop_feedback import calibrate_response_loops


def test_reciprocity_defect_has_correct_sign_and_potential_derivative():
    rng = np.random.default_rng(8)
    raw = rng.normal(size=(6, 6))
    h = -np.eye(6)-raw@raw.T
    raw = rng.normal(size=(6, 6))
    g = (raw-raw.T)/2
    raw = rng.normal(size=(6, 6))
    b = (raw-raw.T)/2
    r = np.linalg.inv(h+g)
    defect = r.T-r
    assert np.sum(g*defect) > 0
    assert np.allclose(np.sum(g*defect), 2*np.trace(g@r))
    epsilon = 1e-5
    numeric = (np.linalg.slogdet(-(h+g+epsilon*b))[1]
               -np.linalg.slogdet(-(h+g-epsilon*b))[1])/(2*epsilon)
    assert np.allclose(np.sum(b*defect), 2*numeric, rtol=1e-7)


def test_scalar_measurement_calibration_and_callback_isolation(monkeypatch):
    # Analytic crossed responses of J=-I+g[[0,1],[-1,0]]; the learner sees
    # only their difference. No solve or Jacobian is used by this experiment.
    true_gain = 0.7

    def measure(coefficients):
        residual = true_gain+coefficients[0]
        crossed_defect = 4*residual/(1+residual**2)
        coefficients[:] = 1000  # caller data must not alias the learner
        return np.array([crossed_defect]), dict(equilibrations=4, scalar_reads=4)

    def forbidden(*args, **kwargs):
        raise AssertionError("calibration must not solve a matrix system")

    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    coefficients, counts, history = calibrate_response_loops(
        measure, 1, steps=40, learning_rate=0.1,
        callback=lambda row, c: c.fill(1000))
    assert abs(coefficients[0]+true_gain) < 1e-7
    assert counts == dict(equilibrations=160, scalar_reads=160)
    assert len(history) == 40
