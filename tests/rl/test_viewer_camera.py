"""Keep the image-demo camera opt-in for other policy playback."""
from types import SimpleNamespace

import pytest
from myoacross.rl import viewer


@pytest.mark.parametrize('camera,expected', [(None, (90., -45., 5.)), ('front-side', (-45., -12., 3.7))])
def test_camera_preset(camera, expected, monkeypatch):
    cfg = SimpleNamespace(azimuth=90., elevation=-45., distance=5., lookat=(0.,0.,0.))
    closed = []
    env = SimpleNamespace(cfg=SimpleNamespace(viewer=cfg), close=lambda: closed.append(True))
    monkeypatch.setattr(viewer, '_initialize_inference', lambda **kw: (env, object(), None))
    observed = []
    class FakeViewer:
        def __init__(self, env, policy, **kw):
            observed.append((env.cfg.viewer.azimuth, env.cfg.viewer.elevation, env.cfg.viewer.distance))
        def run(self): pass
    monkeypatch.setattr('mjlab.viewer.NativeMujocoViewer', FakeViewer)
    viewer.view_policy('model.xml', 'motion.npz', 'final.pt', camera=camera)
    assert observed == [expected]
    assert closed == [True]
