import unittest
from types import SimpleNamespace
from vid2player.utils.racket_dynamics import resolve_racket_dynamics, configure_racket_body


def vector():
    return SimpleNamespace(x=0., y=0., z=0.)


class TestRacketDynamics(unittest.TestCase):
    def test_reference_has_physical_inertia_and_measured_recoil(self):
        s=resolve_racket_dynamics({'model':'published_reference_v1'})
        # Cross's normal recoil comparison: material e=.9, apparent e approximately .4.
        effective=1/(1/s.mass+s.com_x**2/s.transverse_inertia)
        apparent=(.9-.057/effective)/(1+.057/effective)
        self.assertAlmostEqual(apparent,.4,delta=.015)
        self.assertFalse(resolve_racket_dynamics().enabled)

    def test_body_properties_are_explicit(self):
        body=SimpleNamespace(mass=1.,com=vector(),inertia=SimpleNamespace(x=vector(),y=vector(),z=vector()))
        gym=SimpleNamespace(get_actor_rigid_body_properties=lambda env,actor:[body])
        calls=[]
        gym.set_actor_rigid_body_properties=lambda *args:calls.append(args)
        configure_racket_body(gym,'env','actor',0,{'model':'published_reference_v1'})
        self.assertEqual(body.mass,.270)
        self.assertEqual(body.com.x,.180)
        self.assertAlmostEqual(body.inertia.y.y-body.inertia.y.z,.0123)
        self.assertAlmostEqual(body.inertia.y.y+body.inertia.y.z,.01349)
        self.assertFalse(calls[0][-1])

    def test_invalid_inertial_properties_fail(self):
        for cfg in [{'model':'unknown'},{'mass':-1},{'com_x':float('nan')},
                    {'normal_inertia':1.},{'long_axis_inertia':0.},{'extra':True}]:
            with self.subTest(cfg=cfg),self.assertRaises(ValueError):resolve_racket_dynamics(cfg)
