import unittest
import numpy as np
from neuron.neuron import NeuronParameters,RetrogradeSignalEvent
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron
from neuron.extensions.experimental.graded_eligibility import GradedEligibilityNeuron


def cell(cls, gain=0, ports=(1,)):
    np.random.seed(11)
    p=NeuronParameters(num_inputs=3,c=3,lambda_param=4,r_base=.6,b_base=.85,
                       eta_post=1e-6,eta_retro=1e-7,w_r=np.zeros(2),w_b=np.zeros(2),w_tref=np.zeros(2))
    n=cls(1,p,log_level='CRITICAL',metadata=dict(bounded_plasticity=True,
          eligibility_ports=list(ports),graded_gain=gain,plasticity_rate_boost=4.))
    for sid,q in enumerate((2.,.1,0.)):
        n.add_synapse(sid,1);point=n.postsynaptic_points[sid]
        point.u_i.info=q;point.u_i.plast=0;point.u_i.adapt[:]=0
        n.register_source(sid,2,900)
    n.add_axon_terminal(900,1);n.presynaptic_points[900].u_o.info=1.;n.presynaptic_points[900].u_o.mod[:]=0
    return n


class GradedEligibilityTests(unittest.TestCase):
    def test_default_matches_eligibility_dynamics_and_events(self):
        a,b=cell(GradedEligibilityNeuron),cell(EligibilityTraceNeuron)
        for t in range(64):
            for n in (a,b):n.input_buffer[:,0]=(1. if t%4==0 else 0.,.5 if t%7==0 else 0.,0.)
            ea,eb=a.tick({},t),b.tick({},t)
            self.assertEqual((a.S,a.O,a.t_ref,a.t_last_fire),(b.S,b.O,b.t_ref,b.t_last_fire))
            np.testing.assert_array_equal(a.eligibility_pre,b.eligibility_pre)
            np.testing.assert_array_equal([p.u_i.info for p in a.postsynaptic_points.values()],[p.u_i.info for p in b.postsynaptic_points.values()])
            self.assertEqual(len(ea),len(eb))
            for x,y in zip(ea,eb):
                if isinstance(x,RetrogradeSignalEvent):np.testing.assert_array_equal(x.error_vector,y.error_vector)
                else:self.assertEqual(x,y)

    def test_graded_release_keeps_bounded_adaptation_and_no_fake_spike(self):
        n=cell(GradedEligibilityNeuron,.25,())
        for t in range(32):
            n.input_buffer[0,0]=1. if t%4==0 else 0.
            n.tick({},t)
            self.assertEqual(n.O,.25*max(0.,float(n.S)))
            self.assertEqual(n.t_last_fire,-np.inf)
            self.assertGreater(n.params.eta_post,0)
            self.assertGreater(n.params.eta_retro,0)
        self.assertLess(n.postsynaptic_points[0].u_i.info,2.)
        self.assertGreater(n.postsynaptic_points[0].u_i.info,0.)

    def test_rejects_undefined_graded_postsynaptic_eligibility(self):
        with self.assertRaises(ValueError):cell(GradedEligibilityNeuron,.25,(1,))


if __name__=='__main__':unittest.main()
