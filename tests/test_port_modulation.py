import copy
import unittest
import numpy as np
from neuron.neuron import NeuronParameters,RetrogradeSignalEvent
from neuron.extensions.experimental.eligibility_trace import EligibilityTraceNeuron
from neuron.extensions.experimental.port_modulation import PortModulationNeuron
from neuron.extensions.experimental.bounded_plasticity import magnitude_step


def cell(cls=PortModulationNeuron, sensitivity=None):
    np.random.seed(11)
    p=NeuronParameters(num_inputs=3,c=3,lambda_param=1.,r_base=.65,b_base=.9,
                       eta_post=1e-5,eta_retro=1e-7,w_r=np.zeros(2),w_b=np.zeros(2),w_tref=np.zeros(2))
    meta=dict(bounded_plasticity=True,eligibility_ports=[1],plasticity_rate_boost=499.)
    if sensitivity is not None:meta['native_port_modulation']=[dict(port=2,sensitivity=sensitivity)]
    n=cls(1,p,log_level='CRITICAL',metadata=meta)
    for sid,q in enumerate((2.,.1,-.08)):
        n.add_synapse(sid,1);n.postsynaptic_points[sid].u_i.info=q
        n.postsynaptic_points[sid].u_i.plast=0.;n.postsynaptic_points[sid].u_i.adapt[:]=0.
        n.register_source(sid,2,900)
    n.M_vector[0]=.1
    return n


class PortModulationTests(unittest.TestCase):
    def test_default_and_unit_sensitivity_are_exact(self):
        for sensitivity in (None,1.):
            a,b=cell(sensitivity=sensitivity),cell(EligibilityTraceNeuron)
            for t in range(40):
                for n in (a,b):n.input_buffer[:,0]=[1. if t%8==0 else 0.,.2,.5]
                ea,eb=a.tick({},t),b.tick({},t)
                self.assertEqual((a.S,a.O,a.t_ref,a.eligibility_post),(b.S,b.O,b.t_ref,b.eligibility_post))
                np.testing.assert_array_equal(a.eligibility_pre,b.eligibility_pre)
                np.testing.assert_array_equal(a.M_vector,b.M_vector)
                self.assertEqual(a.propagation_queue,b.propagation_queue)
                self.assertEqual([p.u_i.info for p in a.postsynaptic_points.values()],[p.u_i.info for p in b.postsynaptic_points.values()])
                self.assertEqual(len(ea),len(eb))
                for x,y in zip(ea,eb):
                    if isinstance(x,RetrogradeSignalEvent):np.testing.assert_array_equal(x.error_vector,y.error_vector)
                    else:self.assertEqual(x,y)

    def test_zero_sensitivity_still_adapts_and_preserves_forward_event(self):
        a,b=cell(sensitivity=0.),cell(EligibilityTraceNeuron)
        for n in (a,b):n.t_last_fire=-1;n.input_buffer[2,0]=1.
        error=float(np.linalg.norm(np.array([np.float32(1.)-(-.08),np.float32(0.),np.float32(0.),np.float32(0.)])))
        expected=-magnitude_step(.08,error,1,1e-5)
        ea,eb=a.tick({},0),b.tick({},0)
        self.assertEqual(a.postsynaptic_points[2].u_i.info,expected)
        self.assertNotEqual(expected,-.08)
        self.assertGreater(abs(b.postsynaptic_points[2].u_i.info),abs(expected))
        self.assertEqual(a.propagation_queue,b.propagation_queue)
        for x,y in zip(ea,eb):np.testing.assert_array_equal(x.error_vector,y.error_vector)
        self.assertEqual(a.params.eta_post,1e-5)

    def test_clone_and_fractional_rate(self):
        a=cell(sensitivity=.25);b=copy.deepcopy(a)
        for t in range(16):
            for n in (a,b):n.input_buffer[:,0]=[1.,.2,.5];n.tick({},t)
        self.assertEqual(a.native_port_modulation,{2:.25})
        self.assertEqual([p.u_i.info for p in a.postsynaptic_points.values()],[p.u_i.info for p in b.postsynaptic_points.values()])
        self.assertLess(a.postsynaptic_points[2].u_i.info,0.)

    def test_rejects_invalid_declarations(self):
        for entries in ([dict(port=1,sensitivity=0.)],[dict(port=3,sensitivity=0.)],
                        [dict(port=2,sensitivity=-1.)],[dict(port=2,sensitivity=float('nan'))],
                        [dict(port=2,sensitivity=0.),dict(port=2,sensitivity=1.)]):
            with self.assertRaises(ValueError):
                PortModulationNeuron(1,NeuronParameters(num_inputs=3),log_level='CRITICAL',
                    metadata=dict(bounded_plasticity=True,eligibility_ports=[1],native_port_modulation=entries))


if __name__=='__main__':unittest.main()
