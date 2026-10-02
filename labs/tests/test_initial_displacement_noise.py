"""Scientific coverage, paired RNG identity, and guarded reference reuse."""
import unittest
from types import SimpleNamespace
from experiments.replay_initial_displacement_noise import noise_draws, check_noise_record, BpttCache
from experiments.calibrate_initial_relative_output import matched_beta

class CloneValue:
    def clone(self):
        return CloneValue()

class ReplayContractTests(unittest.TestCase):
    def test_full_grid_counts_one_clean_and_disjoint_noise(self):
        cfg=dict(noise_sigmas=[0,1e-4,5e-4,1e-3,2e-3,5e-3],read_noise_seeds=[91000000,92000000,93000000])
        draws=noise_draws(cfg)
        self.assertEqual(len(draws),16)
        self.assertEqual(draws[0],(0,None,-1))
        self.assertEqual(sum((d+1)*15*36*len(draws) for d in (1,2,3)),77760)
        self.assertEqual(len({seed+1000*b+100*p+l for seed in cfg['read_noise_seeds'] for b in range(36) for p in (1,2) for l in range(4)}),864)

    def test_rng_pairing_allows_scaling_and_architecture_shape(self):
        seen={}
        record=dict(phase='positive',state_layer_index=1,seed=91000201,element_count=20,standard_normal_sha256='a')
        check_noise_record(seen,record,architecture='conv1',sigma=.0001,seed=91000000,batch_index=0)
        check_noise_record(seen,dict(record,standard_normal_sha256='b'),architecture='conv1',sigma=.001,seed=91000000,batch_index=0)
        check_noise_record(seen,dict(record,element_count=80),architecture='conv2',sigma=.0001,seed=91000000,batch_index=0)
        with self.assertRaises(AssertionError):
            check_noise_record(seen,dict(record,standard_normal_sha256='bad'),architecture='conv1',sigma=.0001,seed=91000000,batch_index=0)

    def test_reference_cache_rejects_changed_state_or_parameters(self):
        calls=[]
        def compute(runtime,**kw):
            calls.append(kw)
            return {'weight':CloneValue()},{'original':True}
        base=SimpleNamespace(_parameter_state_sha256=lambda p:p)
        cache=BpttCache(compute,base)
        cache.context=('conv1','ours',0,'batch_hash')
        runtime=dict(gradient_iterations=4,minimizer_gradient=SimpleNamespace(num_iterations=4),parameters='p',weight_names=['weight'])
        first=cache(runtime,post_t_states=[],post_t_hash='s',expected_k=4)
        second=cache(runtime,post_t_states=[],post_t_hash='s',expected_k=4)
        self.assertEqual(len(calls),1)
        self.assertIsNot(first[0]['weight'],second[0]['weight'])
        with self.assertRaises(AssertionError):
            cache(runtime,post_t_states=[],post_t_hash='changed',expected_k=4)
        with self.assertRaises(AssertionError):
            cache(dict(runtime,parameters='changed'),post_t_states=[],post_t_hash='s',expected_k=4)

    def test_matching_stops_after_one_correction(self):
        seen=[]
        def bad(beta):
            seen.append(beta)
            return dict(relative_output_rms=.1)
        with self.assertRaises(RuntimeError):
            matched_beta(bad,1,.2,.01,1)
        self.assertEqual(len(seen),2)

if __name__=='__main__':
    unittest.main()
