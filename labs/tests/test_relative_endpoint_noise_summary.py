import copy
import itertools
import unittest

from experiments.plot_relative_endpoint_noise import (
    LAYERS, LEVELS, MODELS, SCHEMES, SEEDS, TARGETS, summarize, scale_controls,
)


def complete_rows():
    rows = []
    acquisitions = [('clean', 0., -1)] + list(itertools.product(MODELS, LEVELS, range(3)))
    for scheme, target, layer, (model, level, draw), batch in itertools.product(
            SCHEMES, TARGETS, LAYERS, acquisitions, range(36)):
        cosine = (.9 if scheme == 'legacy' else .5) if model == 'absolute' else .8
        # The cohort mean is unchanged, while individual batches differ.
        cosine += (batch - 17.5) * .001
        rows.append(dict(scheme=scheme, relative_target=target, parameter_name=layer,
            noise_model=model, noise_level=level, noise_draw_index=draw,
            noise_seed=None if model == 'clean' else SEEDS[draw], batch_index=batch,
            cosine=cosine, noisy_clean_eqprop_cosine=cosine,
            clean_eqprop_bptt_cosine=1., noise_over_clean_norm=.1,
            relative_l2_difference_over_bptt=.2))
    return rows


class RelativeEndpointNoiseSummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = complete_rows()

    def test_direction_and_complete_grid(self):
        cells, draws, contrasts = summarize(self.rows)
        self.assertEqual((len(cells), len(draws), len(contrasts)), (120, 312, 192))
        self.assertTrue(all(abs(r['advantage_reduction'] - .4) < 1e-12 for r in contrasts))

    def test_missing_or_duplicate_samples_fail(self):
        for values in (self.rows[:-1], self.rows + [self.rows[0]]):
            with self.assertRaises(ValueError):
                summarize(values)

    def test_undefined_cosine_cannot_become_positive_evidence(self):
        rows = copy.deepcopy(self.rows)
        affected = next(r for r in rows if r['scheme'] == 'legacy' and r['noise_model'] == 'absolute')
        affected['cosine'] = None
        cells, draws, contrasts = summarize(rows)
        selected = [r for r in contrasts if r['metric'] == 'cosine'
                    and r['relative_target'] == affected['relative_target']
                    and r['parameter_name'] == affected['parameter_name']
                    and r['noise_level'] == affected['noise_level']
                    and r['noise_draw_index'] == affected['noise_draw_index']]
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(r['advantage_reduction'] is None for r in selected))

    def test_physical_ratios_pool_squares_before_dividing(self):
        states = [dict(scheme='baseline', relative_target=2., state_layer_index=0,
                       batch_index=b, element_count=16,
                       free_rms=1. if b < 18 else 3.,
                       centered_half_difference_rms=2. if b < 18 else 4.,
                       endpoint_rms=3. if b < 18 else 5.) for b in range(36)]
        physical, _ = scale_controls(states, [])
        self.assertAlmostEqual(physical[0]['D_over_F'], 2.**.5)
        self.assertAlmostEqual(physical[0]['D_over_P'], (10./17.)**.5)


if __name__ == '__main__':
    unittest.main()
