"""Coverage counts must follow the selected checkpoint roles before GPU replay."""
from experiments.run_phase_displacement_t_sweep import completion_counts


def test_initialization_only_completion_counts_do_not_include_trained_role():
    config = dict(dataset={'expected_batch_guards': list(range(36))},
                  checkpoint_roles=['reconstructed_initialization'],
                  cases=[{'architecture': 'conv2'}])
    assert completion_counts(config) == {
        'production_replay_count': 36,
        'production_layer_comparison_count': 108,
    }
    config['checkpoint_roles'].append('best_validation')
    assert completion_counts(config) == {
        'production_replay_count': 72,
        'production_layer_comparison_count': 216,
    }
