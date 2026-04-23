import csv
import json
from pathlib import Path
from types import SimpleNamespace

import main_experiment

def test_run_whole_population_evaluation_phase_writes_report_without_summary_key(tmp_path, monkeypatch):
    class DummyNetwork:
        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

        def to(self, device):
            return self

        def eval(self):
            return self

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def evaluate(self, **kwargs):
            return {
                'prob_opponent_0_p0.10': {
                    'average_reward': 2.5,
                    'cooperation_rate': 0.4,
                }
            }

    output_dirs = {
        'logs': str(tmp_path / 'logs'),
        'results': str(tmp_path / 'results'),
    }
    Path(output_dirs['logs']).mkdir(parents=True, exist_ok=True)
    Path(output_dirs['results']).mkdir(parents=True, exist_ok=True)

    checkpoint_path = tmp_path / 'dummy_checkpoint.pth'
    checkpoint_path.write_text('placeholder')

    monkeypatch.setattr(main_experiment.os.path, 'exists', lambda path: True)
    monkeypatch.setattr(main_experiment.torch, 'load', lambda *args, **kwargs: {
        'game_name': 'prisoners-dilemma',
        'task_id': 0,
        'model_state_dict': {},
    })
    monkeypatch.setattr(main_experiment, 'create_network', lambda *args, **kwargs: DummyNetwork())
    monkeypatch.setattr(main_experiment, 'GameTrainer', DummyTrainer)

    result = main_experiment.run_whole_population_evaluation_phase(
        checkpoint_path=str(checkpoint_path),
        test_game='prisoners-dilemma',
        test_opponents=[0.1],
        wp_config={'evaluation_config': {'num_sessions': 1, 'enable_detailed_testing': False}},
        network_config=SimpleNamespace(),
        device='cpu',
        output_dirs=output_dirs,
        model_id=0,
        test_task_id=0,
    )

    report_path = Path(output_dirs['results']) / 'eval_model_0_task_0_report.json'
    assert report_path.exists()

    report = json.loads(report_path.read_text())
    assert abs(report['mean_reward'] - 2.5) < 1e-6
    assert abs(report['mean_cooperation_rate'] - 0.4) < 1e-6
    assert result['experiment_type'] == 'whole_population'

def test_save_evaluation_metrics_to_csv_accepts_whole_population_condition(tmp_path):
    csv_path = tmp_path / 'whole_population_eval.csv'
    eval_results = {
        'prob_opponent_0_p0.10': {
            'average_reward': 1.5,
            'cooperation_rate': 0.6,
        }
    }

    main_experiment.save_evaluation_metrics_to_csv(
        eval_results,
        str(csv_path),
        {
            'game': 'prisoners-dilemma',
            'opponents': [0.1],
            'trained_game': 'prisoners-dilemma',
        }
    )

    with csv_path.open(newline='') as handle:
        row = next(csv.DictReader(handle))

    assert row['test_game'] == 'prisoners-dilemma'
    assert row['test_opponent_range'] == '0.1'
    assert abs(float(row['mean_reward']) - 1.5) < 1e-6


def test_main_whole_population_eval_uses_args_test_game(tmp_path, monkeypatch):
    captured = {}

    def fake_parse_arguments():
        return SimpleNamespace(
            experiment_mode='whole-population',
            task_id=15,
            run_id=None,
            matrix_config='config/generalization_matrix_config.json',
            wp_config='config/whole_population_config.json',
            config=None,
            train_game='prisoners-dilemma',
            training_games='prisoners-dilemma,hawk-dove',
            test_game='stag-hunt',
            train_opponents='0.1,0.3',
            test_opponents='0.3',
            opponents='0.1,0.3,0.5,0.7,0.9',
            equally_spaced_opponents=False,
            num_opponents=11,
            include_opponent_boundaries=False,
            segmented_experiments=False,
            opponents_per_segment=5,
            num_games=100,
            max_epochs=10,
            output_dir=str(tmp_path / 'outputs'),
            seed=42,
            device='cpu',
            adaptive_loss=False,
            verbose=False,
            experiment_name='whole_population_regression_test',
            agent_type='vanilla',
            mode='eval-only',
            checkpoint_path='dummy_checkpoint.pth',
            test_condition_ids=None,
        )

    def fake_create_output_dirs(base_dir, experiment_name):
        base = Path(base_dir) / experiment_name
        logs = base / 'logs'
        results = base / 'results'
        plots = base / 'plots'
        checkpoints = base / 'checkpoints'
        for directory in (logs, results, plots, checkpoints):
            directory.mkdir(parents=True, exist_ok=True)
        return {
            'base': str(base),
            'logs': str(logs),
            'results': str(results),
            'plots': str(plots),
            'checkpoints': str(checkpoints),
        }

    def fake_run_whole_population_experiment(**kwargs):
        captured.update(kwargs)
        return {'status': 'ok'}

    fake_logger = SimpleNamespace(info=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)

    monkeypatch.setattr(main_experiment, 'parse_arguments', fake_parse_arguments)
    monkeypatch.setattr(main_experiment, 'set_random_seeds', lambda seed: None)
    monkeypatch.setattr(main_experiment, 'setup_device', lambda device: 'cpu')
    monkeypatch.setattr(main_experiment, 'create_output_dirs', fake_create_output_dirs)
    monkeypatch.setattr(main_experiment, 'setup_logging', lambda level, log_file: fake_logger)
    monkeypatch.setattr(main_experiment, 'run_whole_population_experiment', fake_run_whole_population_experiment)

    main_experiment.main()

    assert captured['mode'] == 'eval-only'
    assert captured['task_id'] == 15
    assert captured['model_id'] == 1
    assert captured['test_game'] == 'stag-hunt'
    assert captured['test_opponents'] == [0.3]
    assert captured['checkpoint_path'] == 'dummy_checkpoint.pth'