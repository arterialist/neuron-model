"""Offline tests for the preserved CIFAR instruments, using synthetic data only."""
import contextlib
import importlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from cifar_ceiling import analyze_ceiling, common, run_kmeans_ceiling, run_learned_ladder


class CifarCeilingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def synthetic_data(self):
        rng = np.random.RandomState(17)
        train = rng.uniform(0, 255, (20, 3, 32, 32)).astype(np.float32)
        test = rng.uniform(0, 255, (4, 3, 32, 32)).astype(np.float32)
        return train, np.arange(20) % 2, test, np.arange(4) % 2

    def test_split_covers_all_examples_without_overlap_and_rejects_empty_splits(self):
        train, val = common.train_val_split(20, 0.2, 3)
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(set(train) | set(val), set(range(20)))
        for n, fraction in [(1, 0.2), (20, 0), (20, 1), (2, 0.01)]:
            with self.subTest(n=n, fraction=fraction), self.assertRaises(ValueError):
                common.train_val_split(n, fraction, 3)

    def test_numpy_and_torch_preprocessing_agree_and_preserve_fitted_statistics(self):
        rng = np.random.RandomState(5)
        train = rng.uniform(0, 255, (40, 12)).astype(np.float32)
        val = rng.uniform(0, 500, (8, 12)).astype(np.float32)
        pre = common.Preproc(2, 10, True, 0.1, True).fit(train)
        before_mean, before_matrix = pre.mean.copy(), pre.W.copy()
        np.testing.assert_allclose(pre.apply(val), pre.torch_apply(torch.from_numpy(val)).numpy(), atol=1e-6)
        np.testing.assert_array_equal(pre.mean, before_mean)
        np.testing.assert_array_equal(pre.W, before_matrix)

    def test_dataset_encoding_is_independent_of_batch_size(self):
        images = self.synthetic_data()[0][:5]
        patches = common.sample_patches(images, 32, 2, 7)
        pre = common.Preproc(2, 10, True, 0.1, True).fit(patches)
        dictionary = common.normalize_dict(common.random_patch_dict(pre.apply(patches), 4, 3))
        a = common.encode_dataset(images, dictionary, pre, 'cpu', 2, batch=1)
        b = common.encode_dataset(images, dictionary, pre, 'cpu', 2, batch=3)
        self.assertEqual(a.shape, (5, 16))
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)

    def test_contrasts_pair_by_seed_and_reject_duplicate_seed_records(self):
        records = [
            {'arm': 'random', 'seed': 2, 'test_acc': 20},
            {'arm': 'krotov', 'seed': 1, 'test_acc': 11},
            {'arm': 'random', 'seed': 1, 'test_acc': 10},
            {'arm': 'krotov', 'seed': 2, 'test_acc': 22},
            {'arm': 'krotov', 'seed': 7, 'test_acc': 99},
        ]
        seeds, differences = analyze_ceiling.paired_differences(records, 'random', 'krotov')
        self.assertEqual(seeds, [1, 2])
        np.testing.assert_array_equal(differences, [1, 2])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            analyze_ceiling.paired_differences(records + [records[0]], 'random', 'krotov')

    def test_cache_requires_matching_protocol_and_preserves_existing_records(self):
        args = SimpleNamespace(out='unused', verbose=False, readout_epochs=2, c_grid=[0.1])
        protocol = common.experiment_protocol(args, arm='random', K=4, seed=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            common.write_record(path, {'test_acc': 50}, protocol)
            before = path.read_bytes()
            self.assertEqual(common.read_cached_record(path, protocol)['test_acc'], 50)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                common.read_cached_record(path, {**protocol, 'settings': {'readout_epochs': 3}})
            with self.assertRaises(FileExistsError):
                common.write_record(path, {'test_acc': 60}, protocol)
            self.assertEqual(path.read_bytes(), before)
            legacy = Path(directory) / 'legacy.json'
            legacy.write_text('{"test_acc":69.55}')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                common.read_cached_record(legacy, protocol)

    def test_kmeans_runner_executes_a_tiny_synthetic_protocol(self):
        data = self.synthetic_data()
        with tempfile.TemporaryDirectory() as directory:
            argv = ['kmeans', '--stage', 'calibrate', '--out', directory, '--k', '4',
                    '--train-n', '20', '--n-patches', '32', '--kmeans-iters', '2',
                    '--readout-epochs', '2', '--c-grid', '0.1', '--device', 'cpu']
            with patch.object(sys, 'argv', argv), patch.object(run_kmeans_ceiling, 'load_cifar', return_value=data), contextlib.redirect_stdout(io.StringIO()):
                run_kmeans_ceiling.main()
            path = Path(directory) / 'canonical_K4_s0.json'
            record = json.loads(path.read_text())
            self.assertEqual(record['n_train'] + record['n_val'], 20)
            self.assertEqual(record['feat_dim'], 16)
            self.assertTrue(0 <= record['test_acc'] <= 100)
            self.assertEqual(record['protocol']['experiment']['cfg']['n_patches'], 32)
            before = path.read_bytes()
            with patch.object(sys, 'argv', argv), patch.object(run_kmeans_ceiling, 'load_cifar', side_effect=AssertionError('cache must not reacquire data')), contextlib.redirect_stdout(io.StringIO()):
                run_kmeans_ceiling.main()
            self.assertEqual(path.read_bytes(), before)

    def test_sweep_refuses_to_start_without_a_valid_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, 'argv', ['kmeans', '--stage', 'sweep', '--out', directory, '--device', 'cpu']), patch.object(run_kmeans_ceiling, 'load_cifar', side_effect=AssertionError('must not acquire data')), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    run_kmeans_ceiling.main()
            self.assertEqual(caught.exception.code, 2)

    def test_ladder_runner_executes_small_random_patch_kmeans_and_krotov_arms(self):
        data = self.synthetic_data()
        with tempfile.TemporaryDirectory() as directory:
            argv = ['ladder', '--arms', 'random,randpatch,kmeans,krotov', '--seeds', '0',
                    '--out', directory, '--k', '4', '--patch', '2', '--n-patches', '32',
                    '--kmeans-iters', '2', '--krotov-epochs', '1', '--readout-epochs', '2',
                    '--encode-batch', '4', '--c-grid', '0.1', '--device', 'cpu']
            with patch.object(sys, 'argv', argv), patch.object(run_learned_ladder, 'load_cifar', return_value=data), contextlib.redirect_stdout(io.StringIO()):
                run_learned_ladder.main()
            for arm in ['random', 'randpatch', 'kmeans', 'krotov']:
                record = json.loads((Path(directory) / f'{arm}_K4_s0.json').read_text())
                self.assertTrue(record['matched'])
                self.assertEqual(record['K'], 4)
                self.assertTrue(0 <= record['test_acc'] <= 100)
                self.assertEqual(record['protocol']['experiment']['arm'], arm)

    def test_sweep_analysis_excludes_ablation_records(self):
        records = [
            {'name': 'canonical', 'K': 400, 'test_acc': 70},
            {'name': 'sweep_K800', 'K': 800, 'test_acc': 75},
            {'name': 'scale1', 'K': 400, 'test_acc': 1, 'cfg': {'dict_rule': 'kmeans'}},
        ]
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(output):
            analyze_ceiling.do_sweep(records, directory)
            self.assertTrue((Path(directory) / 'ceiling_curve.png').is_file())
        self.assertIn('70.00%', output.getvalue())
        self.assertNotIn('35.50%', output.getvalue())

    def test_historical_helpers_do_not_start_jobs_when_imported(self):
        prefix = 'snn_classification_realtime.foveation.recovered_snn_debug_tools.scripts.'
        with patch('snn_classification_realtime.activity_dataset_builder.vision_datasets.load_dataset_by_name', side_effect=AssertionError('import must not acquire data')):
            for name in ['exp_teacher_modes', 'probe_memory_content', 'build_artifact', 'gen_artifact', 'progress']:
                module = importlib.import_module(prefix + name)
                self.assertTrue(callable(module.main))

    def test_progress_uses_retained_records_and_artifact_builder_refuses_overwrite(self):
        prefix = 'snn_classification_realtime.foveation.recovered_snn_debug_tools.scripts.'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'plast_one_seed0.json').write_text('{"complete": true}')
            (root / 'plast_two_seed0.json').write_text('{"traj_t":[10],"tag":"two"}')
            (root / 'plast_bad_seed0.json').write_text('partial')
            completed = subprocess.run([sys.executable, '-m', prefix + 'progress', '--input-root', directory], check=True, capture_output=True, text=True)
            self.assertIn('1/2 complete', completed.stdout)
            self.assertIn('unreadable: 1', completed.stdout)
            self.assertNotIn('/25', completed.stdout)
            output = root / 'original.html'
            output.write_text('preserved report')
            completed = subprocess.run([sys.executable, '-m', prefix + 'build_artifact', '--output', str(output)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2)
            self.assertIn('refusing to replace', completed.stderr)
            self.assertEqual(output.read_text(), 'preserved report')


if __name__ == '__main__':
    unittest.main()
