"""Interface, metric and plot checks only: no backward pass or optimizer step."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from model import DipoleNet, loss_parts
from generate import make_dataset
from evaluation import metrics, errors, predict, history_plot, effect_plot, load_split


class TrainingCodeChecks(unittest.TestCase):
    def setUp(self):
        self.data = make_dataset(np.array([[0.,0.,.15], [.1,0.,.3], [-.15,.15,.45]]))

    def test_network_shape_and_loss_definition(self):
        model = DipoleNet()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 9480)
        x = torch.tensor(self.data['X'], dtype=torch.float32)
        with torch.inference_mode():
            output = model(x)
            self.assertEqual(tuple(output.shape), (3,8))
            total, b, g = loss_parts(torch.ones((3,8)), torch.zeros((3,8)))
            self.assertEqual((total.item(), b.item(), g.item()), (2.,1.,1.))
            self.assertEqual(predict(model, self.data['X'], torch.device('cpu'), 2).shape, (3,8))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_physical_metric_reference(self):
        d = self.data
        exact = metrics(d['Y'], d['Y'], d['position_m'])
        self.assertEqual(exact['loss'], 0.)
        self.assertEqual(exact['G_NRMSE'], 0.)
        self.assertLess(exact['Euler_relative']['p95'], 1e-12)
        double = metrics(2*d['Y'], d['Y'], d['position_m'])
        self.assertAlmostEqual(double['B_NRMSE'], 1.)
        self.assertAlmostEqual(double['G_relative']['median'], 1.)
        baseline = errors(d['X'][:,:8], d['Y'], d['position_m'])
        self.assertTrue(np.all(baseline['B_relative'] > 0))
        _, b3, _ = __import__('generate').evaluate(d['position_m'], 3)
        bt = d['B_target_T']
        expected = 2*np.linalg.norm(b3, axis=1)/(3*np.linalg.norm(bt, axis=1))
        np.testing.assert_allclose(baseline['Euler_relative'], expected, rtol=1e-12)

    def test_plots_and_checkpoint_roundtrip(self):
        d = self.data
        m = metrics(d['Y'], d['Y'], d['position_m'])
        base = metrics(d['X'][:,:8], d['Y'], d['position_m'])
        history = [dict(epoch=1, train_loss=.01, validation_loss=.02,
                        validation_metrics=m, baseline_metrics=base)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            history_plot(history, path/'progress.png')
            effect_plot(d['Y'], d, path/'effects.png', 'Synthetic metric check')
            for name in ['progress.png', 'effects.png']:
                self.assertGreater((path/name).stat().st_size, 1000)
            model = DipoleNet()
            torch.save({'model_state':model.state_dict()}, path/'untrained.pt')
            restored = DipoleNet()
            restored.load_state_dict(torch.load(path/'untrained.pt', weights_only=True)['model_state'])
            np.testing.assert_array_equal(predict(model, d['X'], torch.device('cpu')),
                                          predict(restored, d['X'], torch.device('cpu')))

    def test_actual_data_interface(self):
        d = load_split(Path(__file__).parent/'data', 'validation')
        self.assertEqual(d['X'].shape, (10000,9))


if __name__ == '__main__': unittest.main()
