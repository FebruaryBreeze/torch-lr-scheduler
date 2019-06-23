import unittest

import torch.nn
import torch.optim

import torch_lr_scheduler


class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, kernel_size=3)

    def forward(self):
        pass


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.model = MockModule()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0)

    def test_lr_scheduler(self):
        lr_scheduler = torch_lr_scheduler.LrScheduler.factory(config={
            'line_chain': [{
                'ratio': 1.0,
                'target': 0.2
            }]
        })
        lr_scheduler.update(self.optimizer, ratio=0.0)
        self.assertEqual(lr_scheduler.lr, 0.2)
        lr_scheduler.update(self.optimizer, ratio=1.0)
        self.assertEqual(lr_scheduler.lr, 0.2)

        lr_scheduler = torch_lr_scheduler.LrScheduler.factory(config={
            'line_chain': [{
                'mode': 'linear',
                'ratio': 0.8,
                'start': 0.2,
                'target': 0.4
            }]
        })
        lr_scheduler.update(self.optimizer, ratio=0.0)
        self.assertEqual(lr_scheduler.lr, 0.2)
        lr_scheduler.update(self.optimizer, ratio=0.4)
        self.assertAlmostEqual(lr_scheduler.lr, 0.3)
        lr_scheduler.update(self.optimizer, ratio=0.8)
        self.assertEqual(lr_scheduler.lr, 0.4)
        lr_scheduler.update(self.optimizer, ratio=1.0)
        self.assertEqual(lr_scheduler.lr, 0.4)

        lr_scheduler = torch_lr_scheduler.LrScheduler.factory(config={
            'line_chain': [{
                'mode': 'fixed',
                'ratio': 0.8,
                'target': 0.4
            }, {
                'mode': 'cosine',
                'ratio': 1.0,
                'target': 0.0
            }]
        })
        lr_scheduler.update(self.optimizer, ratio=0.0)
        self.assertEqual(lr_scheduler.lr, 0.4)
        lr_scheduler.update(self.optimizer, ratio=0.4)
        self.assertAlmostEqual(lr_scheduler.lr, 0.4)
        lr_scheduler.update(self.optimizer, ratio=0.8)
        self.assertEqual(lr_scheduler.lr, 0.4)
        lr_scheduler.update(self.optimizer, ratio=0.9)
        self.assertAlmostEqual(lr_scheduler.lr, 0.2)
        lr_scheduler.update(self.optimizer, ratio=1.0)
        self.assertEqual(lr_scheduler.lr, 0.0)

        lr_scheduler = torch_lr_scheduler.LrScheduler.factory(config={
            'learning_rate_scale': 2.0,
            'line_chain': [{
                'mode': 'fixed',
                'ratio': 0.8,
                'target': 0.4
            }, {
                'mode': 'cosine',
                'ratio': 1.0,
                'target': 0.0
            }]
        })
        lr_scheduler.update(self.optimizer, ratio=0.0)
        self.assertEqual(lr_scheduler.lr, 0.8)
        lr_scheduler.update(self.optimizer, ratio=0.4)
        self.assertAlmostEqual(lr_scheduler.lr, 0.8)
        lr_scheduler.update(self.optimizer, ratio=0.8)
        self.assertAlmostEqual(lr_scheduler.lr, 0.8)
        lr_scheduler.update(self.optimizer, ratio=0.9)
        self.assertAlmostEqual(lr_scheduler.lr, 0.4)
        lr_scheduler.update(self.optimizer, ratio=1.0)
        self.assertAlmostEqual(lr_scheduler.lr, 0.0)

        self.assertTrue(str(lr_scheduler).startswith('LrScheduler'))
        print(lr_scheduler)


if __name__ == '__main__':
    unittest.main()
