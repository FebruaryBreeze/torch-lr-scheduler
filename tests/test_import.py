import unittest


class MyTestCase(unittest.TestCase):
    def test_import(self):
        import torch_lr_scheduler
        self.assertIsNotNone(torch_lr_scheduler)


if __name__ == '__main__':
    unittest.main()
