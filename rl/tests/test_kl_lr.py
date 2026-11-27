import csv

import torch

import rl.train as tr
from rl.config import Config, parse_args


def test_kl_lr_step_moves_toward_target_and_is_bounded():
    cfg = Config(lr_kl_target=0.002, lr_kl_step=1.1, lr_kl_min=1e-5, lr_kl_max=1e-3, lr_kl_ema=0.9)
    lr, ema = tr.kl_lr_step(2.5e-4, None, 0.008, cfg)
    assert ema == 0.008 and abs(lr - 2.5e-4 / 1.1) < 1e-12  # far above target: one bounded step down
    lr2, ema2 = tr.kl_lr_step(1e-4, 0.002, 0.002, cfg)
    assert lr2 == 1e-4 and ema2 == 0.002  # on target: unchanged
    lr3, _ = tr.kl_lr_step(1e-4, 0.0019, 0.0019, cfg)
    assert 1e-4 < lr3 < 1.1e-4  # slightly below target: small step up
    assert tr.kl_lr_step(1.0e-5, 0.1, 0.1, cfg)[0] == 1e-5  # floor
    assert tr.kl_lr_step(1.0e-3, 1e-9, 1e-9, cfg)[0] == 1e-3  # ceiling
    assert Config().lr_kl_target == 0.0


def test_tiny_training_adapts_lr_and_resumes_it(tmp_path):
    args = ['--preset', 'tiny', '--run_dir', str(tmp_path), '--lr_kl_target', '1e-6', '--lr', '1e-3']
    first = tr.train(parse_args(args + ['--total_updates', '3']))
    rows = list(csv.DictReader(open(tmp_path / 'progress.csv')))
    lrs = [float(r['lr']) for r in rows]
    assert lrs[0] == 1e-3 and lrs[1] < lrs[0] and lrs[2] < lrs[1]  # KL far above the tiny target: lr falls
    assert 'kl_ema' in rows[0]
    ck = torch.load(tmp_path / 'ckpt/latest.pt', weights_only=False)
    assert abs(ck['lr_now'] - first['lr_now']) < 1e-15 and ck['kl_ema'] is not None
    tr.train(parse_args(args + ['--total_updates', '4', '--resume', 'auto']))
    rows = list(csv.DictReader(open(tmp_path / 'progress.csv')))
    assert abs(float(rows[3]['lr']) - first['lr_now']) < 1e-12  # the 4th update starts from the saved lr
