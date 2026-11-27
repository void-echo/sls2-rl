import contextlib
import csv
import io
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import rl.train as tr
from rl.config import REPO, Config, parse_args
from rl.model import MLP, ModelCfg, Policy, SwiGLU, n_params
from rl.ppo import Rollout, ValueNorm, ppo_update


BASELINE_TRAIN = '''
import itertools
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import rl.ppo as ppo
import rl.train as tr
from rl.config import parse_args
from rl.encode import Encoder
from rl.fake_engine import FakeServer

torch.set_num_threads(1)
random.seed(17)
ticks = itertools.count()
tr.time = ppo.time = SimpleNamespace(time=lambda: next(ticks) * .001)
cfg = parse_args(['--preset', 'tiny', '--seed', '17', '--run_dir', sys.argv[1],
                  '--n_workers', '1', '--envs_per_worker', '1', '--rollout_steps', '128',
                  '--total_updates', '3'])
assert tr.train(cfg)['update'] == 3
model, _, _, _ = tr.load_policy(Path(sys.argv[1]) / 'ckpt/latest.pt')
model.eval()
srv, enc = FakeServer(), Encoder(cfg.caps)
obs = []
for seed in ['old-a', 'old-b', 'old-c']:
    r = srv.handle(dict(cmd='rl_start', seed=seed))
    obs.append(enc.encode(r))
batch = ppo.to_torch({k: np.stack([o[k] for o in obs]) for k in obs[0]}, 'cpu')
batch = ppo.sliced(batch, torch.arange(len(obs)))
with torch.no_grad():
    logits, value = model(batch)
torch.save(dict(batch=batch, logits=logits, value=value), Path(sys.argv[1]) / 'forward.pt')
'''


@pytest.fixture(scope='module')
def baseline():
    os.makedirs(Path(REPO) / 'runs', exist_ok=True)
    with tempfile.TemporaryDirectory(dir=Path(REPO) / 'runs', prefix='kd_baseline_') as root:
        root = Path(root)
        source = root / 'baseline'
        archive = subprocess.run(['git', 'archive', '076ccb0', 'rl'], cwd=REPO, check=True, capture_output=True)
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(source, filter='data')
        for code, run in [(source, root / 'before'), (Path(REPO), root / 'after')]:
            env = {**os.environ, 'PYTHONPATH': str(code), 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            subprocess.run([sys.executable, '-c', BASELINE_TRAIN, str(run)], cwd=code, env=env,
                           check=True, capture_output=True, text=True, timeout=120)
        yield root


def test_disabled_training_matches_076ccb0_bytes(baseline):
    assert (baseline / 'before/progress.csv').read_bytes() == (baseline / 'after/progress.csv').read_bytes()
    with (baseline / 'after/progress.csv').open() as f:
        assert not any(k.startswith('distill_') for k in next(csv.reader(f)))
    ck = torch.load(baseline / 'after/ckpt/latest.pt', weights_only=False)
    assert 'distill_update' not in ck


def test_old_checkpoint_forward_is_identical(baseline):
    model, _, cfg, ck = tr.load_policy(baseline / 'before/ckpt/latest.pt')
    assert 'teacher' not in ck['cfg']
    assert {'value_tower', 'value_layers', 'mlp'}.isdisjoint(ck['cfg']['model'])
    assert cfg.teacher == '' and not cfg.model.value_tower and cfg.model.mlp == 'gelu'
    expected = torch.load(baseline / 'before/forward.pt', weights_only=False)
    with torch.no_grad():
        logits, value = model.eval()(expected['batch'])
    assert torch.equal(logits, expected['logits'])
    assert torch.equal(value, expected['value'])


@pytest.mark.parametrize('mlp', ['gelu', 'swiglu'])
@pytest.mark.parametrize('value_layers', [0, 3])
def test_value_tower_gradients_are_separate(baseline, mlp, value_layers):
    torch.manual_seed(5)
    model = Policy(ModelCfg(d=32, layers=2, heads=4, value_tower=True, value_layers=value_layers, mlp=mlp))
    batch = torch.load(baseline / 'before/forward.pt', weights_only=False)['batch']
    assert len(model.value_blocks) == (value_layers or 2)
    if mlp == 'swiglu':
        for block in [*model.blocks, *model.cand_blocks, *model.value_blocks]:
            assert isinstance(block.mlp[1], SwiGLU)
    _, value = model(batch)
    (value - 2).square().mean().backward()
    for name, p in model.named_parameters():
        if name == 'table' or name.startswith(('value.', 'value_')):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        else:
            assert p.grad is None or torch.count_nonzero(p.grad) == 0, name
    assert model.table.grad.abs().sum() > 0
    assert model.value_blocks[0].attn.q.weight.grad.abs().sum() > 0
    assert model.value[1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('d', [64, 256])
@pytest.mark.parametrize('value_tower', [False, True])
def test_swiglu_parameter_budget(d, value_tower):
    gelu = n_params(MLP(d, 4))
    swiglu = MLP(d, 4, 'swiglu')
    assert swiglu[2].in_features == (int(2 / 3 * 4 * d) + 7) // 8 * 8
    assert abs(n_params(swiglu) / gelu - 1) <= .05
    counts = [n_params(Policy(ModelCfg(d=d, value_tower=value_tower, mlp=kind))) for kind in ['gelu', 'swiglu']]
    assert abs(counts[1] / counts[0] - 1) <= .05


class FixedPolicy(torch.nn.Module):
    def __init__(self, logits, values):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))
        self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, b):
        ix = b['decision'].long()
        return self.logits[ix], self.values[ix]


@pytest.mark.parametrize('student_norm,teacher_norm', [(True, True), (True, False), (False, True), (False, False)])
def test_distillation_mask_normalization_and_sample_mean(student_norm, teacher_norm):
    cfg = Config(epochs=1, minibatch=3, micro_batch=2, value_norm=student_norm)
    buf = Rollout(cfg.caps, 3, 1, torch.device('cpu'))
    buf.n = 3
    buf.obs['n_tok'][:] = 1
    buf.obs['n_cand'][:] = torch.tensor([2, 1, 3])
    buf.obs['decision'][:] = torch.arange(3)
    student = FixedPolicy([[.2, 1.1, -1e9], [0, -1e9, -1e9], [.1, -.3, 1.2]], [1, 2, 3])
    teacher = FixedPolicy([[1.5, -.2, 1e4], [.2, 1e4, 1e4], [.4, .9, -.7]], [2, -1, .5]).eval()
    vn, tvn = ValueNorm(), ValueNorm()
    vn.update([1, 3])
    tvn.update([10, 14])
    ret = np.array([2, 4, 6], np.float32)
    out = ppo_update(student, torch.optim.SGD(student.parameters(), lr=0), buf, np.zeros(3, np.float32),
                     ret, np.ones(3, bool), vn, cfg, np.random.default_rng(0), contextlib.nullcontext,
                     teacher=(teacher, tvn, Config(value_norm=teacher_norm)), distill_coef_now=.4)
    cmask = torch.arange(3)[None] < buf.obs['n_cand'].long()[:, None]
    p = torch.log_softmax(teacher.logits.detach().masked_fill(~cmask, -1e9), -1)
    q = torch.log_softmax(student.logits.detach(), -1)
    expected_kl = (p.exp() * (p - q)).masked_fill(~cmask, 0).sum(-1).mean()
    target = tvn.denormalize(teacher.values.detach()) if teacher_norm else teacher.values.detach()
    if student_norm:
        target = vn.normalize(target)
    expected_vmse = (student.values.detach() - target).square().mean()
    assert out['distill_coef_now'] == .4
    assert out['distill_kl'] == pytest.approx(expected_kl.item(), abs=1e-6)
    assert out['distill_vmse'] == pytest.approx(expected_vmse.item(), rel=1e-6)
    assert all(p.grad is None for p in teacher.parameters())


def test_distillation_reduces_kl_from_random_student(baseline):
    teacher, tvn, tcfg, _ = tr.load_policy(baseline / 'before/ckpt/latest.pt')
    teacher.eval().requires_grad_(False)
    batch = torch.load(baseline / 'before/forward.pt', weights_only=False)['batch']
    cfg = parse_args(['--preset', 'tiny', '--model.value_tower', 'true', '--model.mlp', 'swiglu',
                      '--model.d', '32', '--model.layers', '2', '--epochs', '2', '--micro_batch', '2'])
    torch.manual_seed(9)
    student = Policy(cfg.model)
    vn = ValueNorm()
    vn.load_state_dict(tvn.state_dict())
    n = len(batch['n_tok'])
    buf = Rollout(cfg.caps, n, 1, torch.device('cpu'))
    buf.add([0] * n, batch, np.zeros(n, np.int64), np.zeros(n, np.float32), np.zeros(n, np.float32))
    with torch.no_grad():
        _, teacher_v = teacher(batch)
    ret = tvn.denormalize(teacher_v).numpy()
    opt = torch.optim.AdamW(student.parameters(), lr=.001)
    rng = np.random.default_rng(3)
    stats = [ppo_update(student, opt, buf, np.zeros(n, np.float32), ret, np.ones(n, bool), vn, cfg, rng,
                        contextlib.nullcontext, teacher=(teacher, tvn, tcfg), distill_coef_now=1.) for _ in range(12)]
    assert stats[0]['distill_kl'] > 1e-5
    assert np.mean([s['distill_kl'] for s in stats[-3:]]) < .3 * stats[0]['distill_kl']
    assert all(p.grad is None for p in teacher.parameters())


@pytest.mark.parametrize('cap,value', [('max_tok', 192), ('enc_rev', 1)])
def test_teacher_caps_must_match(baseline, tmp_path, monkeypatch, cap, value):
    monkeypatch.setattr(tr, 'ActorPool', lambda *a, **k: pytest.fail('actors must not start on caps mismatch'))
    cfg = parse_args(['--preset', 'tiny', '--run_dir', str(tmp_path),
                      '--teacher', str(baseline / 'before/ckpt/latest.pt'), f'--caps.{cap}', str(value)])
    with pytest.raises(ValueError, match='teacher caps must match student caps exactly'):
        tr.train(cfg)


def test_distillation_schedule_resume_and_teacher_lifecycle(baseline, tmp_path, monkeypatch):
    real_load, real_update, real_collect = tr.load_policy, tr.ppo_update, tr.Collector.collect
    teachers, calls, coefficients, norms = [], [], [], []

    def load(*args, **kwargs):
        model, vn, cfg, ck = real_load(*args, **kwargs)
        teachers.append(model)
        def forward(m, batch):
            assert not m.training and not torch.is_grad_enabled()
            assert all(not p.requires_grad for p in m.parameters())
            calls.append(len(coefficients))
        model.register_forward_pre_hook(forward)
        return model, vn, cfg, ck

    def collect(self, buf):
        assert self.model is not teachers[-1]
        before = len(calls)
        result = real_collect(self, buf)
        assert len(calls) == before
        return result

    def update(*args, **kwargs):
        coef = kwargs['distill_coef_now']
        coefficients.append(coef)
        norms.append(args[6].state_dict())
        before = len(calls)
        result = real_update(*args, **kwargs)
        assert len(calls) - before == (4 if coef > 0 else 0)
        assert all(p.grad is None for p in teachers[-1].parameters())
        return result

    monkeypatch.setattr(tr, 'load_policy', load)
    monkeypatch.setattr(tr, 'ppo_update', update)
    monkeypatch.setattr(tr.Collector, 'collect', collect)
    args = ['--preset', 'tiny', '--run_dir', str(tmp_path), '--n_workers', '1', '--envs_per_worker', '1',
            '--rollout_steps', '64', '--epochs', '1', '--minibatch', '32', '--micro_batch', '16',
            '--teacher', str(baseline / 'before/ckpt/latest.pt'), '--distill_updates', '4',
            '--model.value_tower', 'true', '--model.mlp', 'swiglu']
    assert tr.train(parse_args(args + ['--total_updates', '2']))['distill_update'] == 2
    ck = torch.load(tmp_path / 'ckpt/latest.pt', weights_only=False)
    assert tr.train(parse_args(args + ['--total_updates', '6', '--resume', 'auto']))['distill_update'] == 6
    teacher_ck = torch.load(baseline / 'before/ckpt/latest.pt', weights_only=False)
    assert norms[0] == teacher_ck['vnorm']
    assert norms[2] == ck['vnorm'] and norms[2] != teacher_ck['vnorm']
    assert coefficients == [1., .75, .5, .25, 0., 0.]
    assert len(calls) == 16
    with (tmp_path / 'progress.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert [float(r['distill_coef_now']) for r in rows] == coefficients
    assert all({'distill_kl', 'distill_vmse'} <= r.keys() for r in rows)
    assert all(float(r['distill_kl']) == float(r['distill_vmse']) == 0 for r in rows[-2:])


def test_distillation_starts_at_zero_when_resuming_old_experiment(baseline, tmp_path):
    cfg = parse_args(['--preset', 'tiny', '--run_dir', str(tmp_path), '--n_workers', '1', '--envs_per_worker', '1',
                      '--rollout_steps', '32', '--epochs', '1', '--teacher', str(baseline / 'before/ckpt/latest.pt'),
                      '--resume', str(baseline / 'before/ckpt/latest.pt'), '--total_updates', '4', '--distill_updates', '4'])
    state = tr.train(cfg)
    assert state['update'] == 4 and state['distill_update'] == 1
    with (tmp_path / 'progress.csv').open() as f:
        assert float(next(csv.DictReader(f))['distill_coef_now']) == 1.
