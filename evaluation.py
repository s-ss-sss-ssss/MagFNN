"""Physical metrics and plots shared by training and standalone evaluation."""
import json
from pathlib import Path
import numpy as np
import torch
from generate import B0, G0, BETA, unpack


def load_split(directory, split):
    meta = json.loads((Path(directory)/'metadata.json').read_text())
    for key, expected in [('B0_T', B0), ('G0_T_per_m', G0), ('beta', BETA)]:
        if not np.isclose(meta['normalization'][key], expected, rtol=1e-12, atol=0):
            raise ValueError(f'Incompatible normalization: {key}')
    with np.load(Path(directory)/f'{split}.npz', allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    x, y = data['X'], data['Y']
    if x.shape != (len(y), 9) or y.shape != (len(y), 8) or not len(y):
        raise ValueError('Expected nonempty X(N,9), Y(N,8)')
    if not all(np.all(np.isfinite(v)) for v in data.values()):
        raise ValueError('Nonfinite data')
    return data


@torch.inference_mode()
def predict(model, x, device, batch_size=4096):
    model.eval()
    return np.concatenate([model(torch.as_tensor(x[i:i+batch_size], dtype=torch.float32,
                              device=device)).cpu().numpy() for i in range(0, len(x), batch_size)])


def errors(pred, target, position):
    b, g = unpack(np.asarray(pred, dtype=np.float64))
    bt, gt = unpack(target)
    bn = np.maximum(np.linalg.norm(bt, axis=1), 1e-30)
    gn = np.maximum(np.linalg.norm(gt, axis=(1, 2)), 1e-30)
    return dict(B_relative=np.linalg.norm(b-bt, axis=1)/bn,
                G_relative=np.linalg.norm(g-gt, axis=(1, 2))/gn,
                Euler_relative=np.linalg.norm(np.einsum('nij,nj->ni', g, position)+3*b, axis=1)/(3*bn))


def metrics(pred, target, position):
    residual = np.asarray(pred, dtype=np.float64)-target
    result = dict(loss=float(np.mean(residual[:, :3]**2)+np.mean(residual[:, 3:]**2)))
    for name, v in errors(pred, target, position).items():
        result[name] = dict(median=float(np.median(v)), p95=float(np.quantile(v, .95)),
                            mean=float(np.mean(v)))
    b, g = unpack(pred); bt, gt = unpack(target)
    result['B_NRMSE'] = float(np.linalg.norm(b-bt)/np.linalg.norm(bt))
    result['G_NRMSE'] = float(np.linalg.norm(g-gt)/np.linalg.norm(gt))
    return result


def history_plot(history, path, live=False):
    import matplotlib.pyplot as plt
    fig = plt.figure('Training progress', figsize=(11, 8))
    fig.clear()
    axes = fig.subplots(2, 2)
    epoch = [v['epoch'] for v in history]
    for key, label in [('train_loss', 'Train'), ('validation_loss', 'Validation')]:
        axes[0, 0].semilogy(epoch, np.maximum([v[key] for v in history], 1e-16), label=label)
    axes[0, 0].set_title('Normalized loss'); axes[0, 0].legend()
    for ax, key, title in [(axes[0, 1], 'B_relative', 'Field relative error (%)'),
                            (axes[1, 0], 'G_relative', 'Gradient relative error (%)'),
                            (axes[1, 1], 'Euler_relative', 'Euler residual (%)')]:
        for stat in ['median', 'p95']:
            ax.plot(epoch, [100*v['validation_metrics'][key][stat] for v in history], label=stat)
        baseline = history[0]['baseline_metrics'][key]['median']*100
        ax.axhline(baseline, color='gray', linestyle='--', label='Uncorrected median')
        ax.set_title(title); ax.legend()
    for ax in axes.flat:
        ax.set_xlabel('Epoch'); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=140)
    if live:
        plt.show(block=False); plt.pause(.01)
    else:
        plt.close(fig)


def effect_plot(pred, data, path, title):
    import matplotlib.pyplot as plt
    baseline = data['X'][:, :8]
    target = data['Y']
    model_errors = errors(pred, target, data['position_m'])
    raw_errors = errors(baseline, target, data['position_m'])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, sl, label in [(axes[0, 0], slice(0, 3), 'Field / B0'),
                           (axes[0, 1], slice(3, 8), 'Gradient / G0')]:
        idx = np.linspace(0, len(pred)-1, min(2000, len(pred)), dtype=int)
        truth = target[idx, sl].ravel()
        ax.scatter(truth, baseline[idx, sl].ravel(), s=3, alpha=.25, label='Uncorrected')
        ax.scatter(truth, pred[idx, sl].ravel(), s=3, alpha=.25, label='Network')
        lo, hi = truth.min(), truth.max(); ax.plot([lo, hi], [lo, hi], 'k--')
        ax.set_xlabel('Target'); ax.set_ylabel('Prediction'); ax.set_title(label); ax.legend()
    radius = np.linalg.norm(data['position_m'], axis=1)
    for ax, key in zip(axes[1], ['B_relative', 'G_relative', 'Euler_relative']):
        idx = np.linspace(0, len(pred)-1, min(3000, len(pred)), dtype=int)
        ax.scatter(radius[idx], raw_errors[key][idx]*100, s=3, alpha=.3, label='Uncorrected')
        ax.scatter(radius[idx], model_errors[key][idx]*100, s=3, alpha=.3, label='Network')
        ax.set_xlabel('Distance (m)'); ax.set_ylabel('Error (%)'); ax.set_title(key); ax.legend()
    ax = axes[0, 2]
    for values, label in [(raw_errors['G_relative'], 'Uncorrected'), (model_errors['G_relative'], 'Network')]:
        values = np.sort(values)*100
        ax.plot(values, np.arange(1, len(values)+1)/len(values), label=label)
    ax.set_xlabel('Gradient error (%)'); ax.set_ylabel('Cumulative fraction'); ax.legend()
    fig.suptitle(title)
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
