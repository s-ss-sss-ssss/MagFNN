"""Explicit training entrypoint. Importing this module never starts training."""
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import DipoleNet, loss_parts, select_device
from generate import B0, G0, BETA
from evaluation import load_split, predict, metrics, history_plot, effect_plot

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'data')
    p.add_argument('--output', type=Path, default=ROOT/'runs'/'baseline')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--seed', type=int, default=4098)
    p.add_argument('--device', choices=['cpu', 'mps', 'auto'], default='cpu')
    p.add_argument('--live', action='store_true', help='Show a live plot window as well as saving PNGs')
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.patience) <= 0 or not np.isfinite(args.lr) or args.lr <= 0:
        p.error('epochs, batch-size, patience and lr must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Output directory is nonempty; choose a new --output to preserve prior results')
    import matplotlib
    if not args.live: matplotlib.use('Agg')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = select_device(args.device)
    train = load_split(args.data, 'train'); validation = load_split(args.data, 'validation')
    loader = DataLoader(TensorDataset(torch.tensor(train['X'], dtype=torch.float32),
                                     torch.tensor(train['Y'], dtype=torch.float32)),
                        batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    model = DipoleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(device_used=str(device), architecture=[9,64,64,64,8], activation='SiLU',
                  optimizer='Adam', dtype='float32', parameters=sum(v.numel() for v in model.parameters()),
                  torch_version=str(torch.__version__), loss='mean(B residual squared)+mean(5 G residual squared)')
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    baseline = metrics(validation['X'][:, :8], validation['Y'], validation['position_m'])
    history = []; best = float('inf'); stale = 0
    print(f'Device={device}, parameters={config["parameters"]}; baseline validation loss={baseline["loss"]:.6g}', flush=True)
    for epoch in range(1, args.epochs+1):
        model.train(); total = field = gradient = 0.
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x); loss, lb, lg = loss_parts(pred, y)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward(); optimizer.step()
            total += loss.item()*len(x); field += lb.item()*len(x); gradient += lg.item()*len(x)
        vp = predict(model, validation['X'], device)
        vm = metrics(vp, validation['Y'], validation['position_m'])
        if not np.isfinite(vm['loss']): raise RuntimeError('Nonfinite validation loss')
        row = dict(epoch=epoch, train_loss=total/len(train['X']), train_B_loss=field/len(train['X']),
                   train_G_loss=gradient/len(train['X']), validation_loss=vm['loss'],
                   validation_metrics=vm, baseline_metrics=baseline)
        history.append(row)
        checkpoint = dict(model_state=model.state_dict(), epoch=epoch, config=config,
                          validation_metrics=vm, normalization=dict(B0_T=B0,
                          G0_T_per_m=G0, beta=BETA))
        if vm['loss'] < best:
            best = vm['loss']; stale = 0
            torch.save(checkpoint, args.output/'best.pt')
        else: stale += 1
        torch.save({**checkpoint, 'optimizer_state': optimizer.state_dict()}, args.output/'last.pt')
        (args.output/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        with (args.output/'history.csv').open('w', newline='') as f:
            rows = [{**{k:v for k,v in r.items() if not isinstance(v, dict)},
                     **{f'val_{key}_{stat}': val for key, item in r['validation_metrics'].items()
                        if isinstance(item, dict) for stat, val in item.items()}} for r in history]
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        history_plot(history, args.output/'training_progress.png', args.live)
        print(f'Epoch {epoch:04d} train={row["train_loss"]:.6g} val={vm["loss"]:.6g} '
              f'B median/P95={vm["B_relative"]["median"]:.2%}/{vm["B_relative"]["p95"]:.2%} '
              f'G median/P95={vm["G_relative"]["median"]:.2%}/{vm["G_relative"]["p95"]:.2%}', flush=True)
        if stale >= args.patience:
            print('Early stopping: validation loss did not improve.', flush=True); break
    # Test data are first loaded after checkpoint selection has finished.
    model.load_state_dict(torch.load(args.output/'best.pt', map_location=device, weights_only=True)['model_state'])
    report = {}
    for split in ['validation', 'test', 'diagnostic_grid']:
        data = validation if split == 'validation' else load_split(args.data, split)
        output = predict(model, data['X'], device)
        report[split] = dict(network=metrics(output, data['Y'], data['position_m']),
                             uncorrected=metrics(data['X'][:, :8], data['Y'], data['position_m']))
        effect_plot(output, data, args.output/f'{split}_effects.png', split)
        np.savez_compressed(args.output/f'{split}_predictions.npz', prediction=output,
                            target=data['Y'], position_m=data['position_m'])
    (args.output/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Finished. Best validation loss={best:.6g}. Results: {args.output}', flush=True)


if __name__ == '__main__': main()
