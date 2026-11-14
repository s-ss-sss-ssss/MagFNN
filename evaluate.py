"""Evaluate an existing checkpoint without performing any training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model import DipoleNet, select_device
from evaluation import load_split, predict, metrics, effect_plot
from generate import B0, G0, BETA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path(__file__).parent/'data')
    p.add_argument('--split', choices=['validation','test','diagnostic_grid'], default='test')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu','mps','auto'], default='cpu')
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    for key, value in [('B0_T',B0), ('G0_T_per_m',G0), ('beta',BETA)]:
        if not np.isclose(checkpoint['normalization'][key], value, rtol=1e-12, atol=0):
            raise ValueError('Checkpoint normalization mismatch')
    model = DipoleNet().to(device); model.load_state_dict(checkpoint['model_state'])
    data = load_split(args.data, args.split)
    prediction = predict(model, data['X'], device)
    report = dict(split=args.split, checkpoint=str(args.checkpoint.resolve()), epoch=checkpoint['epoch'],
                  network=metrics(prediction, data['Y'], data['position_m']),
                  uncorrected=metrics(data['X'][:, :8], data['Y'], data['position_m']))
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Output directory must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    effect_plot(prediction, data, args.output/'effects.png', args.split)
    np.savez_compressed(args.output/'predictions.npz', prediction=prediction,
                        target=data['Y'], position_m=data['position_m'])
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
