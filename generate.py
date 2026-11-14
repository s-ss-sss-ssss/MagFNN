"""Generate and independently verify analytic D4098 training data (SI, float64)."""
import argparse
import json
from pathlib import Path
import numpy as np

M1 = 93.56451259694958
M3 = 0.287540068
C = 1e-7  # mu0/(4*pi), conventional value used by the experimental model
A = 0.1
B0 = C * M1 / A**3
G0 = B0 / A
BETA = M3 / (M1 * A**2)
LOW = np.array([-0.15, -0.15, 0.15])
HIGH = np.array([0.15, 0.15, 0.45])
COMPONENTS = ['Bx', 'By', 'Bz', 'Gxx', 'Gyy', 'Gxy', 'Gxz', 'Gyz']


def term(position, p, k):
    """Value, gradient and Hessian of z**p / r**k; nonsingular on the axis."""
    x = np.asarray(position, dtype=np.float64)
    r = np.linalg.norm(x, axis=1)
    if np.any(r == 0):
        raise ValueError('Source origin is singular')
    z = x[:, 2]
    h = z**p
    dh = p * z**(p-1)
    ddh = p*(p-1)*z**(p-2) if p >= 2 else np.zeros_like(z)
    e = np.array([0., 0., 1.])
    cross = e[None, :, None]*x[:, None, :] + x[:, :, None]*e[None, None, :]
    grad = dh[:, None]*e*r[:, None]**(-k) - k*h[:, None]*x*r[:, None]**(-k-2)
    hess = (ddh[:, None, None]*np.outer(e, e)*r[:, None, None]**(-k)
            - k*dh[:, None, None]*cross*r[:, None, None]**(-k-2)
            - k*h[:, None, None]*np.eye(3)*r[:, None, None]**(-k-2)
            + k*(k+2)*h[:, None, None]*x[:, :, None]*x[:, None, :]*r[:, None, None]**(-k-4))
    return h*r**(-k), grad, hess


def evaluate(position, degree):
    if degree == 1:
        v, d, h = term(position, 1, 3)
        scale = C*M1
    elif degree == 3:
        v1, d1, h1 = term(position, 3, 7)
        v2, d2, h2 = term(position, 1, 5)
        v, d, h = (5*v1-3*v2)/2, (5*d1-3*d2)/2, (5*h1-3*h2)/2
        scale = C*M3
    else:
        raise ValueError('degree must be 1 or 3')
    return scale*v, -scale*d, -scale*h


def pack(b, g):
    return np.column_stack((b/B0, g[:, 0, 0]/G0, g[:, 1, 1]/G0,
                            g[:, 0, 1]/G0, g[:, 0, 2]/G0, g[:, 1, 2]/G0))


def unpack(v):
    b = v[:, :3]*B0
    g = np.zeros((len(v), 3, 3), dtype=np.float64)
    g[:, 0, 0], g[:, 1, 1] = v[:, 3]*G0, v[:, 4]*G0
    g[:, 2, 2] = -g[:, 0, 0]-g[:, 1, 1]
    for col, i, j in [(5, 0, 1), (6, 0, 2), (7, 1, 2)]:
        g[:, i, j] = g[:, j, i] = v[:, col]*G0
    return b, g


def make_dataset(position):
    _, b1, g1 = evaluate(position, 1)
    _, b3, g3 = evaluate(position, 3)
    b, g = b1+b3, g1+g3
    return dict(position_m=position, X=np.column_stack((pack(b, g), np.full(len(b), BETA))),
                Y=pack(b1, g1), B_input_T=b, G_input_T_per_m=g,
                B_target_T=b1, G_target_T_per_m=g1)


def positions(seed):
    rng = np.random.default_rng(seed)
    random = rng.uniform(LOW, HIGH, size=(100000, 3))
    axes = [np.linspace(lo, hi, 11) for lo, hi in zip(LOW, HIGH)]
    axes[0][5] = axes[1][5] = 0.0
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    return dict(train=random[:80000], validation=random[80000:90000],
                test=random[90000:], diagnostic_grid=grid)


def relative(a, b):
    return float(np.max(np.linalg.norm((a-b).reshape(len(a), -1), axis=1) /
                        np.maximum(np.linalg.norm(b.reshape(len(b), -1), axis=1), 1e-30)))


def stats(values):
    return dict(zip(['min', 'median', 'p95', 'max'], map(float, np.quantile(values, [0, .5, .95, 1]))))


def verify_physics(points):
    report = {}
    step = 1e-6
    for degree in (1, 3):
        _, b, g = evaluate(points, degree)
        fd_b, fd_g = np.empty_like(b), np.empty_like(g)
        for j in range(3):
            delta = np.zeros(3); delta[j] = step
            vp, bp, _ = evaluate(points+delta, degree)
            vm, bm, _ = evaluate(points-delta, degree)
            fd_b[:, j] = -(vp-vm)/(2*step)
            fd_g[:, :, j] = (bp-bm)/(2*step)
        symmetry = relative(g, g.transpose(0, 2, 1))
        trace = float(np.max(np.abs(np.trace(g, axis1=1, axis2=2))/np.linalg.norm(g, axis=(1, 2))))
        euler = relative(np.einsum('nij,nj->ni', g, points), -(degree+2)*b)
        eb, eg = relative(fd_b, b), relative(fd_g, g)
        assert max(symmetry, trace, euler) < 1e-10
        assert max(eb, eg) < 1e-7
        report[str(degree)] = dict(symmetry=symmetry, trace=trace, euler=euler,
                                   finite_difference_B=eb, finite_difference_G=eg)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'data')
    parser.add_argument('--seed', type=int, default=4098)
    args = parser.parse_args()
    splits = positions(args.seed)
    repeated = positions(args.seed)
    assert all(np.array_equal(v, repeated[k]) for k, v in splits.items())
    random = np.concatenate([splits[k] for k in ('train', 'validation', 'test')])
    assert len(np.unique(random, axis=0)) == 100000
    physics = verify_physics(np.concatenate((random[:128], splits['diagnostic_grid'])))
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, pos in splits.items():
        dataset = make_dataset(pos)
        assert all(v.dtype == np.float64 and np.all(np.isfinite(v)) for v in dataset.values())
        assert np.all((pos >= LOW) & (pos <= HIGH))
        assert dataset['X'].shape == (len(pos), 9) and dataset['Y'].shape == (len(pos), 8)
        assert np.all(dataset['X'][:, -1] == BETA)
        regenerated = make_dataset(repeated[name])
        assert all(np.array_equal(v, regenerated[k]) for k, v in dataset.items())
        del regenerated
        _, b1, g1 = evaluate(pos, 1)
        _, b3, g3 = evaluate(pos, 3)
        assert np.array_equal(dataset['B_input_T'], b1+b3)
        assert np.array_equal(dataset['G_input_T_per_m'], g1+g3)
        assert np.array_equal(dataset['B_target_T'], b1)
        assert np.array_equal(dataset['G_target_T_per_m'], g1)
        for key, bkey, gkey in [('X', 'B_input_T', 'G_input_T_per_m'), ('Y', 'B_target_T', 'G_target_T_per_m')]:
            rb, rg = unpack(dataset[key])
            assert relative(rb, dataset[bkey]) < 1e-10
            assert relative(rg, dataset[gkey]) < 1e-10
        np.savez_compressed(args.output/f'{name}.npz', **dataset)
        with np.load(args.output/f'{name}.npz', allow_pickle=False) as loaded:
            assert all(np.array_equal(v, loaded[k]) for k, v in dataset.items())
        table = np.column_stack((pos, dataset['X'], dataset['Y']))
        header = ['position_x_m', 'position_y_m', 'position_z_m'] + ['input_'+s for s in COMPONENTS]+['input_beta']+['target_'+s for s in COMPONENTS]
        np.savetxt(args.output/f'{name}.csv', table, delimiter=',', header=','.join(header), comments='', fmt='%.17g')
        # Verify CSV column ordering, shape and double-precision round trip.
        restored = np.loadtxt(args.output/f'{name}.csv', delimiter=',', skiprows=1)
        assert np.array_equal(restored, table)
        radius = np.linalg.norm(pos, axis=1)
        summary[name] = dict(samples=len(pos), distance_m=stats(radius),
            outside_experiment_fraction=float(np.mean(radius > .253)),
            B_input_T=stats(np.linalg.norm(b1+b3, axis=1)),
            B_target_T=stats(np.linalg.norm(b1, axis=1)),
            G_input_T_per_m=stats(np.linalg.norm(g1+g3, axis=(1, 2))),
            G_target_T_per_m=stats(np.linalg.norm(g1, axis=(1, 2))),
            correction_B_relative=stats(np.linalg.norm(b3, axis=1)/np.linalg.norm(b1, axis=1)),
            correction_G_relative=stats(np.linalg.norm(g3, axis=(1, 2))/np.linalg.norm(g1, axis=(1, 2))))
    metadata = dict(schema_version=1, seed=args.seed, dtype='float64', numpy_version=np.__version__,
        source=dict(M1_A_m2=M1, M3_A_m4=M3, mu0_over_4pi=C,
            reference_document='D4098_联合扫描评估实验.md, sections 8 and 12',
            original_S0_center_m=[.007026319407456449, -.009362592228017384, .1479186758775004],
            original_S0_axis=[-.03162771224786257, .02232363526770045, .9992503906060788],
            original_transform_applied=False),
        frame='source center at origin, magnetic axis +Z, fixed sensor axes',
        bounds_m=dict(lower=LOW.tolist(), upper=HIGH.tolist()),
        normalization=dict(a_m=A, B0_T=B0, G0_T_per_m=G0, beta=BETA),
        X_columns=COMPONENTS+['beta'], Y_columns=COMPONENTS,
        npz_units=dict(position_m='m', X='dimensionless', Y='dimensionless', B_input_T='T',
            B_target_T='T', G_input_T_per_m='T/m', G_target_T_per_m='T/m'),
        gradient_convention='G[i,j] = dB_i/dx_j; Gzz = -Gxx-Gyy',
        sampling='PCG64 uniform volume; consecutive 80/10/10 split; no noise or rotations',
        correction_definition='norm(order-3)/norm(order-1); gradient uses Frobenius norm',
        limitation='Distances >0.253 m extrapolate the fitted analytic model; no experimental validation implied.',
        verification=dict(status='passed', finite_difference_step_m=1e-6, physics=physics,
            reproducible_arrays=True, unique_random_positions=True, npz_csv_roundtrip=True),
        statistics=summary)
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
    lines = ['# 数据检查摘要', '', '全部检查通过。重复生成比较数组内容，不要求压缩文件字节一致。', '',
             '| 集合 | 样本数 | 距离范围 m | 输入场强范围 mT | 三阶场修正中位数 | 三阶梯度修正中位数 |',
             '| --- | ---: | --- | --- | ---: | ---: |']
    for name, s in summary.items():
        lines.append(f"| {name} | {s['samples']} | {s['distance_m']['min']:.6f}–{s['distance_m']['max']:.6f} | {s['B_input_T']['min']*1e3:.6f}–{s['B_input_T']['max']*1e3:.6f} | {s['correction_B_relative']['median']:.2%} | {s['correction_G_relative']['median']:.2%} |")
    lines += ['', '修正幅度以一阶场/梯度为分母。完整 min/median/P95/max 及物理残差见 metadata.json。', '',
              '通过：解析场/梯度的中心差分检查；对称、零迹、欧拉关系；叠加与标签；归一化往返；有限值与范围；无重复随机位置；数组复现；NPZ/CSV 读取往返。']
    (args.output/'validation_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(args.output.resolve()), samples={k: len(v) for k,v in splits.items()}, physics=physics), indent=2))


if __name__ == '__main__':
    main()
