"""Controlled width ablation; hold data, seed, loss and optimizer fixed."""
import json
from pathlib import Path
import numpy as np
import torch
from model import DipoleNet, loss_parts
from evaluation import load_split, predict, metrics

ROOT = Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1)
    output = ROOT/'runs'/'capacity_comparison'
    output.mkdir(exist_ok=False)
    train, val = [load_split(ROOT/'data', s) for s in ('train', 'validation')]
    x, y = [torch.tensor(train[k], dtype=torch.float32) for k in ('X','Y')]
    results = []
    for width in (64,32,16,8):
        torch.manual_seed(4098)
        generator = torch.Generator().manual_seed(4098)
        model = DipoleNet(width)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        best, stale, history = float('inf'), 0, []
        path = output/f'width_{width}.pt'
        for epoch in range(1,501):
            model.train()
            for indices in torch.randperm(len(x), generator=generator).split(512):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_parts(model(x[indices]), y[indices])[0]
                loss.backward(); optimizer.step()
            vm = metrics(predict(model,val['X'],'cpu'),val['Y'],val['position_m'])
            history.append({'epoch':epoch,'validation_loss':vm['loss']})
            if vm['loss'] < best:
                best, stale, best_epoch = vm['loss'], 0, epoch
                torch.save({'model_state':model.state_dict(),'width':width,'epoch':epoch},path)
            else: stale += 1
            if stale >= 50: break
        model.load_state_dict(torch.load(path,weights_only=True)['model_state'])
        row = dict(width=width,parameters=sum(p.numel() for p in model.parameters()),best_epoch=best_epoch,epochs=epoch)
        for split in ('validation','test','diagnostic_grid'):
            d = load_split(ROOT/'data',split)
            row[split] = metrics(predict(model,d['X'],'cpu'),d['Y'],d['position_m'])
        results.append(row)
        (output/f'history_{width}.json').write_text(json.dumps(history,indent=2))
        (output/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(row),flush=True)
    lines=['# 网络参数缩减实验','', '同一数据划分、种子4098、三个SiLU隐藏层、Adam(lr=0.001)、batch=512；最多500轮，验证损失50轮无改善早停。各宽度重新训练，按验证集选择检查点，测试集仅最终评价。单种子结果，不代表统计显著性。','', '|隐藏层宽度|参数量|最佳轮次|测试场中位误差|测试梯度中位误差|测试梯度P95|','|---|---|---|---|---|---|']
    for r in results:
        m=r['test']; lines.append(f"|{r['width']}|{r['parameters']}|{r['best_epoch']}|{m['B_relative']['median']:.3%}|{m['G_relative']['median']:.3%}|{m['G_relative']['p95']:.3%}|")
    lines += ['', '数据是固定磁源、固定姿态的解析模拟数据，结论仅限同区域插值。原有runs/baseline未覆盖。本次64宽度为同一实验流程重训对照。', '', '完整指标、历史与权重保存在 runs/capacity_comparison（Git忽略）；复现：`.venv/bin/python compare_capacity.py`。输出目录必须不存在。']
    (ROOT/'网络参数缩减实验.md').write_text('\n'.join(lines)+'\n')

if __name__ == '__main__': main()
