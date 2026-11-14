"""Report chapters 3/4 noise propagation and evaluation; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import DipoleNet
from generate import B0, G0
from evaluation import load_split, predict, metrics, errors, effect_plot
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'传感器噪声与误差建模实验报告_20261107.md'
LABELS={'single_sample':'单次采样','mean_1s_on_off_white':'理想1秒开关均值',
        'mean_1s_on_off_rms_sensitivity':'均值RMS比例敏感性情景'}


def table(text,start,end):
    rows=[]
    for line in text.split(start,1)[1].split(end,1)[0].splitlines():
        cells=[s.strip() for s in line.strip().strip('|').split('|')]
        if len(cells)==4 and cells[0].isdigit(): rows.append([float(v) for v in cells])
    a=np.asarray(rows)
    if a.shape!=(12,4) or not np.array_equal(a[:,0],np.arange(1,13)):
        raise ValueError('Expected sensors 1..12 and three axes')
    return a[:,1:]


def noise_model():
    text=SOURCE.read_text()
    sigma=table(text,'### 4.2 ','### 4.3 ')*1e-6
    ratio=table(text,'### 4.4 ','## 5.')
    pos=np.array([[-1,1,-2],[-1,1,0],[-1,1,2],[-1,-1,-2],[-1,-1,0],[-1,-1,2],
                  [1,1,-2],[1,1,0],[1,1,2],[1,-1,-2],[1,-1,0],[1,-1,2]])*1e-3
    design=np.zeros((36,8)) # unknown order Bx By Bz Gxx Gyy Gxy Gxz Gyz
    for i,(x,y,z) in enumerate(pos):
        design[3*i:3*i+3,:3]=np.eye(3)
        design[3*i,3:]=[x,0,y,z,0]
        design[3*i+1,3:]=[0,y,x,0,z]
        design[3*i+2,3:]=[-z,-z,0,x,y]
    op=np.linalg.pinv(design)
    np.testing.assert_allclose(op@design,np.eye(8),atol=1e-10)
    factor=np.sqrt(1/468+1/477) # +5V and 0V first-second complete rounds, report 4.3
    scales={'single_sample':np.ones((12,3)),
            'mean_1s_on_off_white':np.full((12,3),factor),
            'mean_1s_on_off_rms_sensitivity':factor*ratio}
    return sigma,ratio,pos,design,op,scales


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'runs/baseline/best.pt')
    p.add_argument('--output',type=Path,default=ROOT/'runs/baseline/noise_evaluation')
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--seed',type=int,default=20261114)
    args=p.parse_args()
    if args.repeats<2:p.error('At least two repetitions required')
    if args.output.exists() and any(args.output.iterdir()):p.error('Output must be empty')
    before=sha(args.checkpoint)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    for k,v in [('B0_T',B0),('G0_T_per_m',G0)]:
        if not np.isclose(ck['normalization'][k],v,rtol=1e-12,atol=0):raise ValueError('Incompatible normalization')
    model=DipoleNet();model.load_state_dict(ck['model_state']);model.eval()
    sigma,ratio,pos,design,op,scales=noise_model()
    cov={k:(op*(sigma*s).reshape(-1)**2)@op.T for k,s in scales.items()}
    for v in cov.values():assert np.linalg.eigvalsh(v).min()>0
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'noise_model.npz',sensor_sigma_T=sigma,rms_ratios=ratio,
                        sensor_position_m=pos,design=design,operator=op,**cov)
    result={}
    for split in ['test','diagnostic_grid']:
        data=load_split(ROOT/'data',split);x,y,position=data['X'],data['Y'],data['position_m']
        clean=predict(model,x,torch.device('cpu'))
        result[split]={'clean':{'network':metrics(clean,y,position),
                                'uncorrected':metrics(x[:,:8],y,position)}}
        for name,scale in scales.items():
            rng=np.random.default_rng(args.seed+(split!='test'))
            stored={m:{k:[] for k in ['B_relative','G_relative','Euler_relative']} for m in ['network','uncorrected']}
            repeats=[]
            for repeat in range(args.repeats):
                sensor=rng.normal(size=(len(x),36))*(sigma*scale).reshape(-1)
                delta=sensor@op.T
                noisy=x.copy();noisy[:,:3]+=delta[:,:3]/B0;noisy[:,3:8]+=delta[:,3:]/G0
                assert np.array_equal(noisy[:,-1],x[:,-1])
                pred=predict(model,noisy,torch.device('cpu'));row={}
                for method,values in [('network',pred),('uncorrected',noisy[:,:8])]:
                    row[method]=metrics(values,y,position)
                    for key,v in errors(values,y,position).items():stored[method][key].append(v)
                repeats.append(row)
                if repeat==0:
                    np.savez_compressed(args.output/f'{split}_{name}_example.npz',X_noisy=noisy,
                                        prediction=pred,target=y,position_m=position)
                    if split=='test':effect_plot(pred,{**data,'X':noisy},args.output/f'{name}_effects.png',f'{name}: realization 1')
            summary={'repeat_metrics':repeats,'distance_bins':[]}
            for method,quantities in stored.items():
                summary[method]={}
                for key,arrays in quantities.items():
                    values=np.concatenate(arrays);medians=[np.median(a) for a in arrays]
                    summary[method][key]={'median':float(np.median(values)),
                        'p95':float(np.quantile(values,.95)),'mean':float(values.mean()),
                        'repeat_median_range':[float(min(medians)),float(max(medians))]}
                for key in ['loss','B_NRMSE','G_NRMSE']:
                    summary[method][key+'_repeat_mean']=float(np.mean([r[method][key] for r in repeats]))
            summary['improved_fraction']={k:float(np.mean(np.concatenate(stored['network'][k])<np.concatenate(stored['uncorrected'][k]))) for k in stored['network']}
            radius=np.linalg.norm(position,axis=1)
            for lo,hi in [(.15,.2),(.2,.3),(.3,.4),(.4,.5)]:
                mask=(radius>=lo)&(radius<hi);row={'range_m':[lo,hi],'positions':int(mask.sum())}
                for method in stored:
                    row[method]={k:{'median':float(np.median(np.concatenate([a[mask] for a in arrays]))),
                        'p95':float(np.quantile(np.concatenate([a[mask] for a in arrays]),.95))}
                        for k,arrays in stored[method].items()}
                summary['distance_bins'].append(row)
            result[split][name]=summary
            print(split,name,'network B/G medians',summary['network']['B_relative']['median'],summary['network']['G_relative']['median'],flush=True)
    assert sha(args.checkpoint)==before
    payload={'seed':args.seed,'repeats':args.repeats,'checkpoint_sha256':before,'source_sha256':sha(SOURCE),
        'definitions':{'single_sample':'Single snapshot, no background subtraction.',
          'mean_1s_on_off_white':'Independent on/off means N=468 and 477; variance factor 1/468+1/477.',
          'mean_1s_on_off_rms_sensitivity':'Multiply each sensor-axis mean standard deviation by report 4.4 RMS ratio; sensitivity only.'},
        'assumptions':['Diagonal 36D covariance from rounded report 4.2 standard deviations; full covariance unavailable locally.',
          'Zero-mean independent Gaussian draws are modeling assumptions, not experimentally established.',
          'Corrected array axes aligned with fixed synthetic source axes; 12 sensors, 2x2x4 mm array.',
          'Joint B/G propagation op Sigma op.T; no independent gradient-noise prescription.',
          'Constant spatial noise; no drift, calibration bias, curvature or geometry errors.',
          'Clean labels and unchanged checkpoint; no training.',
          'Pooled quantiles over repeated noise realizations of the same positions, not independent spatial samples.',
          'No statistics from report chapters 5/6.'],
        'std_BG_SI':{k:np.sqrt(np.diag(v)).tolist() for k,v in cov.items()},'results':result}
    (args.output/'results.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    plot_comparison(result,args.output)
    append_report(payload,args.output)


def plot_comparison(result,output):
    fig,axes=plt.subplots(1,3,figsize=(14,4.5));names=list(LABELS)
    for ax,key,title in zip(axes,['B_relative','G_relative','Euler_relative'],['Field error (%)','Gradient error (%)','Euler residual (%)']):
        for method,color,offset in [('uncorrected','tab:blue',-.17),('network','tab:orange',.17)]:
            ax.bar(np.arange(3)+offset,[100*result['test'][n][method][key]['median'] for n in names],width=.34,color=color,label=method)
            ax.scatter(np.arange(3)+offset,[100*result['test'][n][method][key]['p95'] for n in names],color=color,marker='_',s=180)
        ax.axhline(100*result['test']['clean']['network'][key]['median'],color='gray',ls='--',label='Clean network median')
        ax.set_xticks(range(3),['Single sample','1s on/off white','1s RMS sensitivity'],rotation=18,ha='right')
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle(f"{len(result['test']['single_sample']['repeat_metrics'])} noise realizations: bars = median, marks = P95")
    fig.tight_layout();fig.savefig(output/'noise_comparison.png',dpi=150);plt.close(fig)


def append_report(payload,output):
    path=output.parent/'首次训练评估.md'
    if not path.exists():return
    marker='\n## 测量噪声鲁棒性补充评估'
    original=path.read_text().split(marker)[0];results=payload['results']
    lines=[marker,'','评估日期：2026-11-14。仅对输入加噪，沿用第236轮最佳模型，不重新训练，理想标签保持不变。','',
      '### 模型来源与假设','',
      '采用噪声报告第3、4章。第4.2节12颗传感器36分量单次标准差5.535～5.844 µT。完整36×36协方差在当前工作区不可用，因此采用表格边际方差的对角近似；高斯、时间独立及跨空间恒定协方差均为模拟假设。原报告的跨传感器相关系数绝对值中位数0.043、最大0.078，不能据此确定全部交叉协方差。', '',
      '使用D4098记录的12颗2×2×4 mm阵列，在已校正阵列坐标与合成源坐标对齐的假设下，通过局部模型 b_i=B+G d_i 的对称无迹最小二乘算子L传播：δ[B,g]=Lε，联合协方差LΣLᵀ。B与G来自同一组传感器噪声，保留传播后交叉协方差；不重复应用T/Q、不把µT直接当作梯度单位。不叠加有限孔径偏差或校准系统误差。','',
      '情景1：单次完整阵列快照，不扣背景。情景2：1秒通电均值减关断均值，采用第4.3节+5V/0V首秒完整轮数468/477，独立白噪声方差因子为1/468+1/477。情景3：在情景2上逐轴乘第4.4节均值差RMS比例；该比例不是通用噪声放大系数，此项仅为敏感性假设，不代表已识别的时间相关模型。','',
      f'每个位置{payload["repeats"]}次噪声实现，种子{payload["seed"]}；情景间共用标准正态随机数。测试集10000位置、诊断网格1331位置；统计为池化中位数/P95，重复实现不是新增独立空间样本。逐次中位数范围保存于JSON，不是置信区间。','',
      '| 情景 | 场中位数/P95：不修正 | 场中位数/P95：网络 | 梯度中位数/P95：不修正 | 梯度中位数/P95：网络 |',
      '| --- | ---: | ---: | ---: | ---: |']
    for name,label in LABELS.items():
        s=results['test'][name]
        def cell(method,key):return f'{s[method][key]["median"]:.3%} / {s[method][key]["p95"]:.3%}'
        lines.append(f'| {label} | {cell("uncorrected","B_relative")} | {cell("network","B_relative")} | {cell("uncorrected","G_relative")} | {cell("network","G_relative")} |')
    lines+=['','### 传播后的噪声尺度','','顺序为 Bx、By、Bz、Gxx、Gyy、Gxy、Gxz、Gyz。','',
      '| 情景 | 三个场分量标准差 µT | 五个梯度分量标准差 mT/m |','| --- | --- | --- |']
    for name,label in LABELS.items():
        sd=np.array(payload['std_BG_SI'][name]);fmt=lambda a:', '.join(f'{v:.4f}' for v in a)
        lines.append(f'| {label} | {fmt(sd[:3]*1e6)} | {fmt(sd[3:]*1e3)} |')
    lines+=['','### 距离分段','','| 情景 | 距离 cm | 网络场中位数 | 网络梯度中位数/P95 | 不修正梯度中位数 |','| --- | --- | ---: | ---: | ---: |']
    for name,label in LABELS.items():
        for row in results['test'][name]['distance_bins']:
            lo,hi=row['range_m'];n=row['network'];u=row['uncorrected']
            lines.append(f'| {label} | {lo*100:.0f}–{hi*100:.0f} | {n["B_relative"]["median"]:.3%} | {n["G_relative"]["median"]:.3%} / {n["G_relative"]["p95"]:.3%} | {u["G_relative"]["median"]:.3%} |')
    lines+=['','### 欧拉关系与边界网格','','| 情景 | 测试网络欧拉中位数/P95 | 测试不修正欧拉中位数/P95 | 网格网络梯度中位数/P95 |','| --- | ---: | ---: | ---: |']
    for name,label in LABELS.items():
        s=results['test'][name];n=s['network']['Euler_relative'];u=s['uncorrected']['Euler_relative'];g=results['diagnostic_grid'][name]['network']['G_relative']
        lines.append(f'| {label} | {n["median"]:.3%} / {n["p95"]:.3%} | {u["median"]:.3%} / {u["p95"]:.3%} | {g["median"]:.3%} / {g["p95"]:.3%} |')
    lines+=['','本评估不包含慢漂移、绝对零偏、增益误差、场强依赖或几何误差；未使用第5/6章FDM有效误差。固定源参数、固定姿态的强先验可能使输出对某些输入扰动不敏感，不能将其推广为通用传感器去噪能力或实测定位保证。', '',
      '复现入口：evaluate_noise.py。noise_evaluation/results.json记录参数来源哈希、全部逐次指标、改善比例、距离分段；noise_model.npz记录标准差、几何、算子及联合协方差。各情景保存首组加噪输入和预测。检查点运行前后哈希一致。','',
      '![噪声情景比较](noise_evaluation/noise_comparison.png)']
    path.write_text(original+'\n'.join(lines)+'\n')


if __name__=='__main__':main()
