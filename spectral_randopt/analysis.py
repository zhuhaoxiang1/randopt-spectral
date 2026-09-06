"""Small, dependency-light tables; uncertainty resamples antithetic pairs together."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def _load(path):
    return json.loads(Path(path).read_text())


def density_interval(records: list[dict], base_score: float, margin: float, seed: int) -> tuple[float, float, float]:
    strata = defaultdict(lambda: defaultdict(list))
    hits = []
    for record in records:
        c = record['candidate']
        hit = float(record['probe']['metrics']['accuracy'] >= base_score + margin)
        scale = c.get('sigma') if record.get('method', {}).get('kind') == 'original' else c['rho']
        strata[scale][c['pair_id']].append(hit)
        hits.append(hit)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(500):
        sample = []
        for groups in strata.values():
            clusters = list(groups.values())
            for i in rng.integers(0, len(clusters), size=len(clusters)):
                sample.extend(clusters[i])
        bootstrap.append(np.mean(sample))
    lo, hi = np.quantile(bootstrap, [0.025, 0.975])
    return float(np.mean(hits)), float(lo), float(hi)


def paired_accuracy_interval(left: list[dict], right: list[dict], seed: int) -> dict:
    if not left or [p['id'] for p in left] != [p['id'] for p in right]:
        raise ValueError('Paired comparison requires identical nonempty question IDs and order')
    differences = np.array([int(a['correct']) - int(b['correct']) for a, b in zip(left, right)])
    rng = np.random.default_rng(seed)
    samples = differences[rng.integers(0, len(left), size=(1000, len(left)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {'accuracy_delta': float(differences.mean()), 'delta_ci_low': float(low), 'delta_ci_high': float(high),
            'questions': len(left), 'left_only_correct': int((differences == 1).sum()),
            'right_only_correct': int((differences == -1).sum())}


def analyze(output: str | Path) -> Path:
    output = Path(output)
    manifest, base = _load(output / 'manifest.json'), _load(output / 'base.json')
    if not (output / 'complete.json').exists():
        raise ValueError('Analysis requires a completed run; use --resume to finish it')
    all_records, summary = {}, []
    flattened = []
    for method in manifest['config']['methods']:
        name = method['name']
        records = [_load(p) for p in sorted((output / 'candidates' / name).glob('*.json'))]
        if len(records) != manifest['config']['population']:
            raise ValueError(f'Incomplete candidate set: {name}')
        all_records[name] = records
        density, low, high = density_interval(records, base['probe']['metrics']['accuracy'],
                                             manifest['config']['margin'], manifest['config']['seed'])
        ensembles = _load(output / 'ensembles' / f'{name}.json')
        row = {'method': name, 'candidates': len(records), 'probe_density': density,
               'distance_scope': records[0]['perturbation'].get('distance_scope', 'selected_attention_heads'),
               'density_ci_low': low, 'density_ci_high': high,
               'mean_probe_accuracy': float(np.mean([r['probe']['metrics']['accuracy'] for r in records])),
               'mean_distance': float(np.mean([r['perturbation']['relative_distance'] for r in records])),
               'mean_reference_kl': float(np.mean([r['reference']['reference_kl'] for r in records])),
               'mean_reference_nll_delta': float(np.mean([r['reference']['reference_nll_delta'] for r in records])),
               'mean_probe_truncation': float(np.mean([r['probe']['metrics']['truncation_rate'] for r in records])),
               'sampling_seconds': sum(r['elapsed_seconds'] for r in records),
               'sampling_generated_tokens': sum(r[s]['metrics']['generated_tokens'] for r in records
                                                 for s in ('selection', 'probe'))}
        test_records = [_load(p) for p in sorted((output / 'test' / name).glob('*.json'))]
        row['test_generated_tokens'] = sum(r['metrics']['generated_tokens'] for r in test_records)
        row['test_seconds'] = sum(r.get('total_elapsed_seconds', r.get('elapsed_seconds', 0.)) for r in test_records)
        for ensemble in ensembles:
            row[f'test_top{ensemble["k"]}'] = ensemble['accuracy']
        summary.append(row)
        for r in records:
            flattened.append({'method': name, **r['candidate'],
                              'selection_accuracy': r['selection']['metrics']['accuracy'],
                              'probe_accuracy': r['probe']['metrics']['accuracy'],
                              'probe_strict_accuracy': r['probe']['metrics']['strict_accuracy'],
                              'probe_truncation_rate': r['probe']['metrics']['truncation_rate'],
                              **r['reference'],
                              'relative_distance': r['perturbation']['relative_distance'],
                              'distance_scope': row['distance_scope'],
                              'elapsed_seconds': r['elapsed_seconds']})
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output / 'candidates.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in flattened))
    lines = ['# 谱扰动实验报告', '', f'模型：`{manifest["model_id"]}`', '',
             '**离线随机小模型 smoke：以下数值只验证执行链路，不代表 1.5B 的任务性能。**' if manifest['smoke_only']
             else '真实预训练模型实验；候选只根据选择集排序，命中率由独立探测集计算。', '',
             f'基础准确率：选择集 {base["selection"]["metrics"]["accuracy"]:.4f}；'
             f'探测集 {base["probe"]["metrics"]["accuracy"]:.4f}；测试集 {base["test"]["metrics"]["accuracy"]:.4f}。', '',
             '| 方法 | 探测集命中率 | 平均探测准确率 | 实际相对距离 | 参考 KL | 测试集 Top-1 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f'| {row["method"]} | {row["probe_density"]:.4f} | {row["mean_probe_accuracy"]:.4f} | '
                     f'{row["mean_distance"]:.6f} | {row["mean_reference_kl"]:.6f} | {row.get("test_top1", float("nan")):.4f} |')
    comparison = []
    baselines = [m['name'] for m in manifest['config']['methods'] if m['kind'] in {'original', 'gaussian'}]
    for baseline in baselines:
        baseline_ensembles = {e['k']: e for e in _load(output / 'ensembles' / f'{baseline}.json')}
        for method in manifest['config']['methods']:
            if method['name'] == baseline:
                continue
            for ensemble in _load(output / 'ensembles' / f'{method["name"]}.json'):
                comparison.append({'method': method['name'], 'baseline': baseline, 'k': ensemble['k'],
                                   **paired_accuracy_interval(ensemble['predictions'],
                                       baseline_ensembles[ensemble['k']]['predictions'], manifest['config']['seed'])})
    if comparison:
        with (output / 'fair_comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
        lines += ['', '## 同 K 公平比较', '',
                  '差值均为 method − baseline；配对题目 bootstrap 区间条件于这次选中的模型，不包含搜索种子波动。', '',
                  '| 方法 | 对照 | K | 准确率差 | 95% 配对区间 |', '|---|---|---:|---:|---|']
        for row in comparison:
            lines.append(f'| {row["method"]} | {row["baseline"]} | {row["k"]} | {row["accuracy_delta"]:+.4f} | '
                         f'[{row["delta_ci_low"]:+.4f}, {row["delta_ci_high"]:+.4f}] |')
    lines += ['', '## 配对交叉重构', '']
    gaussian_names = [m['name'] for m in manifest['config']['methods'] if m['kind'] == 'gaussian']
    if gaussian_names:
        source = {r['candidate']['id']: r for r in all_records[gaussian_names[0]]}
        pair_rows = []
        baseline = base['probe']['metrics']['accuracy']
        for m in manifest['config']['methods']:
            if m['kind'] not in {'direction', 'values'}:
                continue
            for r in all_records[m['name']]:
                g = source[r['candidate']['id']]
                g_score, h_score = g['probe']['metrics']['accuracy'], r['probe']['metrics']['accuracy']
                pair_rows.append({'candidate': r['candidate']['id'], 'method': m['name'],
                                  'gaussian_improved': g_score >= baseline + manifest['config']['margin'],
                                  'gaussian_gain': g_score - baseline, 'hybrid_gain': h_score - baseline,
                                  'hybrid_minus_gaussian': h_score - g_score})
        (output / 'hybrid_pairs.json').write_text(json.dumps(pair_rows, indent=2) + '\n')
        for m in manifest['config']['methods']:
            if m['kind'] not in {'direction', 'values'}:
                continue
            for improved in (True, False):
                rows = [r for r in pair_rows if r['method'] == m['name'] and r['gaussian_improved'] == improved]
                if rows:
                    gain = np.mean([r['hybrid_gain'] for r in rows])
                    lines.append(f'- {m["name"]}，高斯候选{"命中" if improved else "未命中"}组：'
                                 f'{len(rows)} 个候选，重构后相对基础模型的平均准确率变化 {gain:+.4f}。')
    lines += ['', '## 解释边界', '',
              '- original_randopt 使用全部文本参数的原始噪声规则，适配到共同 Transformers 后端；不是 vLLM 位级复现。',
              '- 原始基线的 sigma 是逐元素绝对噪声标准差；其他方法的 rho 是每个目标头的相对距离，两者不能直接等同。',
              '- 全参数距离与注意力头距离分母不同；请按 summary.csv 的 distance_scope 分开解释。',
              '- 共同 N/K、题目与生成上限控制评估预算；生成 token 数、SVD 成本和实测耗时仍须分别报告。',
              '- 命中率按指定 margin 定义；置信区间在每个半径内以正负候选对为簇重采样，条件于固定探测集。',
              '- 区间只反映种子抽样波动，不包含题目抽样误差；全零/全一命中时 bootstrap 会退化，不能据此认定真实概率等于 0/1。',
              '- 交叉重构不重新校准距离；请结合其实际距离和参考损失解释收益。',
              '- 精确保谱指构造时的单个头矩阵；低精度转换和拼接后的整层谱另当别论。',
              '- 谱带是索引分位分组，非自动发现的谱峰；详细子空间与重构诊断见候选文件。',
              '- Top-K 结果和错误互补指标见 ensembles；本实验仅有 GSM8K，不声称测得跨任务专家多样性。',
              '- 参考文本只做局部 KL/NLL 诊断，不能据此认定通用能力没有遗忘。',
              '- 生成截断率、严格格式准确率和完整候选指标见 candidates.jsonl；方法优劣需真实实验确认。', '']
    report = output / 'report.md'
    report.write_text('\n'.join(lines))
    return report
