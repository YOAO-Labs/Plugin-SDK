"""Evaluate mask→polygon fidelity on an explicitly supplied PNG fixture directory.

Private dataset-derived fixtures and reports must stay outside the public repository.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from yoao_plugin_sdk.inference.mask_utils import mask_to_multi_polygon, mask_to_polygon


def _load_mask(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return (img > 0).astype(np.uint8)


def _polygon_iou(poly_pixel_coords: list[list[float]], mask: np.ndarray) -> float:
    """单 polygon (像素坐标) 重栅格化 → 与原 mask 的 IoU."""
    if not poly_pixel_coords:
        return 0.0
    rast = np.zeros_like(mask)
    pts = np.array(poly_pixel_coords, dtype=np.int32)
    cv2.fillPoly(rast, [pts], 1)
    inter = np.logical_and(rast, mask).sum()
    union = np.logical_or(rast, mask).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _multi_polygon_iou(rings: list[dict], mask: np.ndarray) -> float:
    """将含内外环的多边形重栅格化，计算与原 mask 的 IoU。

    将各外环填为 1、各空洞填为 0，以评测多连通域和空洞的保真度。
    """
    if not rings:
        return 0.0
    rast = np.zeros_like(mask)
    for r in rings:
        ext_pts = np.array(r["exterior"], dtype=np.int32)
        cv2.fillPoly(rast, [ext_pts], 1)
        for hole in r.get("holes", []):
            hpts = np.array(hole, dtype=np.int32)
            cv2.fillPoly(rast, [hpts], 0)
    inter = np.logical_and(rast, mask).sum()
    union = np.logical_or(rast, mask).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _summary(values: list[float], precision: int = 3) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0}
    sorted_vals = sorted(values)
    p95_idx = max(0, int(0.95 * (len(sorted_vals) - 1)))
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), precision),
        "median": round(statistics.median(values), precision),
        "p95": round(sorted_vals[p95_idx], precision),
        "min": round(min(values), precision),
    }


def _eval_dir(
    masks_dir: Path, tolerances: list[float]
) -> tuple[list[dict], dict[float, dict]]:
    rows: list[dict] = []
    per_tol_iou: dict[float, list[float]] = {t: [] for t in tolerances}
    per_tol_verts: dict[float, list[int]] = {t: [] for t in tolerances}
    # 使用相同的容差比较单外环和多连通域两种转换。
    per_tol_iou_multi: dict[float, list[float]] = {t: [] for t in tolerances}
    per_tol_verts_multi: dict[float, list[int]] = {t: [] for t in tolerances}
    per_tol_rings_multi: dict[float, list[int]] = {t: [] for t in tolerances}

    files = sorted(p for p in masks_dir.glob("*.png"))
    if not files:
        raise SystemExit(f"no *.png masks found under {masks_dir}")

    for png in files:
        mask = _load_mask(png)
        if mask is None or mask.sum() == 0:
            print(f"[skip] {png.name}: empty / unreadable", file=sys.stderr)
            continue
        h, w = mask.shape
        row: dict = {"file": png.name, "size": f"{w}x{h}", "area": int(mask.sum())}
        for tol in tolerances:
            poly_pixel = mask_to_polygon(mask, tolerance=tol, normalize_to=None)
            iou = _polygon_iou(poly_pixel, mask)
            n_verts = len(poly_pixel)
            row[f"iou@{tol}"] = round(iou, 3)
            row[f"verts@{tol}"] = n_verts
            per_tol_iou[tol].append(iou)
            per_tol_verts[tol].append(n_verts)

            rings = mask_to_multi_polygon(mask, tolerance=tol, normalize_to=None)
            iou_m = _multi_polygon_iou(rings, mask)
            verts_m = sum(
                len(r["exterior"]) + sum(len(h) for h in r["holes"]) for r in rings
            )
            row[f"iou_multi@{tol}"] = round(iou_m, 3)
            row[f"verts_multi@{tol}"] = verts_m
            row[f"rings@{tol}"] = len(rings)
            row[f"iou_diff@{tol}"] = round(iou_m - iou, 3)
            per_tol_iou_multi[tol].append(iou_m)
            per_tol_verts_multi[tol].append(verts_m)
            per_tol_rings_multi[tol].append(len(rings))
        rows.append(row)

    summary: dict[float, dict] = {}
    for tol in tolerances:
        summary[tol] = {
            "iou": _summary(per_tol_iou[tol]),
            "verts": _summary([float(v) for v in per_tol_verts[tol]], precision=1),
            "iou>=0.95_pct": (
                round(
                    100.0
                    * sum(1 for v in per_tol_iou[tol] if v >= 0.95)
                    / max(1, len(per_tol_iou[tol])),
                    1,
                )
            ),
            "iou_multi": _summary(per_tol_iou_multi[tol]),
            "verts_multi": _summary(
                [float(v) for v in per_tol_verts_multi[tol]], precision=1
            ),
            "iou_multi>=0.95_pct": (
                round(
                    100.0
                    * sum(1 for v in per_tol_iou_multi[tol] if v >= 0.95)
                    / max(1, len(per_tol_iou_multi[tol])),
                    1,
                )
            ),
            "rings_multi": _summary(
                [float(v) for v in per_tol_rings_multi[tol]], precision=1
            ),
            "multi_only_helps_pct": (
                round(
                    100.0
                    * sum(
                        1
                        for s, m in zip(per_tol_iou[tol], per_tol_iou_multi[tol])
                        if m - s >= 0.02
                    )
                    / max(1, len(per_tol_iou[tol])),
                    1,
                )
            ),
        }
    return rows, summary


def _render_markdown(
    rows: list[dict],
    summary: dict[float, dict],
    tolerances: list[float],
    masks_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("# 掩码转多边形：简化容差与保真度评测")
    lines.append("")
    lines.append(
        "> `scripts/eval_simplify.py` 自动生成于 "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}; 输入 `{masks_dir}` ({len(rows)} samples)。"
    )
    lines.append("")
    lines.append("## 数据说明")
    lines.append("")
    sample_names = [r["file"] for r in rows]
    n_synth = sum(1 for n in sample_names if n.startswith("synthetic_"))
    n_other = len(sample_names) - n_synth
    lines.append(
        f"按文件名分类：**{n_synth} 张以 `synthetic_` 开头，{n_other} 张其他样本**。"
        "合成样本与实际掩码的边界复杂度可能不同；更换样本后应重新评测。"
    )
    lines.append("")
    lines.append("## `mask_to_polygon`：按容差汇总最大外环的结果")
    lines.append("")
    lines.append(
        "| tolerance (px) | n | IoU mean | IoU median | IoU p95 | IoU≥0.95 % | verts mean | verts median | verts p95 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for tol in tolerances:
        s = summary[tol]
        lines.append(
            f"| {tol} | {s['iou']['n']} | {s['iou']['mean']} | {s['iou']['median']} | "
            f"{s['iou']['p95']} | {s['iou>=0.95_pct']} | {s['verts']['mean']} | "
            f"{s['verts']['median']} | {s['verts']['p95']} |"
        )
    lines.append("")
    lines.append("## `mask_to_multi_polygon`：按容差汇总多连通域与空洞的结果")
    lines.append("")
    lines.append(
        "`mask_to_multi_polygon` (`RETR_CCOMP`) 与 `mask_to_polygon` "
        "(`RETR_EXTERNAL`，只取最大外环) 使用相同的容差。`outer rings median` 是外环数量的中位数；"
        "`multi_only_helps %` 是前者的 IoU 比后者高至少 0.02 的样本占比。"
    )
    lines.append("")
    lines.append(
        "| tolerance | IoU mean | IoU median | IoU≥0.95 % | outer rings median | multi_only_helps % |"
    )
    lines.append("|---|---|---|---|---|---|")
    for tol in tolerances:
        s = summary[tol]
        lines.append(
            f"| {tol} | {s['iou_multi']['mean']} | {s['iou_multi']['median']} | "
            f"{s['iou_multi>=0.95_pct']} | {s['rings_multi']['median']} | "
            f"{s['multi_only_helps_pct']} |"
        )
    lines.append("")
    lines.append("## `mask_to_polygon` 的简化容差分析")
    lines.append("")
    # IoU≥0.95 占比按 tolerance 看; 取 1.0 档作主参考 (默认值).
    primary_tol = 1.0 if 1.0 in summary else tolerances[0]
    iou95 = summary[primary_tol]["iou>=0.95_pct"]
    iou_mean = summary[primary_tol]["iou"]["mean"]
    verts_med = summary[primary_tol]["verts"]["median"]
    # 四档判定:
    # ① ≥95%: 完全达标
    # ② 85% ≤ x < 95% 且 IoU mean ≥ 0.95: 接近验收线
    # ③ 其余未达标样本，IoU spread < 0.05: 调整容差的收益有限
    # ④ 其余未达标样本，IoU spread ≥ 0.05: 推荐达标率最高的容差
    if iou95 >= 95.0:
        lines.append(
            "**`mask_to_polygon` 在当前样本满足验收线** "
            f"（tolerance={primary_tol} IoU≥0.95 占比 {iou95}%、IoU mean {iou_mean}、"
            f"顶点数中位 {verts_med}）。"
            "支持 `Context.simplify_tolerance` 的推理插件允许在单次请求中覆盖容差；"
            "更换样本后，若 IoU < 0.95 的占比超过 5%，应重新评估所用容差。"
        )
    elif iou95 >= 85.0 and iou_mean >= 0.95:
        verts_warn = (
            ""
            if verts_med <= 200
            else "（**注意**：顶点数中位数超过推理插件的 200 顶点告警阈值）"
        )
        lines.append(
            "**`mask_to_polygon` 在当前样本接近验收线但未达标** "
            f"（tolerance={primary_tol} IoU≥0.95 占比 {iou95}% < 95%、但 IoU mean {iou_mean} ≥ 0.95、"
            f"顶点数中位 {verts_med}）{verts_warn}。"
        )
        lines.append("")
        lines.append(
            "**低 IoU 样本分析**：`mask_to_polygon` 仅取最大连通域，"
            "且 `cv2.RETR_EXTERNAL` 不保留空洞。对于含多个连通域或空洞的样本，"
            "可在同一容差下对照 `mask_to_multi_polygon` 的结果，再判断是否要调整容差。"
        )
    else:
        lines.append(
            f"**⚠️ 当前样本 tolerance={primary_tol} IoU≥0.95 占比仅 {iou95}%（< 95% 验收线），"
            f"IoU mean {iou_mean}、顶点数中位 {verts_med}**。"
        )
        lines.append("")
        # 比较各容差的平均 IoU，判断调节容差是否有明显收益。
        iou_means = [summary[t]["iou"]["mean"] for t in tolerances]
        iou_spread = max(iou_means) - min(iou_means)
        if iou_spread < 0.05:
            lines.append(
                f"**调整容差的收益有限**：测试的容差范围 ({min(tolerances)}→{max(tolerances)}) 内，"
                f"IoU mean 的最大差异仅 {iou_spread:.3f}。可检查以下原因："
            )
            lines.append("")
            lines.append(
                "- `mask_to_polygon` 只取面积最大的连通域，其他片段不会出现在重栅格化结果中。"
            )
            lines.append(
                "- `cv2.RETR_EXTERNAL` 不保留内部空洞，重栅格化可能因填满空洞而降低 IoU。"
            )
            lines.append(
                "- 像素级边界噪声也会影响保真度，可单独评估形态学闭运算预处理的效果。"
            )
            lines.append("")
            verts_warn = (
                "（顶点数中位数超过推理插件的 200 顶点告警阈值）"
                if verts_med > 200
                else ""
            )
            lines.append(
                f"**不建议仅为提升 IoU 而调整容差**{verts_warn}。"
                "需要保留多个连通域或空洞时，对照 `mask_to_multi_polygon` 的 IoU；"
                "需要减少顶点数时，按表中的 IoU 和顶点数选择容差。"
            )
        else:
            # IoU 在 tolerance 间有显著差异 → 调 tolerance 有意义, 推荐最优档
            best_tol = max(tolerances, key=lambda t: summary[t]["iou>=0.95_pct"])
            lines.append(
                f"**建议在已测试的容差中为 `mask_to_polygon` 选择 {best_tol}** "
                f"（该档 IoU≥0.95 占比 {summary[best_tol]['iou>=0.95_pct']}% 最优）。"
            )
    lines.append("")
    lines.append("## `mask_to_polygon` 逐样本明细")
    lines.append("")
    headers = ["file", "size", "area"]
    for tol in tolerances:
        headers.append(f"iou@{tol}")
        headers.append(f"verts@{tol}")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = [str(row.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 复现")
    lines.append("")
    lines.append("```bash")
    lines.append(
        f"python packages/python/scripts/eval_simplify.py \\\n"
        f"    --masks-dir {masks_dir} \\\n"
        f"    --tolerances {','.join(str(t) for t in tolerances)} \\\n"
        "    --out /private/mask-evaluation.md"
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate mask→polygon simplify tolerance vs IoU + vertex count"
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        required=True,
        help="目录, 内含二值 mask png (单通道, 0=背景, 非零=前景)",
    )
    parser.add_argument(
        "--tolerances",
        type=str,
        default="0.5,1.0,2.0,3.0,5.0",
        help="逗号分隔的 tolerance 像素值列表",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="输出 markdown 报告路径; 设为 - 输出到 stdout",
    )
    args = parser.parse_args()

    if not args.masks_dir.is_dir():
        print(f"--masks-dir not a directory: {args.masks_dir}", file=sys.stderr)
        return 2

    tolerances = [float(t.strip()) for t in args.tolerances.split(",") if t.strip()]
    if not tolerances:
        print("--tolerances empty", file=sys.stderr)
        return 2

    rows, summary = _eval_dir(args.masks_dir, tolerances)
    md = _render_markdown(rows, summary, tolerances, args.masks_dir)

    if str(args.out) == "-":
        sys.stdout.write(md)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(
            f"✓ wrote {args.out} ({len(rows)} samples × {len(tolerances)} tolerances)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
