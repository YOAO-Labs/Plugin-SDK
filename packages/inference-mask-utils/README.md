# anno-mask-utils

共享的 mask → polygon 工具包。被以下 backend 复用：

- 平台的 grounded-sam2 和 sam3 backend。

## 安装（开发模式）

```bash
cd packages/inference-mask-utils
pip install -e ".[test]"
pytest -v
```

RLE 黄金用例 `tests/fixtures/rasterMaskRle.json` 是平台源码 `apps/web/src/__fixtures__/rasterMaskRle.json` 在提交 `0a21bc402432aa93a80fcb9f12d2f532e7109635` 的冻结一致性快照，SHA256 为 `f1dbba03cfb982dd0d747ceb424da267f63fc0132109c4025508cbfe99453f0b`。平台源码仍是唯一权威来源；不得在本仓库静默修改或独立维护这些用例。


## 用法

```python
import numpy as np
from mask_utils import mask_to_polygon

mask = np.zeros((512, 512), dtype=np.uint8)
mask[100:300, 150:400] = 1

# 像素坐标
poly = mask_to_polygon(mask, tolerance=1.0)

# 归一化到 [0, 1]（Label Studio polygonlabels 风格）
poly_norm = mask_to_polygon(mask, tolerance=1.0, normalize_to=(512, 512))
```

## API

### `mask_to_polygon(mask, tolerance=1.0, normalize_to=None) -> list[list[float]]`

把二值 mask `(H, W)` 转成简化后的多边形顶点列表。

- `mask`：`np.ndarray`（uint8/bool），非零像素视为前景
- `tolerance`：`shapely.simplify` 容差（像素单位），越大顶点越少；典型 0.5-2.0
- `normalize_to`：`(W, H)` 元组；为 `None` 返回像素坐标，否则坐标归一化到 `[0, 1]`

返回最大连通域的多边形 `[[x1, y1], [x2, y2], ...]`；空 mask 返回 `[]`。

## 性能 / 边界

- O(N) findContours + O(N log N) simplify；512×512 mask < 5ms
- 多连通域只取面积最大者（v0.9.x 业务约定每实例只一个多边形）
- tolerance 过大时可能简化为线段（< 4 顶点），调用方应判断 `len(poly) >= 3`
