# 立直麻将手牌分析工具

> 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。
> OCR 仅读取用户明确提供的本地静态图片；不包含抓包、进程注入、自动截图、
> 模拟点击或网络请求。

## 安装与运行

需要 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

也可以直接传入手牌，或运行 3 组内置测试：

```bash
python main.py --hand 1m 2m 3m 1p 2p 3p 1s 2s 3s 7s 8s 9s 1z 9m
python main.py --demo
python -m unittest discover -s tests -v
```

## 本地图片 OCR

把图片裁到只保留自己的一行横向、正置暗牌，然后运行：

```bash
python main.py --ocr ./hand.png --ocr-count 14
```

有吃、碰、杠时，只识别可操作的暗牌，不把副露牌当作候选舍牌。暗牌合法
张数为 `13/14、10/11、7/8、4/5、1/2`；例如已经副露两组、摸牌后有 8 张
暗牌时使用：

```bash
python main.py --ocr ./open-hand.png --ocr-count 8
```

程序会先打印标准化识别结果和每张牌的置信度，再交给现有分析核心。保存带
检测框的调试图片：

```bash
python main.py --ocr ./hand.png --ocr-count 14 --ocr-debug ./ocr-debug.png
```

默认低于 `0.62` 的结果会停止，避免错误牌名直接进入分析。人工核对无误后可
使用 `--ocr-allow-low-confidence` 继续。OCR 只对传入的静态文件操作，不会
自动截屏或连接游戏。

默认牌图模板来自公共领域的
[FluffyStuff/riichi-mahjong-tiles](https://github.com/FluffyStuff/riichi-mahjong-tiles)。
模板匹配最适合电脑截图；实体牌照片、明显透视、遮挡或不同游戏皮肤可能需要
自定义模板。自定义模板目录通过 `--ocr-templates DIR` 指定，命名规则见
`assets/ocr_templates/README.md`。

### 校准雀魂牌面

如果有雀魂“麻将牌”说明页（按 9 万+4 风、9 筒+3 元、9 索排列的 34 种牌
总览图），可以一次性生成本机专用模板：

```bash
python main.py --ocr-calibrate-mahjong-soul ./mahjong-soul-tile-guide.png
```

模板会保存到 `local_ocr_templates/mahjong_soul/`，之后运行 `--ocr` 会自动
优先使用。该目录已被 Git 忽略，雀魂参考图和裁出的素材不会上传到公开仓库。

## 关于闭门与副露后的张数

规则上，`3n+1` 张暗牌是等待摸牌状态，不能先舍牌；`3n+2` 张暗牌才会遍历
候选舍牌。闭门时对应 13/14 张；每有一组副露，暗牌张数减少 3。向听库会
把缺少的面子视为已经副露，并按以下规则排序：

1. 舍牌后的向听数越小越优先；
2. 向听数相同，理论有效进张总枚数越大越优先。

理论有效枚数按每种牌 4 枚减去当前手牌中的枚数计算，不扣除牌河、副露等
场上可见牌。分析同时考虑普通形、七对子与国士无双的最小向听数。

## OCR 与输入层的接口

OCR 模块输出标准化牌名列表，例如
`["1m", "2m", "3m", "2p", ...]`，然后调用：

```python
internal_hand = hand_input(ocr_output_tiles)
result = core_analyze(internal_hand)
```

输入层会负责格式校验和 34 数组转换，核心算法无需修改。
