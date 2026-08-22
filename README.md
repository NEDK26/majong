# 立直麻将手牌分析工具

> 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。
> 不包含抓包、进程注入、模拟点击、OCR 或网络请求。

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

## 关于 13 张与 14 张

规则上，13 张是等待摸牌的状态，不能先从 13 张中舍一张。工具会为 13 张
手牌显示当前向听数和有效摸牌，并在交互模式中允许补录刚摸到的第 14 张；
拿到 14 张后，才会遍历合法候选舍牌并按以下规则排序：

1. 舍牌后的向听数越小越优先；
2. 向听数相同，理论有效进张总枚数越大越优先。

理论有效枚数按每种牌 4 枚减去当前手牌中的枚数计算，不扣除牌河、副露等
场上可见牌。分析同时考虑普通形、七对子与国士无双的最小向听数。

## 后续接入 OCR

OCR 模块只需要输出标准化牌名列表，例如
`["1m", "2m", "3m", "2p", ...]`，然后调用：

```python
internal_hand = hand_input(ocr_output_tiles)
result = core_analyze(internal_hand)
```

输入层会负责格式校验和 34 数组转换，核心算法无需修改。
