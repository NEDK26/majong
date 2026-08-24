# 雀魂 OCR 分层分类器

`mahjong_soul_family_classifier.npz` 由
[`pjura/mahjong_souls_tiles`](https://huggingface.co/datasets/pjura/mahjong_souls_tiles)
训练生成，数据集采用 Apache-2.0 许可。

原始图片右上角带有类别数字或字母。训练与运行时共用 `ocr_input.py` 的预处理：
先移除牌角高饱和色角标，再提取 24×32 灰度图案和颜色占比。模型按万、筒、索、
字牌分别训练一对一线性判别器；产物只包含线性权重和归一化风格原型，不包含原图。

重新训练：

```bash
python tools/train_ocr_family_classifier.py \
  /path/to/mahjong_souls_tiles \
  assets/ocr_models/mahjong_soul_family_classifier.npz
```

分类器只在模板对同花色牌判断含糊、且截图更接近训练风格时参与覆盖。自定义牌皮或
高置信度模板结果仍以本地模板为准。
