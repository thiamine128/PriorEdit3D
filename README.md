# PriorEdit3D

### Learning 3D Editing without Paired Supervision via Generative Prior Distillation

**SIGGRAPH Asia 2026 — Conference Papers**

Hao Wen, Weibin Yun, Hongxing Fan, Haotian Lu, Rui Chen, Zehuan Huang, Lu Sheng

[Project Page](https://thiamine128.github.io/PriorEdit3D/) · [Paper](https://arxiv.org/abs/2609.04942) · [PDF](https://arxiv.org/pdf/2609.04942) · [DOI](https://doi.org/10.1145/3829340.3842352)

PriorEdit3D learns instruction-guided, feed-forward 3D editing without paired 3D supervision by distilling visual, semantic, and geometric priors from pretrained foundation models.

![PriorEdit3D local and global editing examples](assets/images/editing-examples.png)

## Method

![PriorEdit3D training framework](assets/images/method-overview.png)

Our editor learns from three complementary signals:

- **2D visual supervision** transfers edits from an image-editing teacher.
- **VLM semantic feedback** encourages instruction following and source identity preservation across views.
- **3D distribution matching** regularizes geometry using a pretrained image-to-3D teacher.

## Code

Training entry points: [warmup](warmup_lightning.py) and [unpaired editing](unpaired_lightning.py), with configurations in [configs/editing](configs/editing).

The current release requires additional setup: training data, pretrained weights, a complete environment specification, and the inference scripts referenced by `inference.sh` are not included.

## Citation

```bibtex
@inproceedings{wen2026prioredit3d,
  title     = {Learning {3D} Editing without Paired Supervision via Generative Prior Distillation},
  author    = {Wen, Hao and Yun, Weibin and Fan, Hongxing and Lu, Haotian and
               Chen, Rui and Huang, Zehuan and Sheng, Lu},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  series    = {SA Conference Papers '26},
  year      = {2026},
  doi       = {10.1145/3829340.3842352},
  url       = {https://doi.org/10.1145/3829340.3842352}
}
```

## License

[Apache License 2.0](LICENSE). Third-party components and model weights retain their respective licenses.
