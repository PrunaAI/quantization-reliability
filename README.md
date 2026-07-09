# Reliability Evaluation of Extreme Quantization for Large Language Models

[![Paper](https://img.shields.io/badge/Paper-OpenReview-red)](https://openreview.net/forum?id=jehhhDHum2&nesting=2&sort=date-desc) [![Poster](https://img.shields.io/badge/Poster-Google%20Drive-4285F4)](https://drive.google.com/file/d/1mwWKyoR03iH0K4S17mO-G17IGpjMORgv/view?usp=sharing) [![Stars](https://img.shields.io/github/stars/sandordaroczi/quantization-reliability?style=social)](https://github.com/sandordaroczi/quantization-reliability)

*Authors: Sirine Ayadi, Sándor Daróczi, Stephan Günneman, Bertrand Charpentier*

*Published at TMLR 2026 & ICML 2026 SCALE Workshop*

We study reliability bit-level scaling laws for quantized LLMs to find the optimal precision that maximizes the reliability under a fixed bit budget. Our reliability evaluation covers uncertainty, calibration, and robustness to 15 natural input perturbations. We find that 4-bit precision offers the best reliability-efficiency tradeoff across tasks, model families, and quantization methods.

## Robustness under text perturbations

Real users type with tpyos, slang, 
emoji :), and miXeD CasE. We implement **15 natural input perturbations** on the character- and word-level to evaluate model robustness.

<p align="center">
  <img src="figures/perturbations.png">
  <br>
  <em>Overview of our character-level and word-level input perturbations. Illustrated is an example where perturbations with intensity level 1 are applied to a standard question prompt.</em>
</p>

<p align="center">
  <img src="figures/robustness_radar.png">
  <br>
  <em>Radar plots of the accuracy (Top) and AUCROC (Entropy) (bottom) across all 15 character-level and word-level perturbations for two intensities. We evaluate the base LLaMa-3-8B model and five 4-bit quantization methods. Quantized models can provide more reliable uncertainty estimates under natural perturbations compared to their base counterparts, while maintaining close performance.</em>
</p>

---

## Scaling laws for reliability

We characterize trends in reliability as the total number of bits scales. We model a metric as a function of total bits using a log quadratic scaling law.

<p align="center">
  <img src="figures/intro.png">
  <br>
  <em>Bit-level scaling trends of the accuracy and AUCROC (Entropy) on TriviaQA. We use four base models (blue): LLaMA-3.2-1B, LLaMA-3.2-3B, LLaMA-3-8B, and LLaMA-3-70B, and their corresponding quantized variants using six quantization methods and different bitwidths.</em>
</p>

---

## Code

> **Full code release comign soon.** Star this repo to get notified when it drops.

The release will include:

- Model quantization and loading scripts
- Reliability evaluation pipeline
- KL divergence evaluation scripts
- Hydra configs for all experiments in the paper
- Notebooks to reproduce the figures

---

## Citation

---

## License

MIT — see [LICENSE](LICENSE) for details.
