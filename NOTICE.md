# NOTICE

prefixgate's own code and text are MIT (see `LICENSE`). It downloads, reads or ships the following third-party
material, each under its own terms.

## Data

- **Policy Gates release**, [agentapprove/decision-policy-rules](https://github.com/agentapprove/decision-policy-rules)
  at commit `9e68b07a79cdabc731fae9894c270b9d7321cfe5`: `rounds/r4/dataset.json`, `rounds/r4/confirmation/dataset.json`
  and `rounds/r4/product-templates.json`, from James P. Beno, *Decision Models as Policy Gates for Coding Agents*
  ([arXiv:2611.02155](https://arxiv.org/abs/2611.02155)). CC BY 4.0. The cases are authored tool calls with
  fictional identities and random secret bodies; never run them. prefixgate downloads them into `data/`
  (gitignored) and checks their SHA-256. Rule definitions quoted in `RULES.md` are theirs, verbatim.
- **Attack templates**: the six injection templates in `prefixgate/data.py` (`ATTACKS`) are copied from
  *Ranked but Not Stopped: Detecting Prompt-Injection Hijacks from Typed Decision Models' Output Probabilities*
  ([arXiv:2611.02522](https://arxiv.org/abs/2611.02522)), with the target label set to `allow`.

## Model

- **Qwen2.5-0.5B-Instruct** (Qwen team, Alibaba Cloud), revision `7ae557604adf67be50417f59c2c2f167def9a775`,
  Apache-2.0. Downloaded from Hugging Face at run time, not redistributed. `results/adapter.safetensors` holds only
  the LoRA and head weights trained here.

## Art and fonts (banner only, `scripts/banner.py`)

- Gustave Doré, *Inferno, Canto III: "Abandon all hope, ye who enter here"* (1857), wood engraving. Public domain.
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Gustave_Dor%C3%A9_-_Dante_Alighieri_-_Inferno_-_Plate_8_(Canto_III_-_Abandon_all_hope_ye_who_enter_here).jpg).
- Albrecht Dürer, *Melencolia I* (1514), engraving. Public domain.
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:D%C3%BCrer_Melancholia_I.jpg).
- Pirata One by Rodrigo Fuenzalida and Nicolas Massi, SIL Open Font License 1.1
  ([google/fonts](https://github.com/google/fonts/tree/main/ofl/pirataone)). Rasterised into the banner image; the
  font file is not redistributed.
- DejaVu Sans Mono, Bitstream Vera / DejaVu license. Rasterised into the banner image.

## Software

PyTorch (BSD-3-Clause), Hugging Face Transformers (Apache-2.0), safetensors (Apache-2.0), NumPy and SciPy
(BSD-3-Clause), scikit-learn (BSD-3-Clause), matplotlib (matplotlib license), Pillow (MIT-CMU), Playwright
(Apache-2.0, demo capture only).
