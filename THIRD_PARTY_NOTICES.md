# Third-party notices

This repository is a research fork. Its top-level source remains under the
MIT license in [`LICENSE`](LICENSE). The following notices clarify the
separately licensed components used or studied by the project; they do not
replace the full upstream license texts.

## VAGEN

- Source: [mll-lab-nu/VAGEN](https://github.com/mll-lab-nu/VAGEN)
- License: MIT
- Copyright: Copyright (c) 2025 RAGEN.AI

The full inherited license is retained in [`LICENSE`](LICENSE).

## verl submodule

- Source base: [JamesKrW/verl](https://github.com/JamesKrW/verl), itself a
  fork of [volcengine/verl](https://github.com/volcengine/verl)
- License: Apache License 2.0
- Principal source notices identify ByteDance Ltd. and/or its affiliates;
  individual files also retain notices from their respective contributors.

The full license is retained at [`verl/LICENSE`](verl/LICENSE). The three local
submodule commits remain subject to that license and mark their new test
files as VAGEN contributor work.

## verl-agent

- Source: [langfengQ/verl-agent](https://github.com/langfengQ/verl-agent)
- License: Apache License 2.0
- Use in this project: read-only comparison of the public GiGPO/state-grouping
  design.

No verl-agent code or model artifacts are vendored or redistributed here.

## Qwen2.5-VL-3B-Instruct

- Source: [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- License: [Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE)
- Copyright notice required by that license: Qwen is licensed under the Qwen
  RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.

The weights are downloaded by the user and are not included in this
repository. The model license is research/non-commercial and imposes terms
beyond the MIT code license. Users must review it before use or distribution.

## Environment dependencies

- [AI2-THOR](https://github.com/allenai/ai2thor), Apache License 2.0.
- [gym-sokoban](https://github.com/mpSchrader/gym-sokoban), MIT License,
  Copyright (c) 2018 Max-Philipp Schrader.

These packages are installed from their upstream distributions and are not
vendored. Their complete license texts and any bundled assets remain governed
by the distributions from which they are obtained.

## Papers and attribution

This implementation also relies on ideas described in the VAGEN, Qwen2-VL,
DeepSeekMath, DAPO, and GiGPO papers. Exact links and audited commits are
listed in [`UPSTREAM.md`](UPSTREAM.md); implementation decisions are
separated from citations in [`DECISIONS.md`](DECISIONS.md).
