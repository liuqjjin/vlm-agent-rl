# Contributing to vlm-agent-rl

Thank you for your interest in contributing to this research project.

## Project scope

This repository is a focused research fork of [VAGEN](https://github.com/mll-lab-nu/VAGEN) investigating short-context versus full-trajectory credit assignment for vision-language agents. Contributions should align with the documented research questions in [EXPERIMENTS.md](EXPERIMENTS.md).

## Before contributing

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the data flow and invariants.
2. Review [DECISIONS.md](DECISIONS.md) for the rationale behind key design choices.
3. Check [UPSTREAM.md](UPSTREAM.md) to understand the relationship with VAGEN and verl.

## Development setup

### CPU testing environment

```bash
git clone --recurse-submodules https://github.com/liuqjjin/vlm-agent-rl.git
cd vlm-agent-rl
bash scripts/setup_cpu_env.sh
conda activate vagen
```

Run the focused smoke suite and the full CPU regression suite:

```bash
conda run -n vagen bash scripts/run_smoke.sh
conda run -n vagen python -m pytest -q vagen/tests verl/tests/trainer/ppo
```

Both commands must finish with zero failures. The current count is recorded in
`PROJECT_STATUS.md` rather than duplicated here.

### GPU environment

See [GPU_EXECUTION_CHECKLIST.md](GPU_EXECUTION_CHECKLIST.md) for the complete setup on Linux NVIDIA hardware.

## Code standards

### Testing requirements

- New advantage estimators must include reconstruction, duplicate handling, and zero-variance tests.
- New reward reductions must demonstrate invariance to trajectory splitting in controlled conditions.
- Changes to critic supervision or policy objectives require 20-step gradient checks.
- All parity gate logic must verify both pass and fail paths.

### Code style

- Follow the existing project conventions: type hints, docstrings for public functions, and explicit error messages.
- Keep implementation changes concentrated in tested extension points rather than forking upstream orchestration.
- Document invariants that callers must maintain.

### Docstrings

Use concise docstrings that state:
- What the function does (one sentence).
- Key parameters and return values.
- Invariants or assumptions (if not obvious from types).

Example:

```python
def compute_policy_weights(
    response_mask: torch.Tensor,
    group_idx: Any,
    traj_idx: Any,
    turn_idx: Any,
    *,
    mode: str,
) -> torch.Tensor:
    """Return normalized token weights for token/turn/trajectory objectives.

    The returned weights sum to one over active response tokens. Exact
    (group, trajectory, turn) duplicates receive zero weight, making the
    objective invariant to DP padding.
    """
```

## Submitting changes

### Pull request process

1. Create a feature branch from `main`.
2. Make your changes with clear, atomic commits.
3. Run the full CPU smoke test suite.
4. Update relevant documentation (ARCHITECTURE.md, EXPERIMENTS.md, or this file).
5. Submit a pull request with:
   - Clear description of what changed and why.
   - Test evidence (new test output or updated CPU results).
   - Impact on GPU experiments (if applicable).

### Commit messages

Write clear commit messages in imperative mood:

```
Add state-relative preflight coverage check

- Check anchor presence and comparable-row fraction
- Require action diversity in comparable groups
- Block training when signal is not identifiable
```

### What to avoid

- Do not modify the `verl/` submodule without coordinating separately; changes belong in the upstream fork.
- Do not add new dependencies without justification; the project uses VAGEN's pinned environment.
- Keep GPU planning predictions explicitly labelled and separate from measured results;
  every measured claim must be traceable to manifests and raw artifacts.

## Research contributions

### Proposing new experiments

Open an issue describing:
- The research question.
- How it relates to existing work (cite relevant papers).
- Required changes to the experiment matrix.
- Expected CPU and GPU cost.

### Sharing GPU results

If you run the formal GPU experiments:
1. Preserve all artifacts: manifests, raw rollouts, parity JSON, GPU samples, logs.
2. Run the analyzer: `bash scripts/run_experiment_matrix.sh analyze --run <path>`.
3. Submit results with full provenance (commit hashes, environment snapshots, seeds).
4. Update `results/main_results.csv` with traceable evidence paths.

## License and attribution

- Top-level contributions remain under the MIT license ([LICENSE](LICENSE)).
- Changes to `verl/` are Apache-2.0 ([verl/LICENSE](verl/LICENSE)).
- Credit prior work: cite papers and link to upstream repositories.

## Questions and discussion

For questions about:
- **Architecture and invariants**: open an issue referencing the relevant section of ARCHITECTURE.md.
- **Experiment design**: reference EXPERIMENTS.md and the declarative matrix in `experiments/matrix.yaml`.
- **Upstream VAGEN or verl**: direct questions to their respective repositories.

## Code of conduct

This project follows standard professional open-source norms:
- Be respectful and constructive in discussions.
- Focus on technical merit and research validity.
- Provide evidence for claims about performance or correctness.
- Acknowledge when you are uncertain or when results are preliminary.

## Acknowledgments

This project builds on [VAGEN](https://github.com/mll-lab-nu/VAGEN) and [verl](https://github.com/volcengine/verl). Please cite the VAGEN paper when using this framework:

```bibtex
@inproceedings{wang2025vagen,
  title={VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents},
  author={Kangrui Wang and Pingyue Zhang and Zihan Wang and Yaning Gao and Linjie Li and Qineng Wang and Hanyang Chen and Yiping Lu and Zhengyuan Yang and Lijuan Wang and Ranjay Krishna and Jiajun Wu and Li Fei-Fei and Yejin Choi and Manling Li},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://arxiv.org/abs/2510.16907}
}
```
