# Stage 2 cascade benchmark report

Date: 2026-09-04

## Decision

Use the **rescue cascade** after Stage 1 and continue feeding stratified candidates until it produces 4,000 final accepted editing pairs:

1. Qwen2.5-VL-72B evaluates every pair.
2. Only an ordinary, low-risk Qwen `accepted` is accepted immediately.
3. Every other row—including Qwen `rejected`, parse/error cases, removal/high-risk rows, Stage-1 boundary rows, and robot/synthetic-subject rows—is decided by Gemini 3.8 Flash.
4. A failed or non-binary final answer is quarantined and never enters training.

The hard-reject cascade is rejected because Qwen's false rejects are too frequent. The rescue cascade passes the frozen test gate and is not significantly worse than Gemini-only. Therefore the fallback “Gemini-only on 2,000 rows” is not activated.

## Dataset split

| Split | Rows | Human accepted | Human rejected | Purpose |
|---|---:|---:|---:|---|
| Train | 94 | 76 | 18 | Prompt/error analysis only |
| Validation | 45 | 35 | 10 | Choose and freeze the route |
| Test | 91 | 65 | 26 | One-shot scale-up decision |
| Total | 230 | 176 | 54 | — |

Train, validation, and test results are reported separately. The test result must not be used to modify the prompt and then be reported again as an unseen result.

## Validation route selection

| Method | Bad recall | Accuracy | Good false-reject rate | Caught / bad |
|---|---:|---:|---:|---:|
| Qwen-only | 30.0% | 66.7% | 22.9% | 3 / 10 |
| Gemini-only | 70.0% | 73.3% | 25.7% | 7 / 10 |
| Qwen hard-reject cascade | 60.0% | 62.2% | 37.1% | 6 / 10 |
| **Rescue cascade** | **60.0%** | **77.8%** | **17.1%** | **6 / 10** |

The rescue cascade was selected over the hard-reject cascade. It catches the same six bad rows while rescuing seven good rows that the hard-reject version would wrongly discard. Relative to Gemini-only it catches one fewer bad row, but lowers the validation false-reject rate by 8.6 percentage points and raises accuracy by 4.4 points.

## Frozen test result

| Method | Bad recall | Accuracy | Good false-reject rate | Reject precision | Caught / bad |
|---|---:|---:|---:|---:|---:|
| Qwen-only | 53.8% | 68.1% | 26.2% | 45.2% | 14 / 26 |
| Gemini-only | 84.6% | 82.4% | 18.5% | 64.7% | 22 / 26 |
| Qwen hard-reject cascade | 80.8% | 68.1% | 36.9% | 46.7% | 21 / 26 |
| **Rescue cascade** | **80.8%** | **86.8%** | **10.8%** | **75.0%** | **21 / 26** |

The previously frozen 4K gate was:

- bad/rejected recall at least 80%;
- good/accepted false-reject rate no more than 25%;
- accuracy at least 78%;
- at most one fewer human-rejected row caught than Gemini-only.

The rescue cascade passes all four conditions. It catches 21 rather than Gemini's 22 bad rows, while keeping 58 rather than 53 good rows. It is therefore suitable for the 4K accepted-output run under the agreed rule.

## Cost and runtime

- Qwen processed 230/230 rows with zero parse failures.
- Qwen input/output tokens: 1,824,652 / 26,031.
- Qwen API cost: USD 1.4815.
- Qwen wall time at four-way concurrency: 502.8 seconds; mean API latency 7.62 seconds/pair.
- The selected rescue route would call Gemini on 160/230 rows (69.6%). Reusing existing Gemini results, their recorded routed cost is USD 0.4858.
- Effective benchmark-equivalent rescue cost: USD 1.9673, about USD 0.00855/pair.
- Linear 4,000-input estimate: about USD 34.2 before retry/price drift. Because the required output is now 4,000 accepted pairs, the real input count and cost will be higher and must be projected from the first production batch's per-branch acceptance rates.

Gemini-only is cheaper in this benchmark, but the selected cascade retained more human-accepted rows on test while remaining within one missed-bad example of Gemini-only. The 4K decision follows the predeclared quality gate rather than cost alone.

## Next execution gate

Do not feed the old 4,003-row list directly to Stage 2. First rebuild Stage 1 using the current rules, corrected ReCo unresolved provenance, both-side ReCo quality checks, stricter near-freeze rejection, and the removal blur hard gate. Stage 1 then flows into Stage 2 without an interim human gate. Sample/expand candidates according to the frozen task quotas and keep running the selected rescue cascade until 4,000 editing pairs are finally accepted.

After the complete production run, upload one stratified review package containing direct accepts, Gemini-reviewed accepts, Qwen-reject/Gemini-accept rescues, final rejects, model disagreements, quarantine, and boundary cases. Every row must show dataset/branch, training source/target, instruction, provenance-derived reference/generated roles, both model decisions/reasons, and final route. There is no human pause before this package is built.

## Machine-readable outputs

- `artifacts/cascade_benchmark_v1/qwen72b_prompt_v12_scores.csv`
- `artifacts/cascade_benchmark_v1/qwen72b_prompt_v12_summary.json`
- `artifacts/cascade_benchmark_v1/cascade_benchmark_rows.csv`
- `artifacts/cascade_benchmark_v1/cascade_benchmark_report.json`
