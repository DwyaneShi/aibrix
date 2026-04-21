 Review `python/aibrix_kvcache` in the current branch against `upstream/main`, then carry the whole task through end-to-end without stopping at analysis.

  Goals:
  1. Collect the review findings of `python/aibrix_kvcache` for the branch relative to `upstream/main`.
  2. Write them into `artifacts/code-review.md` as a living document.
  3. Validate every finding against the actual current code.
  4. Assign practical severity to each issue.
  5. Keep runtime receipts under `artifacts/`.
  6. Update the living document with both source-level and runtime evidence.
  7. Draft inline GitHub PR review comments anchored to the exact file and line of each finding, plus a short top-level summary comment.

  Requirements:
  - Treat `artifacts/code-review.md` as a living document. Update it in place if it already exists.
  - For each finding, record:
    - status: `Confirmed`, `Partially confirmed`, or `Not confirmed`
    - source-level evidence with exact file paths and line references
    - practical severity and impact
    - runtime reproduction result, if reproduced
    - receipt paths
    - conclusion
  - Use the real codebase, not assumptions.
  - If a finding is not valid, say so explicitly and explain why.
  - If a finding is only partially valid, narrow it precisely.
  - Run outside the sandbox when needed and ask for approval through the normal tool flow.
  - Save all receipts under a dedicated directory such as `artifacts/repro-runtime-YYYYMMDD/`.
  - Keep logs, command outputs, relevant generated files, and small summaries that make the proof easy to inspect.
  - Do not overwrite unrelated user changes.
  - Do not stop after gathering evidence; finish by updating the document, then present the planned GitHub comments to the user for confirmation before posting.

  Final response to me:
  - Keep it concise.
  - Tell me where the living document is.
  - Tell me where the receipts are.
  - Mention any caveats encountered during reproduction.
