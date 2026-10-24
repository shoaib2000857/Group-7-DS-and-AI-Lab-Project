# Context, authority, and requirements

## Source precedence

Use the user's current POC-only instructions first. For course obligations use current official pages and the TA's written messages. Interpret the meeting transcript in context; it is automatically transcribed and contains errors. Treat the blueprint, old proposals, and other teams' reports as background rather than binding specifications.

## Supplied material inventory

| Material in `../DOCS/` | Interpretation |
|---|---|
| `AllOfficialProjectDocs.txt` | Copied course/team messages, including approval in principle and substantive written novelty feedback |
| `Standard Recording2.txt` | Timestamped review transcript; speaker labels are anonymous and wording may be mistranscribed |
| `Standard Recording2Summary.txt` | Noisy summary; includes unrelated or nonsensical lines and must not establish requirements |
| `guidelines – DSAI Lab Project Docs.html` | Saved official course guideline snapshot |
| `DSAI Lab_Group 7_Milestone 1 (1).pdf` | Group 7 report supplied as context; compare with root submitted artifacts before revisions |
| `TerraWatch M1 Review.pdf` | Review presentation with narrower urban-change scope |
| `TerraWatch_Milestone_1_Redesigned_Pitch_Deck.pdf` | Alternate presentation, not evidence of measured performance |
| `TerraWatch_Master_Research_and_Engineering_Blueprint.md` | Earlier broad engineering proposal; useful ideas but predates latest TA feedback |
| `datasets.txt` | Candidate shortlist, not a completed access/license/label audit |
| `IITM_DS_AI_Lab_Research_Project_Proposals (1).pdf` | Early comparison of possible projects, not an official requirement |
| `DSAI Lab_Group 4_Milestone 1.pdf` | Another team's indoor-navigation report; structural reference only |

The approved broad problem is remote-sensing change detection and interpretation. The submitted M1 report narrows it to visible urban/built-up change. This POC keeps that manageable scope while recording how later extensions would work.

## Official course requirements

The live [guidelines](https://bsc-iitm.github.io/dsai-lab-project/guidelines.html) and [milestones](https://bsc-iitm.github.io/dsai-lab-project/milestones.html) returned HTTP 200 on 24 October 2026 through a direct read after the browsing tool failed to open them. The saved HTML agrees on the headline timetable. Detailed M2–M5 requirements were also available on the live milestone page.

| Milestone | Due date shown | Required evidence relevant here |
|---|---|---|
| M1: problem/literature | October 22 | Precise scope, stakeholders, measurable objectives, comparisons and credible references; Markdown submission |
| M2: dataset preparation | October 29 | Sources, rights, size, characteristics, quality, split leakage, preprocessing reproducibility; explain subtask/data alignment |
| M3: architecture | November 5 | Model rationale, data/tensor interfaces, losses/metrics, and an actual end-to-end subset run |
| M4: training | November 19 | Training configuration, hyperparameter/optimization/regularization comparisons, logs and checkpoints |
| M5: evaluation | November 26 | Reproducible quantitative/qualitative comparison, successful and failed examples, error analysis and limitations |
| M6: deployment/docs | December 10 | Deployment and final documentation per guidelines; recheck detailed requirements when released |

The guideline lists four members; 40 marks split across milestones (15), novelty/complexity (5), and final presentation (20). Weekly individual worklogs and technical slide presentations matter. The final presentations are tentatively December 21 onward. These dates are course dates, not a promise that the entire POC must last until December.

The TA's written message says changes requested following review must be incorporated and resubmitted within 48 hours. The meeting was yesterday according to the user (23 October). The exact resubmission cutoff is not established by these relative timestamps; verify it in the team space. This pass prepares revision notes but does not edit or submit M1.

## TA feedback mapped to research tasks

| Feedback | Evidence location | Response in this pack |
|---|---|---|
| Establish a gap beyond integration; modularity is not inherently novel | Written messages; transcript 00:00–00:01 | Mechanism comparison and qualified contribution statement |
| Provide study/objective/data/design/findings/limitations matrix | Latest written feedback | Literature matrix with explicit reading limits |
| A predicted mask is not independent proof of a sentence | Written feedback; discussion around 00:40–00:46 | Wrong-mask challenge and image-backed verification proposal |
| Overlapping buildings make object counts difficult | Transcript approximately 00:45–00:46 | Separate region counts from building counts; ambiguity/abstention |
| Justify numerical targets | Transcript approximately 00:50–00:55 | Replace promised gains with measured pilot effects and uncertainty |
| Explain data strategy across modules | Transcript approximately 00:57–01:01 | One aligned primary dataset; optional datasets gated |
| Show ablations and reproducible comparisons | Transcript approximately 01:06–01:18 | Fixed contrasts, manifests, checkpoint provenance, error audit |
| Clarify legal/illegal construction | Transcript around 00:22 and 01:14–01:16 | Flag visible changes for human review; external records needed for legality |

Terminology: the literature legitimately calls two-date input *bitemporal*. To reflect the TA's concern around 00:17–00:19, describe the implemented task as **two-date image-pair comparison**, and reserve *time-series development tracking* for a later system with three or more observations and temporal validation.

## Repository observations

Read-only inspection found root `README.md`, `docs/Problem_Statement.md`, `reports/Milestone1_Report.md`, submitted PDFs, placeholder notebooks, configs, and skeletal modules. `app/app.py` prints an entrypoint message; the inspected model test checks YAML loading. Root requirements list general data-science packages but no established deep-learning stack. No functioning detector/captioner was established by this inspection.

Initial `git status --short` showed `?? POC/` and no tracked-file changes. Research additions stay under `POC/`. Existing submitted files remain unchanged. Earlier claims that data links were checked or that a private shared copy exists are historical team statements; this pass does not independently verify those assets.
