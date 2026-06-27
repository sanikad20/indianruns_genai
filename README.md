# Redrob Intelligent Candidate Discovery & Ranking — Submission

**Challenge:** The Data & AI Challenge (Redrob Intelligent Candidate Discovery & Ranking)

**Team:** Genai

| Name | GitHub |
|---|---|
| Sanika Deshmukh (Leader) | [@sanikad20](https://github.com/sanikad20) |
| Divya Sreehari Addagatla | [@adivya15](https://github.com/adivya15) |
| Pragati Kharat | [@pragatikharat17](https://github.com/pragatikharat17) |

## What this is

This is our submission for the Redrob Intelligent Candidate Discovery & Ranking
Challenge. Given a pool of ~100,000 candidate profiles, the goal is to rank them
against a job description for a Retrieval/Ranking-focused AI Engineer role, using
nothing but CPU-bound heuristics — no embeddings, no hosted LLMs, no external APIs.

`rank.py` reads `candidates.jsonl`, scores every candidate, and writes out
`submission.csv` with `candidate_id`, `rank`, `score`, and a short recruiter-style
`reasoning` string explaining why each candidate landed where they did.

## A note on the dataset

We have **not** included `candidates.jsonl` in this repo. The file is around
**500 MB**, well past what GitHub will let you push (GitHub blocks files over
100 MB without LFS, and even with LFS it's not something we wanted to deal with
for a hackathon submission). So instead of a `git push` that quietly fails or a
repo that takes forever to clone, we've just left it out.

If you want to actually run `rank.py`, drop your own `candidates.jsonl` (same
schema as the one provided for the challenge) into the project root before
running the script. The script expects it at `./candidates.jsonl` by default.

## How we approached the scoring

We didn't try to throw a model at this — at 100k candidates with a 5-minute,
16GB, CPU-only budget, embeddings or any kind of inference were off the table.
Instead we built a composite heuristic score out of several components, each
one trying to capture a specific signal a human recruiter would actually look
for:

| Component | What it captures | Weight |
|---|---|---|
| `title_score` | How closely the current title matches the role (tiered: AI/ML Engineer > Backend-with-ML > Data/Analytics Engineer > generic Software Engineer > Consultant-only) | 0.22 |
| `career_score` | Retrieval/ranking/vector-DB depth, production LLM deployment, consulting vs. product-company ratio, job-hopping penalty, career progression | 0.27 |
| `skills_score` | Trust-weighted skill match — proficiency × endorsements × duration × assessment score, with mandatory JD skills (retrieval, ranking, embeddings, vector DBs, Python) weighted above optional ones, and a penalty for keyword-stuffed skill lists | 0.25 |
| `experience_score` | Years of experience, peaking at 6–8y with a smooth (not cliff-edged) falloff on either side | 0.10 |
| `location_score` | India-based preferred, Pune/Noida slightly favored per the JD | 0.08 |
| `notice_score` | Shorter notice periods score higher | 0.05 |
| `education_score` | Institution tier / field of study | 0.03 |

On top of the weighted sum:

- **Behavioral signals** (recruiter response rate, GitHub activity, last-active
  date, verified email/phone, etc.) are applied as a **multiplier** (~0.65×–1.25×)
  rather than an additive bonus, so an unreachable or inactive candidate gets
  scaled down proportionally instead of just losing a flat handful of points.
- **Seniority-vs-experience plausibility**: a "Senior"/"Lead"/"Staff" title
  backed by way less experience than that title would normally require gets a
  proportional discount. This isn't a honeypot disqualification — titles
  legitimately vary by company — but it stops an inflated title with a thin
  career from outranking someone genuinely senior.
- **Honeypot detection** runs first and zeroes out anything with impossible
  data: durations that couldn't fit in the calendar, overlapping full-time
  jobs, junior-to-VP progression in under two years, "expert" skills with zero
  duration behind them, or a stated YOE that doesn't match the actual career
  timeline.

Everything is regex/token-matched with word boundaries (not raw substring
checks), specifically to avoid the kind of false positives that plague naive
keyword matching — e.g. "RAG" as a skill shouldn't light up because someone's
job description happened to contain the word "storage" or "average."

## Architecture

The whole thing is a single linear pipeline — no classes, no external services,
nothing async. Everything happens in one O(n) pass over the candidate list, with
a second tiny O(n) pass at the end to normalize scores. That's deliberate: at
100k candidates and a 5-minute budget, the simplest thing that works is also
the safest thing.

```
candidates.jsonl
       │
       ▼
┌─────────────────────┐
│   load_candidates()  │   stream-parse JSONL, one record at a time
└─────────┬────────────┘
          │  for each candidate record
          ▼
┌─────────────────────────────────────────────────────────────┐
│                     detect_honeypot(candidate)               │
│  impossible dates · overlapping jobs · 0-duration "expert"   │
│  skills · junior→VP in <2y · stated YOE vs career timeline   │
└───────────────────────┬───────────────────────────────────────┘
                         │  honeypot → score = 0.0, skip rest
                         ▼  otherwise, continue
┌─────────────────────────────────────────────────────────────┐
│                       score_candidate(candidate)              │
│                                                                │
│   title_score(title)              ─┐                          │
│   career_score(history, profile)   │   each sub-score is       │
│   skills_score(skills, scores)     │   independent and only    │
│   experience_score(yoe)            │   reads its own slice     │
│   location_score(loc, ...)         │   of the candidate record │
│   notice_score(notice_days)        │   (no cross-talk, no      │
│   education_score(education)      ─┘   shared mutable state)  │
│                                                                │
│   weighted sum of the seven scores above                      │
│        × behavioral_score(signals)        (multiplier)        │
│        × seniority_plausibility_factor()  (multiplier)        │
│                                                                │
│   → (raw_composite_score, components_dict)                    │
└───────────────────────┬───────────────────────────────────────┘
                         │  collect raw scores for every candidate
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   normalize_scores(raw_scores)                 │
│   z-score → sigmoid, so the final 0–1 scores are spread       │
│   sensibly across the whole pool instead of bunching near      │
│   the top (which raw weighted sums tend to do)                 │
└───────────────────────┬───────────────────────────────────────┘
                         │  sort by normalized score, descending
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              generate_reasoning(candidate, components)         │
│   builds the recruiter-readable explanation string per row:    │
│   title + YOE → strongest relevant career evidence → top      │
│   skills → location/notice/behavioral notes → concerns        │
└───────────────────────┬───────────────────────────────────────┘
                         ▼
                  submission.csv
         (candidate_id, rank, score, reasoning)
```

### Why it's structured this way

- **Honeypot detection runs before scoring, not after.** A fabricated profile
  shouldn't get to "compete" on its (fake) merits at all — it's cheaper and
  safer to zero it out up front than to let it slip through on a high score
  and then try to catch it later.
- **Sub-scores are pure functions.** `title_score`, `career_score`,
  `skills_score`, etc. each take plain values in and return a plain float out.
  No shared state, no candidate-to-candidate dependency. That's what keeps the
  whole thing O(n) — you could run these in parallel across candidates with no
  changes if you ever needed to.
- **Behavioral signals and seniority plausibility are multipliers, not
  additive terms.** An unreachable candidate or an inflated title should scale
  the *whole* score down proportionally, not just lose a few flat points that
  matter less the higher the base score already is.
- **Regex patterns are compiled once at module load** (`_compile_patterns`),
  not per-candidate, so keyword/skill matching doesn't pay re-compilation cost
  100,000 times over.
- **Normalization is the only step that needs the full set of scores at once**
  — everything before it can be computed independently per candidate, which is
  why it's split out as its own pass at the end rather than folded into
  `score_candidate`.

## Files

- `rank.py` — the full scoring + ranking pipeline
- `submission.csv` — the final output: 100 ranked candidates with scores and
  reasoning, generated by running `rank.py` against the full candidate pool
- `README.md` — this file

## Running it

```bash
python3 rank.py
```

This expects `candidates.jsonl` in the same directory (see the dataset note
above) and writes `submission.csv` alongside it. On the full ~100k-candidate
set this comfortably finishes well inside the 5-minute / 16GB budget — there's
no nested per-candidate-pair work anywhere, everything is a single O(n) pass
with a handful of precompiled regex patterns reused across all candidates.

## Known limitations

- All scoring is heuristic. It's tuned against the JD and against spot-checks
  of the actual output (we went through several rounds of "wait, why did this
  candidate rank so high" and adjusted), but it's not learned from labeled
  ranking data, because none was available.
- The seniority-plausibility check uses fairly coarse minimum-YOE-per-title-
  level assumptions. It's deliberately a soft discount (floor of 0.55×, not a
  hard cutoff) so it doesn't unfairly tank genuinely fast-tracked candidates.
- Company-name recognition in the reasoning text is descriptive only — it's
  not used as a scoring signal (we deliberately removed an earlier "known
  strong AI company" bonus, since employer brand isn't evidence of skill).
