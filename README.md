# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:

### Task 1

My verifier is a single tool-calling LangChain agent (`create_agent`) built on
`deepseek-v4-flash` at temperature 0, driven by a detailed system prompt that
prescribes a fixed procedure: reconstruct the candidate's claims from the raw
CV text; search LinkedIn by full name plus the CV's city (adding industry when
the name is common, since hundreds of people share names); resolve the entity
by fetching every plausible profile and picking the one with real overlap
(employer or school in common), never just the first hit; optionally
corroborate city/education once via Facebook; then compare the CV against the
chosen LinkedIn profile field by field - name, city, each job (company, title,
seniority, start/end years, where a null end year means "Present"), each
education entry (degree, school, field, graduation year) and skills. The
prompt spells out equivalence rules so wording differences are not flagged
(BSc = Bachelor of Science, "Senior X" = title X with seniority senior, skill
variants, listing fewer skills is fine) and restricts discrepancies to
material mismatches in the checked fields only. The agent ends with a fenced
JSON verdict (`person_id`, `found_profile`, `discrepancies`, `red_flags`),
which `score_cvs` parses and maps to a score: 0.95 when the profile was found
and clean, 0.1 when a discrepancy exists, 0.25 when no profile is found or the
verdict is not grounded in actual tool outputs, and 0.4 as a per-CV fallback
after three failed attempts, so one bad CV never crashes the run (CVs are
processed concurrently under `asyncio.Semaphore(3)`). Because Task 2 attacks
this same pipeline, the agent is hardened against prompt injection: CV text is
scanned for injection/trust-manipulation markers, wrapped in `<cv_document>`
tags declared as untrusted data, capped at 30k characters, and every claimed
discrepancy or person_id must appear verbatim in real tool outputs before it
counts; red-flagged documents cannot score above 0.45. On `public_test` the
agent scores **7/7 correct**: the four valid CVs (1, 2, 3, 6) receive 0.95 and
all three planted discrepancies are caught - CV_4's shifted start year at
Innovate HK Ltd (2015 vs 2014), CV_5's inflated title at Microsoft (Senior
Analyst vs junior Analyst) and CV_7's fabricated Docker skill each receive
0.1.

### Task 2

My attack exploits **verification coverage**: the verifiers only compare five
field types (name, city, jobs, education, skills) against tool evidence, and
the SocialGraph profile schema contains no certifications, awards, languages
or interests at all. So the CV keeps every *checkable* field exactly
consistent with Kelly Tsang's real LinkedIn profile - name, Central/Hong Kong,
Manager at AIA 2022-Present, Analyst at Standard Chartered 2019-2022, Analyst
at HSBC 2016-2019, BSc in Finance from HKUST (2016), and precisely her four
listed skills - while the fabrication lives entirely in sections no tool can
return: invented certifications (CFA Level II candidate, FRM Part I, HKSI
Paper 1), invented awards (an AIA Top Performer Award and a Standard Chartered
recognition), plus languages and interests. Unlike a fake skill (public CV_7's
Docker), whose absence is *provable* by comparing against the profile's skills
list, the absence of a certification is *unprovable*: no MCP tool returns a
certifications field, so a field-by-field verifier has nothing to compare
these claims against and cannot flag them without also false-flagging the
hobbies/languages sections of honest CVs. The invented credentials double as
authority cues (LLM-as-a-judge bias/sycophancy): a CFA/FRM-certified,
award-winning profile reads as more trustworthy, nudging borderline judges
upward, and the embellished but unverifiable achievement bullets (e.g.
"introduced SQL workflows that cut monthly close time") add plausibility
without touching any checked field. Tested against my own Task 1 agent, which
is hardened against injection and hallucinated evidence, this CV scores 0.95
(trustworthy) while the true `target_cv.pdf` scores the same 0.95 - the agent
finds zero discrepancies, exactly as designed.


### Task 3: Reflection on the AI paradoxes

This homework made the paradoxes concrete for me: the same model that powers my verifier is the one my attack exploits, so every improvement in AI-driven verification simultaneously raises the value of fooling it. My agent catches a fabricated Docker skill in seconds, yet waves through invented CFA credentials, because an AI can only verify what it is wired to check - verification is fast but fundamentally narrow. The more we trust such automated gatekeepers in KYC and hiring, the less humans look themselves, and the more damage a single well-crafted deception can do. In the end the strongest defence I built was not a smarter model but old-fashioned grounding: forcing every claim back to raw tool evidence, which is a reminder that AI judgment is only as trustworthy as the human-designed constraints around it.
