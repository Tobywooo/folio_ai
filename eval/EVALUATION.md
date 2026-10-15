# RAG Evaluation

This compares four retrieval approaches in Folio AI on the same notebook and questions, and explains which one the app
uses by default.

**Result: hybrid search with semantic re-ranking (`hybrid_semantic`) is the default.** It produced the best answers
(8 of 8 graded Good, citations correct on all 8), retrieved the most relevant chunks per question, and had the lowest
average response time. Keyword-only search was fastest at retrieval but failed on questions phrased differently from
the source.

Supporting files:

| File | Contents |
|---|---|
| [`results/eval-20261014-195319.xlsx`](results/eval-20261014-195319.xlsx) | Workbook: summary formulas, per-question grades and notes, all 160 retrieved chunks (as snippets), all 32 answers |
| [`results/eval-20261014-195319.md`](results/eval-20261014-195319.md) | Full generated report: every retrieved chunk (as a snippet) and every answer |
| [`results/eval-20261014-195319.json`](results/eval-20261014-195319.json) | Raw results |
| [`questions.json`](questions.json) | The test questions and what each should retrieve |
| [`run_eval.py`](run_eval.py) | The evaluation script |

## Setup

| Setting | Value |
|---|---|
| Notebook | "Evaluation", 4 sources, 963 chunks |
| Sources | Business analytics textbook (Cengage, pages 1–300, PDF, 921 chunks) · course slides "Business Analytics Overview" (PPTX, 19 chunks) · UNCC study guide (PDF, 16 chunks) · Wikipedia "Business analytics" (web page, 7 chunks) |
| Chunking | Recursive character splitting, 1,200 characters, 200 overlap |
| Embedding model | `qwen3-embedding` (2,560 dimensions) |
| LLM | `qwen3.8-27b` |
| Vector store | Azure AI Search (HNSW index) |
| Top-k | 5 chunks per question |
| Runs | 8 questions × 4 methods = 32 runs, one run each, on 2026-10-14 |

## Techniques tested

| Method | How it ranks chunks |
|---|---|
| `keyword` | BM25 full-text search only. No embedding call. |
| `vector` | Embeds the question and returns the nearest chunk vectors (cosine similarity, HNSW). |
| `hybrid` | Runs keyword and vector search together and merges the two rankings with Reciprocal Rank Fusion. |
| `hybrid_semantic` | Hybrid, then Azure AI Search's semantic ranker (a cross-encoder) re-scores the top results against the question. |

Every method is filtered to the notebook's own chunks and feeds its top 5 chunks, numbered, to the same LLM prompt, so
the only difference between runs is retrieval.

## Test questions

Each question targets a different retrieval challenge. Where possible, the question names a source and a phrase that
a relevant chunk must contain, so retrieval can be scored automatically.

| # | Question | Type | Relevant when the chunk… |
|---|---|---|---|
| 1 | According to the textbook, what are the four Vs that IBM uses to describe big data? | Factual | is from the textbook and contains "four Vs" (not the table of contents) |
| 2 | How many pieces of direct mail does Kroger send to its customers each quarter, and how are the coupons chosen? | Factual (number) | is from the textbook and contains "11 million" |
| 3 | What share of data science initiatives end up not succeeding, and what typically goes wrong? | Paraphrase | is from the slides and contains "failure rate" |
| 4 | What is the data-ink ratio? | Exact term | contains "data-ink ratio" |
| 5 | How is business analytics different from business intelligence, and what do descriptive, predictive, and prescriptive analytics each do? | Multi-source | contains "prescriptive" (answer needs Wikipedia and the textbook) |
| 6 | Who were early pioneers of using analytics in business, long before computers? | Paraphrase (web page) | is from Wikipedia and contains "Henry Ford" |
| 7 | What qualities should data have before you trust it for decision making? | Paraphrase | is from the slides and contains "consistent" |
| 8 | What was Amazon's total revenue in 2023? | Unanswerable | — (the answer should say the sources don't cover it) |

## Results

### Retrieval

Hit rate = share of the 7 scored questions where a relevant chunk was in the top 5. MRR (mean reciprocal rank) =
average of 1 / rank of the first relevant chunk, so 1.0 means it was always ranked first.

| Method | Hit rate | MRR | Avg relevant chunks (of 5) | Avg retrieval time |
|---|---|---|---|---|
| keyword | 71% | 0.57 | 1.6 | **0.13 s** |
| vector | **100%** | **0.86** | 2.0 | 1.02 s |
| hybrid | 86% | 0.64 | 1.9 | 1.24 s |
| hybrid_semantic | 86% | 0.76 | **2.1** | 0.84 s |

### Answer quality and response time

Answers were graded by hand: **Good** = correct, complete, correct citations; **Partial** = mostly right or some
citation problems; **Poor** = wrong, invented, or claims the sources don't cover something they do. Quality score =
(2 × Good + Partial) / (2 × graded).

| Method | Good | Partial | Poor | Quality score | Citations correct | Avg generation time | Avg total response time |
|---|---|---|---|---|---|---|---|
| keyword | 6 | 1 | 1 | 81% | 5 of 8 | 9.34 s | 9.47 s |
| vector | 6 | 2 | 0 | 88% | 5 of 8 | 8.52 s | 9.54 s |
| hybrid | 7 | 1 | 0 | 94% | 7 of 8 | 11.61 s | 12.86 s |
| hybrid_semantic | **8** | 0 | 0 | **100%** | **8 of 8** | 8.22 s | **9.06 s** |

Total response time = retrieval (including embedding the question) + LLM generation.

### Per question

Each cell: relevant chunks in the top 5 / rank of the first relevant chunk · total response time · answer quality.

| Q | Type | keyword | vector | hybrid | hybrid_semantic |
|---|---|---|---|---|---|
| 1 | Factual | 3 / #2 · 5.6 s · Good | 3 / #1 · 4.4 s · Good | 3 / #1 · 4.3 s · Good | 3 / #1 · 4.0 s · Good |
| 2 | Factual (number) | 1 / #1 · 6.2 s · Good | 1 / #2 · 5.4 s · Good | 1 / #1 · 5.8 s · Good | 1 / #1 · 5.0 s · Good |
| 3 | Paraphrase | 0 / – · 10.4 s · **Poor** | 1 / #1 · 7.3 s · Partial | 1 / #4 · 7.5 s · Partial | 1 / #3 · 6.0 s · Good |
| 4 | Exact term | 5 / #1 · 6.1 s · Good | 5 / #1 · 9.9 s · Good | 5 / #1 · 11.0 s · Good | 5 / #1 · 7.5 s · Good |
| 5 | Multi-source | 1 / #2 · 29.5 s · Good | 2 / #2 · 27.2 s · Good | 2 / #4 · 23.4 s · Good | **4 / #1** · 16.8 s · Good |
| 6 | Paraphrase (web) | 1 / #1 · 3.9 s · Good | 1 / #1 · 8.6 s · Good | 1 / #1 · 8.6 s · Good | 1 / #1 · 7.4 s · Good |
| 7 | Paraphrase | 0 / – · 8.1 s · Partial | 1 / #1 · 5.7 s · Partial | 0 / – · 33.5 s · Good | 0 / – · 19.7 s · Good |
| 8 | Unanswerable | n/a · 5.9 s · Good | n/a · 7.8 s · Good | n/a · 8.8 s · Good | n/a · 6.0 s · Good |

### Retrieved chunks

Top 5 sources per run, in rank order (✓ = counts as relevant; s = slide, p = page). A 220-character snippet of
each chunk is in the workbook's *Retrieved Chunks* sheet and the generated report. Only snippets are stored, because
the sources include copyrighted course material and this repository is public.

| Q | keyword | vector | hybrid | hybrid_semantic |
|---|---|---|---|---|
| 1 | Textbook p278, p32 ✓, p33 ✓, p34 ✓, p23 | Textbook p32 ✓, p33 ✓, p27, p34 ✓, p7 | Textbook p32 ✓, p33 ✓, p34 ✓, p43, p35 | Textbook p32 ✓, p33 ✓, p34 ✓, p28, p32 |
| 2 | Textbook p35 ✓, p35, p165, p255, p276 | Textbook p35, p35 ✓, p165, p165, p275 | Textbook p35 ✓, p35, p165, p275, p165 | Textbook p35 ✓, p35, p165, p285, p31 |
| 3 | Textbook p185, p226 · Slides s7, s9 · Wikipedia | Slides s8 ✓, s1 · Study guide p3 · Slides s5, s9 | Slides s9 · Textbook p226, p227 · Slides s8 ✓ · Textbook p106 | Textbook p34, p35 · Slides s8 ✓, s9 · Textbook p43 |
| 4 | Textbook p114 ✓, p154 ✓, p115 ✓, p118 ✓, p115 ✓ | Textbook p133 ✓, p114 ✓, p154 ✓, p115 ✓, p115 ✓ | Textbook p114 ✓, p154 ✓, p133 ✓, p115 ✓, p115 ✓ | Textbook p114 ✓, p154 ✓, p115 ✓, p115 ✓, p116 ✓ |
| 5 | Wikipedia ×2 (one ✓), Wikipedia ×3 | Wikipedia · Textbook p2 ✓ · Wikipedia ×2 · Textbook p31 ✓ | Wikipedia ×3 · Textbook p31 ✓ · Wikipedia ✓ | Textbook p42 ✓ · Wikipedia · Textbook p32 ✓, p42 ✓, p31 ✓ |
| 6 | Wikipedia ✓, Wikipedia · Slides s4 · Textbook p226, p188 | Wikipedia ✓ · Slides s4, s1 · Wikipedia ×2 | Wikipedia ✓ · Slides s4 · Wikipedia ×3 | Wikipedia ✓, Wikipedia · Slides s4 · Textbook p29, p36 |
| 7 | Textbook p30, p30 · Wikipedia ×2 · Textbook p49 | Slides s18 ✓, s16 · Textbook p30 · Slides s17, s15 | Textbook p30, p49, p30, p34 · Wikipedia | Wikipedia · Textbook p49, p49, p30, p30 |
| 8 | Textbook p28, p118, p36, p28, p119 | Textbook p156, p156, p119, p120, p120 | Textbook p28, p28, p120, p156, p148 | Textbook p148, p120, p36, p119, p117 |

## What the results show

**Keyword search is fast and precise on exact wording, but misses paraphrases.** It needs no embedding call, so
retrieval took 0.13 s, about 8× faster than the other methods. It found the exact term (Q4: 5 of 5 relevant) and the
specific number (Q2) at rank 1. When the question used different words than the source, it found nothing relevant:
Q3 ("initiatives that don't succeed" vs. the slide's "failure rate") was the only Poor answer in the evaluation. For
Q5 it retrieved only the Wikipedia page, so the textbook's view of the three analytics types was missing from the
context.

**Vector search finds the right source most reliably.** It was the only method with a relevant chunk for every scored
question (hit rate 100%, MRR 0.86), and it found the exact slides for both paraphrased questions (Q3 slide 8, Q7
slide 18). Its answers were still only Partial on Q3 and Q7: the right slide was in the context, but the surrounding
chunks were weaker, and the answers were incomplete.

**Plain hybrid search can bury a good result.** Reciprocal Rank Fusion averages the keyword and vector rankings, so
when keyword search ranks irrelevant chunks highly, the relevant vector hit drops (Q3: rank 1 for vector, rank 4 for
hybrid; Q5: rank 4). Answer quality still improved over the single methods (94%), because the fused context covered
more sources.

**Semantic re-ranking fixes the ordering and gives the best answers.** Re-scoring the fused results against the
question put relevant chunks first on the multi-source question (Q5: 4 of 5 relevant, rank 1, chunks from both the
textbook and Wikipedia) and produced the only Good answer to Q3. All 8 answers were graded Good with correct
citations.

**Automatic scoring undercounts when several sources cover a topic.** On Q7 the hybrid methods retrieved the
textbook's data-quality pages instead of the slides. The automatic score marks that as a miss, but the answers were
complete and graded Good, better than vector's answer from the "correct" slide. Hit rate and MRR are useful signals,
but answer grading is still needed.

**All methods handled the unanswerable question correctly.** For Q8 every method said the sources don't contain
Amazon's 2023 revenue, and none invented a figure. That comes from the prompt instruction to answer only from the
numbered sources, not from retrieval.

## Response time tradeoffs

- **Retrieval is a small part of the response.** Across methods it took 0.13–1.24 s, against 8–12 s of LLM generation.
  Keyword search saves about 0.7–1.1 s per question, which users won't notice next to generation time.
- **The vector-based methods pay for the embedding call.** Embedding the question is an extra request to the model API,
  which accounts for most of the difference between keyword (0.13 s) and the other methods. Hybrid + semantic averaged
  slightly faster retrieval than vector alone, which is run-to-run variance in that API call, not a real advantage.
- **Better context tended to produce faster answers.** On Q5, the semantically re-ranked context took 16.8 s in total,
  against 23–29 s for the other methods. The worst case was hybrid on Q7 (33.5 s), where loosely related chunks
  produced a much longer answer, and that one run raised hybrid's average to 12.9 s.
- **Semantic ranking has a usage limit.** The free semantic ranker plan on Azure AI Search allows 1,000 semantic
  queries per month. That's plenty for a class project; heavier use needs the paid plan, or a fallback to `hybrid`.

## Decision

**`hybrid_semantic` is the default retrieval method** (`RETRIEVAL_MODE` in `.env.example`, the backend config, and the
UI's default selection). It had:

- the best answer quality (8 of 8 Good) and citation accuracy (8 of 8),
- the most relevant chunks per question (2.1 of 5), with the best result on the multi-source question,
- the lowest average total response time (9.1 s).

Vector search alone had a slightly higher hit rate and MRR on this set, but the re-ranked results produced better final
answers, which is what users see. Users can still switch methods in the Chat tab to compare them.

## Limitations

- **Small sample.** 8 questions, one run each, graded by one person. Differences of one grade or a couple of seconds
  shouldn't be read as significant.
- **Timing noise.** Times include calls to an external model API, which vary from run to run.
- **Relevance labels are substring matches.** A chunk counts as relevant only if it contains the expected phrase from the
  expected source, which misses equally good chunks elsewhere (see Q7).
- **Noisy source text.** 136 of the textbook's 921 chunks (15%) are repeated copyright notices, and the scraped Wikipedia
  text is broken into short lines. Removing that boilerplate and cleaning web text before chunking would likely help
  every method; comparing chunking before and after that cleanup would be a natural next experiment.

## Reproducing

From the repository root, with the backend dependencies installed and `.env` filled in:

```bash
python eval/run_eval.py --notebook-id <notebook-id> --modes keyword vector hybrid hybrid_semantic
```

The script writes a Markdown report and a JSON file to `eval/results/`. To rebuild the report from saved results
without calling the models again:

```bash
python eval/run_eval.py --notebook-id <notebook-id> --rerender eval/results/<file>.json
```
