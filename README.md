# TMSP - Test for Medical Stepwise Predictions

A framework for evaluating LLM medical domain understanding through hierarchical ICD-10-CM code traversal.

## Overview

TMSP tests the consistency and accuracy of medical knowledge in language models by guiding them through a **stepwise decision tree** rather than asking for direct answers. Instead of prompting "What ICD-10 codes apply to this patient?", the system presents a series of 0..n candidate selection questions at each level of the medical coding hierarchy.

This approach reveals *how* an LLM reasons about medical concepts, not just *what* it outputs.

## The Problem with Direct Medical Coding

Traditional LLM evaluation asks models to produce final ICD-10-CM codes directly. This has several limitations:

- **Black box reasoning**: No visibility into the decision process
- **Inconsistent specificity**: Models may skip hierarchy levels arbitrarily
- **Missing lateral codes**: Comorbidities and "code also" relationships are often ignored
- **Unverifiable confidence**: No way to assess certainty at each decision point

## The Stepwise Solution

TMSP decomposes medical coding into a traversal problem:

```
Clinical Note: "65yo male with type 2 diabetes, presenting with diabetic retinopathy"

Step 1: Which ICD-10 chapters are relevant?
   [ ] Chapter 1: Infectious diseases
   [x] Chapter 4: Endocrine disorders      ← LLM selects
   [x] Chapter 7: Eye disorders            ← LLM selects
   [ ] Chapter 9: Circulatory system
   ...

Step 2a: Within Endocrine (E00-E89), which categories?
   [x] E11: Type 2 diabetes mellitus       ← LLM selects
   [ ] E13: Other specified diabetes
   ...

Step 2b: Within Eye disorders (H00-H59), which categories?
   [x] H35: Other retinal disorders        ← LLM selects
   ...

Step 3a: Within E11, which manifestations?
   [x] E11.3: With ophthalmic complications  ← LLM selects
   ...

... continues until terminal codes reached ...

Final Output: E11.319 (Type 2 diabetes with unspecified diabetic retinopathy)
```

At each step, the LLM must:
1. Evaluate candidates against clinical context
2. Provide reasoning for selections
3. Handle relationship types (children, codeFirst, codeAlso, useAdditionalCode)

## How It Improves Medical Domain Understanding

### 1. Granular Error Detection

When an LLM fails, TMSP shows *where* it failed:

```
✓ Correctly identified Endocrine chapter
✓ Correctly selected E11 (Type 2 diabetes)
✗ Failed at E11.3x level - selected E11.65 (hyperglycemia) instead of E11.3x (ophthalmic)
```

This pinpoints knowledge gaps: the model understands diabetes exists but misclassifies its manifestations.

### 2. Reasoning Transparency

Each selection includes LLM reasoning:

```json
{
  "batch_id": "E11|children",
  "selected": ["E11.3"],
  "reasoning": "Patient has documented diabetic retinopathy, which is an ophthalmic
               complication. E11.3x covers diabetes with ophthalmic complications."
}
```

Poor reasoning reveals conceptual misunderstandings even when selections are correct.

### 3. Lateral Relationship Handling

ICD-10-CM includes metadata relationships:
- **codeFirst**: Underlying condition should be coded first
- **codeAlso**: Additional codes that commonly co-occur
- **useAdditionalCode**: Supplementary codes for complete picture

TMSP evaluates whether LLMs correctly identify these relationships:

```
E11.3 has useAdditionalCode → H35.x (retinal disorders)
Does the LLM select both the diabetes code AND the eye manifestation code?
```

### 4. Specificity Progression

Medical coding requires drilling down to the most specific applicable code. TMSP tracks whether models:
- Stop too early (insufficiently specific)
- Go too deep (overly specific without evidence)
- Handle 7th character requirements (laterality, encounter type)

### 5. Consistency Testing

The caching system ensures identical contexts produce identical selections. Running the same clinical note multiple times reveals:
- Temperature-dependent variability
- Context window sensitivity
- Provider-specific biases

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Clinical Note  │────▶│  Agent/Burr      │────▶│  ICD-10-CM      │
│                 │     │  Orchestration   │     │  Index          │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌─────────┐  ┌─────────┐  ┌─────────┐
              │ children│  │codeFirst│  │codeAlso │  ... parallel batches
              └────┬────┘  └────┬────┘  └────┬────┘
                   │            │            │
                   ▼            ▼            ▼
              ┌─────────────────────────────────┐
              │   Candidate Selector (LLM)      │
              │   - Structured output           │
              │   - Multi-provider support      │
              │   - Reasoning capture           │
              └─────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────────┐
              │   Final Codes + Decision Tree   │
              └─────────────────────────────────┘
```

### Components

| Module | Purpose |
|--------|---------|
| `agent/` | Burr-based state machine for traversal orchestration |
| `candidate_selector/` | LLM integration with structured outputs |
| `graph/` | ICD-10-CM index and hierarchy utilities |
| `server/` | FastAPI backend with SSE streaming |
| `frontend/` | React visualization of traversal trees |

## Installation

```bash
# Core dependencies
uv sync

# With server (FastAPI + uvicorn)
uv sync --extra server
```

## Usage

### Run the Server

```bash
uv run tmsp-server
# or
uv run python -m server
```

### Programmatic API

```python
from agent import run_traversal

result = await run_traversal(
    clinical_note="Patient with type 2 diabetes and chronic kidney disease stage 3",
    provider="openai",
    api_key="sk-...",
    selector="llm",
)

print(result["final_nodes"])  # ['E11.65', 'N18.3']
print(result["batch_data"])   # Full decision tree with reasoning
```

## Supported LLM Providers

| Provider | Structured Output | Notes |
|----------|-------------------|-------|
| OpenAI | strict=true | Full JSON schema compliance |
| Anthropic | json_schema | Via anthropic-beta header |
| Cerebras | strict=true | 5000 char schema limit |
| SambaNova | best-effort | No strict mode |

More to come...

## Evaluation Metrics

TMSP enables measurement of:

- **Path Accuracy**: Did traversal reach correct terminal codes?
- **Step Accuracy**: Correct selections at each hierarchy level
- **Completeness**: All applicable codes identified (including lateral)
- **Reasoning Quality**: Clinical logic in selection explanations
- **Consistency**: Same input → same output across runs

## Retry and Feedback

Checkpoint persistence allows targeted correction:

```python
from agent import retry_node

# Retry from specific decision point with feedback
result = await retry_node(
    batch_id="E11|children",
    feedback_map={
        "E11|children": "Select E11.3 for ophthalmic complications, not E11.65"
    }
)
```

This tests whether LLMs can incorporate corrective feedback—a key capability for medical AI systems.

