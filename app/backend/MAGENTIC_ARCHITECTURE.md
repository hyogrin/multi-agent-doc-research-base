# MagenticExecutor Architecture Diagrams

## 1. High-Level Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         User Request                                     │
│                    "Research complex topic..."                           │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              PlanSearchOrchestratorAFW                                   │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │  1. IntentAnalyzerExecutor                                     │     │
│  │     • Analyzes user intent                                     │     │
│  │     • Enriches query                                           │     │
│  └──────────────────────────┬─────────────────────────────────────┘     │
│                             │                                            │
│  ┌──────────────────────────▼─────────────────────────────────────┐    │
│  │  2. TaskPlannerExecutor                                         │    │
│  │     • Creates search plan                                       │    │
│  │     • Identifies sub-topics                                     │    │
│  └──────────────────────────┬─────────────────────────────────────┘    │
│                             │                                            │
│  ┌──────────────────────────▼─────────────────────────────────────┐    │
│  │  3. SearchExecutors (optional)                                  │    │
│  │     • WebSearchExecutor                                         │    │
│  │     • AISearchExecutor                                          │    │
│  │     • YouTubeMCPExecutor                                        │    │
│  └──────────────────────────┬─────────────────────────────────────┘    │
│                             │                                            │
│  ┌──────────────────────────▼─────────────────────────────────────┐    │
│  │  4. MagenticExecutor (if multi_agent_type="afw_magentic")      │    │
│  │     • Intelligent multi-agent orchestration                     │    │
│  │     • Final executor (no ResponseGenerator after this)          │    │
│  └──────────────────────────┬─────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Streaming Markdown Output                              │
│                   • Progress updates                                     │
│                   • Research results                                     │
│                   • Citations                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. MagenticExecutor Internal Architecture (3-Agent Pattern)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MagenticExecutor                                  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │              run_magentic_research Handler                      │   │
│  │  • Receives research_data with sub_topics                       │   │
│  │  • Extracts contexts (Web, AI Search, YouTube)                  │   │
│  │  • Processes each sub_topic sequentially                        │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                            │
│                             │ For each sub_topic                         │
│                             ▼                                            │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │        _execute_magentic_sub_topic Method                       │   │
│  │                                                                  │   │
│  │  1. Prepare context (truncate to 15K chars)                     │   │
│  │  2. Create specialized agents (3 agents)                        │   │
│  │  3. Build Magentic workflow                                     │   │
│  │  4. Execute with streaming callbacks                            │   │
│  │  5. Parse and return results                                    │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                            │
│                             ▼                                            │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │              MagenticBuilder Workflow (3 Agents)                │   │
│  │                                                                  │   │
│  │         ┌─────────────────────────────────┐                     │   │
│  │         │   Orchestrator (Manager Agent)   │                    │   │
│  │         │   • Planning                     │                    │   │
│  │         │   • Coordination                 │                    │   │
│  │         │   • Progress tracking            │                    │   │
│  │         │   • Quality assurance            │                    │   │
│  │         └──────┬──────────┬────────┬───────┘                    │   │
│  │                │          │        │                             │   │
│  │       ┌────────▼──┐  ┌───▼──────┐ ┌▼─────────────┐             │   │
│  │       │ResearchA..│  │Research..│ │Research..    │             │   │
│  │       │Analyst    │  │Writer    │ │Reviewer      │             │   │
│  │       │           │  │          │ │              │             │   │
│  │       │•Synthesize│  │•Generate │ │•Validate     │             │   │
│  │       │•Analyze   │  │•Structure│ │•Score        │             │   │
│  │       │•Extract   │  │•Format   │ │•Approve      │             │   │
│  │       │•Sources   │  │•Cite     │ │•Revise       │             │   │
│  │       └───────────┘  └──────────┘ └──────────────┘             │   │
│  │                                                                  │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │              Streaming Callbacks                                │   │
│  │                                                                  │   │
│  │  • MagenticOrchestratorMessageEvent                             │   │
│  │    → Planning messages, coordination updates                    │   │
│  │                                                                  │   │
│  │  • MagenticAgentDeltaEvent                                      │   │
│  │    → Token-by-token streaming (TTFT tracking)                   │   │
│  │                                                                  │   │
│  │  • MagenticAgentMessageEvent                                    │   │
│  │    → Complete agent responses                                   │   │
│  │                                                                  │   │
│  │  • MagenticFinalResultEvent                                     │   │
│  │    → Final orchestration result                                 │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Output per Sub-topic                              │
│  {                                                                       │
│    "status": "success",                                                  │
│    "sub_topic": "Topic name",                                            │
│    "final_answer": "Markdown content...",                                │
│    "citations": [...],                                                   │
│    "orchestration_rounds": 5                                             │
│  }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3. Comparison: GroupChat vs Magentic Flow

### GroupChattingExecutor Flow
```
User Query
    ↓
┌───────────────────────────────┐
│  Writer Agent                 │
│  "Generate initial draft"     │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│  Reviewer Agent               │
│  "Review and provide feedback"│
└───────────────┬───────────────┘
                ↓
         (Is approved?)
         /            \
       No              Yes
       ↓                ↓
┌──────────────┐   ┌──────────┐
│Writer revises│   │  Done!   │
└──────┬───────┘   └──────────┘
       ↓
  (Repeat cycle)
```

### MagenticExecutor Flow
```
User Query
    ↓
┌────────────────────────────────────┐
│  Orchestrator                      │
│  "Analyze task, plan approach"     │
└───────────────┬────────────────────┘
                ↓
         (Planning Phase)
                ↓
┌────────────────────────────────────┐
│  Dynamic Task Decomposition        │
│  • Identify required expertise     │
│  • Assign agents adaptively        │
└───────────────┬────────────────────┘
                ↓
         (Execution Phase)
         /              \
        ↓                ↓
┌──────────────┐   ┌──────────────┐
│Researcher    │   │Writer        │
│(as needed)   │   │(as needed)   │
└──────┬───────┘   └──────┬───────┘
       │                   │
       └──────────┬────────┘
                  ↓
           (Orchestrator)
          "Evaluate progress"
                  ↓
          (Is complete?)
          /           \
        No             Yes
        ↓               ↓
   (Adapt plan)    ┌──────┐
        ↓          │ Done!│
   (Next round)    └──────┘
```

## 4. Event Flow Timeline

```
Time  │ Event                              │ Who           │ Action
──────┼────────────────────────────────────┼───────────────┼─────────────────
T0    │ run_magentic_research              │ Executor      │ Start processing
      │                                     │               │
T1    │ Sub-topic extraction               │ Executor      │ Parse input data
      │                                     │               │
T2    │ Context aggregation                │ Executor      │ Combine sources
      │                                     │               │
T3    │ Workflow build                     │ MagenticBuilder│ Create workflow
      │                                     │               │
T4    │ MagenticOrchestratorMessageEvent   │ Orchestrator  │ "Planning..."
      │                                     │               │
T5    │ MagenticAgentDeltaEvent (TTFT)     │ ResearchAnalyst│ First token
      │                                     │               │
T6-10 │ MagenticAgentDeltaEvent (stream)   │ ResearchAnalyst│ Streaming...
      │                                     │               │
T11   │ MagenticAgentMessageEvent          │ ResearchAnalyst│ Response done
      │                                     │               │
T12   │ MagenticOrchestratorMessageEvent   │ Orchestrator  │ "Next phase..."
      │                                     │               │
T13   │ MagenticAgentDeltaEvent            │ ResearchWriter│ First token
      │                                     │               │
T14-20│ MagenticAgentDeltaEvent (stream)   │ ResearchWriter│ Streaming...
      │                                     │               │
T21   │ MagenticAgentMessageEvent          │ ResearchWriter│ Response done
      │                                     │               │
T22   │ MagenticFinalResultEvent           │ Orchestrator  │ Final result
      │                                     │               │
T23   │ Parse and format output            │ Executor      │ JSON → Markdown
      │                                     │               │
T24   │ Stream to user                     │ Executor      │ Chunk by chunk
      │                                     │               │
T25   │ Complete                           │ Executor      │ Return result
```

## 5. Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Input Data                                     │
│                                                                          │
│  research_data = {                                                       │
│    "question": "Main question",                                          │
│    "sub_topics": [                                                       │
│      {                                                                   │
│        "sub_topic": "Topic 1",                                           │
│        "queries": ["query1", "query2"]                                   │
│      }                                                                   │
│    ],                                                                    │
│    "sub_topic_web_contexts": {...},      ← Web search results           │
│    "sub_topic_ai_search_contexts": {...}, ← AI Search documents          │
│    "sub_topic_youtube_contexts": {...},   ← YouTube videos               │
│    "metadata": {"locale": "en-US", ...}                                  │
│  }                                                                       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Context Processing                                  │
│                                                                          │
│  • Extract contexts per sub-topic                                        │
│  • Combine from multiple sources                                         │
│  • Format: "[Source] Title\nContent\nSource: URL"                        │
│  • Truncate to MAX_CONTEXT_CHARS (15,000)                                │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Agent Instructions                                  │
│                                                                          │
│  ResearchAnalyst:                    ResearchWriter:                     │
│  ├─ Current date: 2025-10-21        ├─ Current date: 2025-10-21        │
│  ├─ Locale: en-US                   ├─ Locale: en-US                   │
│  ├─ Topic: "Topic 1"                ├─ Topic: "Topic 1"                │
│  ├─ Question: "query1, query2"      ├─ Question: "query1, query2"      │
│  └─ Context: [Truncated 15K]        └─ Context: [Truncated 15K]        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Magentic Orchestration                                 │
│                                                                          │
│  Round 1: Orchestrator → ResearchAnalyst                                 │
│    Output: "Key insights: ..."                                           │
│                                                                          │
│  Round 2: Orchestrator → ResearchWriter                                  │
│    Output: {"draft_answer_markdown": "...", "citations": [...]}          │
│                                                                          │
│  Round 3: Orchestrator → Evaluation                                      │
│    Decision: Complete (quality sufficient)                               │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Output Processing                                   │
│                                                                          │
│  • Parse JSON output                                                     │
│  • Extract draft_answer_markdown                                         │
│  • Extract citations                                                     │
│  • Format for streaming                                                  │
│  • Add sub-topic headers                                                 │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Streaming Output                                     │
│                                                                          │
│  ## Topic 1 🎯 (Rounds: 3)                                               │
│                                                                          │
│  [Markdown content streamed in chunks...]                                │
│                                                                          │
│  - Key finding 1                                                         │
│  - Key finding 2                                                         │
│                                                                          │
│  **Sources:**                                                            │
│  1. [Source 1](URL)                                                      │
│  2. [Source 2](URL)                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## 6. Decision Tree: Which Pattern to Use?

```
                        Need multi-agent research?
                                 │
                    ┌────────────┴────────────┐
                   Yes                        No
                    │                          │
                    ▼                          ▼
            Task complexity?              Use single agent
                    │
        ┌───────────┼───────────┐
     Simple      Medium      Complex
        │           │           │
        ▼           ▼           ▼
   GroupChat   (Both work)  Magentic
                    │
            ┌───────┴───────┐
         Speed?          Quality?
            │               │
            ▼               ▼
       GroupChat        Magentic
```

### Decision Factors

**Choose GroupChattingExecutor if:**
```
✅ Task = iterative refinement
✅ Speed > quality
✅ Token budget = tight
✅ Workflow = predictable
✅ Rounds = 3-5 sufficient
```

**Choose MagenticExecutor if:**
```
✅ Task = complex research
✅ Quality > speed
✅ Token budget = flexible
✅ Workflow = adaptive
✅ Rounds = variable (1-5+)
```

## 7. Performance Characteristics

```
Metric                    GroupChattingExecutor    MagenticExecutor
────────────────────────────────────────────────────────────────────
Response Time/Sub-topic        30-60 seconds         45-90 seconds
Token Usage/Sub-topic         3,000-5,000           5,000-8,000
Number of Rounds                   3-5                  1-5+
Quality Score                   ⭐⭐⭐⭐             ⭐⭐⭐⭐⭐
Adaptability                      Low                  High
Orchestration Overhead            None              10-20% tokens
TTFT (Time to First Token)      ~5 seconds           ~8 seconds
Streaming Granularity            Good              Very Good
```

---

## 8. Updated 3-Agent Magentic Pattern

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                   Magentic Orchestrator                          │
│        (Intelligent Planning, Coordination & QA)                 │
│  • Task decomposition and agent selection                        │
│  • Dynamic workflow adaptation                                   │
│  • Quality assurance through reviewer integration                │
└────┬─────────────────┬──────────────────┬──────────────────────┘
     │                 │                  │
     │                 │                  │
┌────▼────────────┐ ┌──▼─────────────┐ ┌─▼────────────────────┐
│ResearchAnalyst  │ │ResearchWriter  │ │ResearchReviewer      │
│                 │ │                │ │                      │
│**Phase 1:**     │ │**Phase 2:**    │ │**Phase 3:**          │
│Information      │ │Content         │ │Quality               │
│Synthesis        │ │Generation      │ │Validation            │
│                 │ │                │ │                      │
│• Extract insights│ │• Structure     │ │• Validate accuracy   │
│• Identify patterns│ │  content      │ │• Check completeness  │
│• Assess sources  │ │• Format markdown│ │• Verify citations   │
│• Note citations  │ │• Include       │ │• Score quality (1-5) │
│                 │ │  citations     │ │• Approve/revise      │
│                 │ │                │ │• Set ready_to_publish│
│**Output:**      │ │**Output:**     │ │**Output:**           │
│Research JSON    │ │Draft JSON      │ │Final JSON            │
│with insights    │ │with markdown   │ │with revised answer   │
└─────────────────┘ └────────────────┘ └──────────────────────┘
```

### Workflow Stages

**Stage 1: Research Analysis (ResearchAnalyst)**
- Input: Raw contexts from searches
- Process: Information extraction, pattern recognition
- Output: Structured research synthesis with key insights and sources

**Stage 2: Content Creation (ResearchWriter)**
- Input: Research synthesis + contexts
- Process: Comprehensive content generation with structure
- Output: Draft answer with citations in markdown format

**Stage 3: Quality Review (ResearchReviewer)**
- Input: Draft answer + original contexts
- Process: Validation, scoring, citation verification
- Output: Final answer with quality score and publish approval

### Key Improvements with 3-Agent Pattern

1. **Better Quality Control**
   - Dedicated reviewer ensures accuracy
   - Citation integrity checking
   - Ready-to-publish flag for confidence

2. **Clearer Separation of Concerns**
   - Analyst focuses on understanding
   - Writer focuses on presentation
   - Reviewer focuses on validation

3. **Enhanced Output Metadata**
   ```python
   {
       "revised_answer_markdown": "...",
       "citations": [...],
       "reviewer_evaluation_score": 4,  # 1-5 scale
       "ready_to_publish": true,
       "major_issues": []
   }
   ```

4. **Adaptive Orchestration**
   - Orchestrator can route back to writer if reviewer finds issues
   - Max 7 rounds (increased from 5 for 3-agent pattern)
   - Automatic quality improvement loop

### Comparison: 2-Agent vs 3-Agent Pattern

| Aspect | 2-Agent (Researcher + Writer) | 3-Agent (+ Reviewer) |
|--------|------------------------------|---------------------|
| **Quality Assurance** | Implicit in writer | Explicit reviewer stage |
| **Citation Validation** | Writer responsibility | Reviewer verifies |
| **Score/Confidence** | Confidence level only | 1-5 evaluation score |
| **Revision Loop** | Limited | Reviewer-driven refinement |
| **Max Rounds** | 5 | 7 |
| **Output Quality** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Token Cost** | Medium-High | High |
| **Best For** | General research | Critical/published content |

### When to Use 3-Agent Pattern

✅ **Recommended for:**
- Published content requiring high accuracy
- Research with critical citation requirements
- Content needing quality scores/metrics
- Multi-source synthesis requiring validation

⚠️ **Consider 2-agent for:**
- Internal documentation
- Rapid prototyping
- Budget-constrained projects
- Simple research tasks

---

These diagrams provide a comprehensive visual understanding of the MagenticExecutor architecture, data flow, and decision-making process, with special emphasis on the enhanced 3-agent pattern for superior quality assurance.
