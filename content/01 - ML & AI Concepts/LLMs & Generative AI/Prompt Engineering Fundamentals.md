# Prompt Engineering Fundamentals

## Overview
<!-- Brief explanation of the concept (2-3 sentences) -->

## Basic Prompting Principles
The foundation of effective prompt engineering lies in clear communication. This includes:
- **Clarity & Specificity**: Being precise about what you want rather than vague. "Summarize this article in 3 bullet points" vs. "Tell me about this article."
- **Positive Instructions**: Stating what you want rather than what you don't want. Models respond better to "Be concise" than "Don't be verbose."
- **Context Provision**: Giving the model necessary background information to understand the task properly.
- **Task Decomposition**: Breaking complex requests into smaller, manageable parts.
- **Assumed Knowledge**: Understanding what the model knows vs. what you need to explain

## Prompt Structure & Formatting

How you organize information significantly impacts model performance:
- **Delimiters**: Using XML tags (`<instructions>`, `<example>`), markdown, or triple quotes to separate different sections of your prompt.
- **Hierarchical Organization**: Using headers, subheaders, and nested structures to create clear information architecture.
- **Separation of Concerns**: Clearly distinguishing between instructions, context, examples, and the actual task/input. (TODO)
- **Visual Clarity**: Strategic use of whitespace, line breaks, and formatting to improve readability.
- **Structured Inputs**: Using tables, lists, or JSON for data-heavy prompts.

## Few-Shot & Zero-Shot Prompting
- **Zero-Shot**: Direct instructions without examples. Best for straightforward tasks or when the model already understands the domain well.
- **One-Shot**: A single example to clarify expectations. Useful for format specification.
- **Few-Shot**: Multiple examples (typically 2-5) to establish patterns. Essential for nuanced tasks, specific styles, or edge cases.

**Example Ordering**: Later examples often have more influence on the model's behavior.

**Balancing Act**: You might need to iterate over when examples help vs. when they constrain creativity or waste context.

## Chain-of-Thought (CoT) Reasoning

Encouraging explicit reasoning for better accuracy, particularly on math, logic and multi-step problems.
- **Explicit CoT**: Directly asking "Let's think step by step" or "Show your reasoning."
- **Implicit CoT**: Providing examples that include reasoning steps, which the model then mimics.
- **Benefits**: Transparency in decision-making; easier debugging.
- **Trade-offs**: Longer responses, increased token usage, unnecessary for simple tasks.

## Tree-of-Thought Reasoning

Tree-of-Thought is particularly powerful for:
- **Complex planning problems** (multi-step with many possible approaches)
- **Creative problem-solving** (where multiple solutions exist)
- **Constraint satisfaction** (puzzles, optimization, scheduling)
- **Strategic decision-making** (chess moves, game theory)
- **Problems with dead ends**
- **Tasks requiring backtracking**

However this comes at a cost of being slow and token-heavy.

Example Prompt:
```
Prompt: "Use tree-of-thought reasoning to plan a 3-day tech conference for 200 
people with a $50,000 budget. Explore multiple allocation strategies, evaluate 
trade-offs, and arrive at the optimal plan.

At each decision point:
1. Identify the decision to be made
2. Generate 2-3 different approaches
3. Evaluate pros/cons of each
4. Score each approach (1-10)
5. Choose the highest-scoring approach and explain why
6. Proceed to next decision with that approach"
```


## Self Consistency
Self-consistency improves answer accuracy and reliability by **generating multiple independent reasoning paths** and then selecting the most consistent answer through voting or aggregation.

**Key Requirements**:
1. **Independence**: Each reasoning path should be generated independently (not seeing previous attempts)
2. **Diversity**: Encourage different reasoning approaches, not just repetition
3. **Aggregation Method**: A way to combine multiple answers (voting, averaging, etc.)

Example Prompt:
```
Prompt: "Solve this logic puzzle using 3 different reasoning approaches, then 
cross-verify:

Puzzle: Five houses in a row, each a different color. The British man lives 
in the red house. The Swedish man has a dog. The Danish man drinks tea. 
The green house is immediately left of the white house. The owner of the 
green house drinks coffee. Who owns the fish?

Approach 1: Use constraint satisfaction (systematically eliminate possibilities)
Approach 2: Use forward chaining (start with definite facts, build outward)
Approach 3: Use backward chaining (assume solutions, test consistency)

After all three, compare answers."
```

### Ideal Use Cases:
1. Math and Logic Problems
2. Factual Questions with Reasoning
3. Ambiguous Interpretation
4. Complex Multi-Step Planning
5. Code Debugging
6. Creative Tasks with Evaluation
###  Poor Use Cases:
1. Pure Creative Generation: Diversity is the goal, not consistency
2. Subjective Preferences
3. Simple Factual Lookups

## Role Prompting & Personas

Shaping responses through identity and expertise:
- **Expert Roles**: "You are an expert neuroscientist..." to access deeper domain knowledge and appropriate terminology.
- **Audience-Aware Roles**: "Explain as if to a 5-year-old" vs. "Explain to a graduate student."
- **Personality Traits**: Professional, casual, empathetic, analytical, creative, etc.
- **Behavioral Guidelines**: How the persona should approach problems, what they prioritize, their communication style.

### Decision Framework: Is Role Prompting Actually Helping?

```
                    START
                      |
        Does the task require specialized
        knowledge or methodology?
                      |
                 YES / NO
                /         \
            YES            NO → Skip role prompting
              |
        Does the model likely have
        this knowledge in training data?
              |
         YES / NO
        /         \
    YES            NO → Role won't help
      |                  (might cause hallucination)
      |
    Does the role affect:
    - Technical depth?
    - Communication style?
    - Analytical framework?
    - Terminology choice?
      |
 YES / NO
  |        \
YES         NO → Role is just theater
  |
Use role prompting ✓
```

## Context Management & Memory
- [[Context Window Fundamentals]]
- **External Memory Techniques**: Summarization, RAG, storing and referencing key information.
- **Context Priming**: Setting up important information early in the conversation.
- **Reset Strategies**: When and how to clear context to avoid confusion or drift.

## Negative Prompting & Constraints

Understanding what doesn't work and why:
- **Common Pitfalls**: Ambiguous pronouns, conflicting instructions, implicit assumptions, over-complexity.
- **Jailbreaking Awareness**: Understanding but not exploiting model limitations and safety features.
- **Constraint Setting**: "Do not include...", "Avoid...", "Must not..." (though positive framing is often better).
- **Boundary Definition**: Scope limitations, topic restrictions, content guidelines.

## Other Techniques
Sophisticated methods for complex scenarios:
- **Prompt Chaining**: Breaking tasks into sequential prompts where each output feeds into the next.
- **[[ReAct Pattern]]** (Reasoning + Acting): Interleaving reasoning with tool use or information retrieval.
- **Meta-Prompting**: Prompts that generate or improve other prompts.

## Testing & Iteration
Systematic improvement of prompt performance:
- **Version Control**: Tracking prompt iterations and their performance.
- **Benchmarking**: Testing against standard datasets or creating custom test suites.
- **User Feedback Integration**: Incorporating real-world usage patterns and failures.
- **Iterative Refinement**: The cycle of test → analyze → modify → retest.


---

**Progress**: 
- [x] Read overview materials
- [x] Understand key concepts
- [ ] Review mathematical foundations
- [ ] Study implementations
- [ ] Complete hands-on practice
- [x] Can explain to others

**Status Options**: `not-started` | `in-progress` | `completed` | `review-needed`
**Difficulty Options**: `beginner` | `intermediate` | `advanced` | `expert`

---
**Back to**: [[ML & AI Index]]
