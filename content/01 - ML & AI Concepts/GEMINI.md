# Antigravity Agent Guidelines

This file documents the user's preferences for note creation and maintenance within this Obsidian vault, specifically for the `01 - ML & AI Concepts` directory and beyond. All agents must adhere to these guidelines to ensure consistency.

## 1. Core Philosophy
- **Exhaustive but Accessible**: The goal is a comprehensive learning vault. Topics should be covered deeply enough to understand the *mechanics* and *intuition*, but not so deep as to become dry academic research papers.
- **Intuition First**: Always prioritize the "mental model" or intuition behind a concept before diving into math or code. Use analogies for complex concepts where necessary.
- **Interconnected**: Usage of internal links (`[[Concept Name]]`) is mandatory. Notes should not exist in isolation. Connect new concepts to existing foundational ones.

## 2. Mathematical Depth
- **Preference**: **Deep Dive without the Fluff**.
- **Do**:
    - Include core definitions, objective functions, and transformations (e.g., SVM primal problem, LoRA update equation).
    - Explain the *purpose* of specific terms in an equation (e.g., "Why divide by $r$?").
    - Use LaTeX for all math (`$$` for blocks, `$` for inline).
- **Do Not**:
    - Include full formal proofs (unless essential for understanding the result).
    - Use undefined notation.
    - Present math without a plain-English explanation of what it represents.

## 3. Note Structure
Most "Concept" notes should follow this template. 



### Standard Sections
*Note: The following sections represent the preferred order. Items marked **(Mandatory)** must always be present. Other sections (Key Ideas, Math, etc.) should be included only when relevant to the content.*

1.  **Title** **(Mandatory)**: `# Title`
2.  **Overview** **(Mandatory)**: Brief summary.
3.  **Key Ideas / Intuition**:
    - Bullet points or diagrams (ASCII/Mermaid) explaining the mental model.
    - "Visual Understanding" is highly valued.
    - Feel free to add an "Intuition" section wherever you find it helpful for visualizing the concept.
4.  **Mathematical Foundation** (if applicable):
    - The core equations.
    - Dimensional analysis (e.g., matrix shapes).
5.  **Practical Application**:
    - "When to Use" / "When NOT to Use".
    - "Common Pitfalls" (e.g., Feature scaling in SVMs).
    - **Calculations/Trade-offs** (e.g., Memory usage, Speed).
6.  **Comparisons**:
    - Tables comparing this concept to siblings (e.g., SVM vs Logistic Regression).
7.  **Resources** **(Mandatory if relevant)**:
    - **Papers**: Links to Arxiv.
    - **Articles**: Tutorials/Blogs.
    - **Videos**: YouTube links.
    - **Code**: Links to repos or small snippets.
8.  **Personal Notes**: A reserved section for the user's insights.
9.  **Progress Checklist** **(Mandatory)**:
    Add this exact checklist to every note.
    - Standard list tracking: `Read overview`, `Understand key concepts`, `Review math`, `Hands-on practice`, `Can explain to others`.
10. **Navigation** **(Mandatory)**: `**Back to**: [[Index Page Name]]`

## 4. Specific Preferences
- **Diagrams**: If a concept is spatial or structural, attempt an ASCII diagram or describe the visual clearly.
- **References**: Always add a `## Resources` section. The user likes to have references present as much as possible.
- **Tone**: Educational, precise, but conversational enough to be readable.
