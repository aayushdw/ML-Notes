// Detect paragraphs that contain ONLY KaTeX math (no sibling text) and mark them
// for block display. This handles cases where $$...$$ is rendered as inline math
// inside a <p> tag without the .katex-display class.

function markBlockMathParagraphs() {
    const paragraphs = document.querySelectorAll("article p")

    for (const p of paragraphs) {
        // Check if paragraph has exactly one .katex child element
        const katexElements = p.querySelectorAll(":scope > .katex")
        if (katexElements.length !== 1) continue

        // Check if there's any non-whitespace text content outside the katex element
        let hasNonMathContent = false
        for (const child of p.childNodes) {
            if (child.nodeType === Node.TEXT_NODE) {
                if (child.textContent?.trim()) {
                    hasNonMathContent = true
                    break
                }
            } else if (child.nodeType === Node.ELEMENT_NODE) {
                const el = child as Element
                if (!el.classList.contains("katex")) {
                    hasNonMathContent = true
                    break
                }
            }
        }

        // If the paragraph only contains the katex element (no meaningful text),
        // mark it as a block math paragraph for centering
        if (!hasNonMathContent) {
            p.classList.add("katex-block-paragraph")
        }
    }
}

document.addEventListener("nav", markBlockMathParagraphs)
