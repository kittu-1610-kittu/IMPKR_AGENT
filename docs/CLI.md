# Terminal CLI Usage Guide

The `cli.py` interface is designed to let you interact with IMPKR-AGENT directly inside your VS Code terminal.

---

## 🚀 Running the CLI

Run the client using python:
```bash
python cli.py
```

---

## 🎨 Interactive Console Output Features

As the reasoning pipeline executes, the console displays real-time colorized ASCII segments:

1. **Planner Stage**: Shows the planner's query decomposition plan and subtasks.
2. **Parallel Retrieval**: Lists active retrievers and displays the parallel execution time ($T_{parallel}$).
3. **Fusion Repository**: Prints the deduplicated and ranked evidence.
4. **Consensus Iterations**: Highlights draft revisions, critic findings, validation claim reports, and trust assessments.
5. **Grounded Answer**: Displays the finalized response with inline evidence citations.

---

## 🔄 User RLHF Submission

After the answer prints, the CLI prompts you:

```text
Rate answer quality (1 to 5 stars) [default: 5]: 4
Add factual corrections to update Knowledge Graph (optional): component validator was upgraded to depth 2 hops
```

- **Quality Rating**: Calibrates fuser scoring parameters.
- **Factual Corrections**: Written back to the relational tables and the Knowledge Graph dynamically.
