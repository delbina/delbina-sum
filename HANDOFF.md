Projects: delbina (PyPI 0.5.0 LIVE) + delbina-sum (Qwen2.5-1.5B LoRA summarizer). Paths C:\Users\ksbms\VSCODE\{delbina,delbina-sum}.
Released: HF public: delbina/delbina-sum main=2bee5aa859161d5a5b0ad73bf26deeeafa24b683 (card update; weights = ff16f9a, v5-1.5b), delbina/delbina-sum-GGUF main=394a733f21a2c0fba52c0f9927890505c039e07c (Q4_K_M default, Q8_0; both pass all 11 bars test+holdout), delbina/delbina-sum-data. delbina 0.5.0: DELBINA_SUM, ollama_summarizer temp 0 + num_ctx 8192. All labelled experimental.
Decisions: no 3B (license + size). Real data = v0.6: eval set first, train after. Holdout used; new training needs fresh holdout.
Known issues (documented on both HF cards): ~0.4% wrong-project facts; invented "not known yet" Context notes on tiny chats.
Open: cleanup temp folders.
v0.6: real-chat eval set (50-100), wrong-project bar, fact-check Context lines, CPU-only speed test.
