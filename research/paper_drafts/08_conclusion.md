# Conclusion

DABE studies tokenization as a learned rate-distortion problem over fixed text chunks. The strongest result is not obtained by changing token-window geometry, but by keeping a stable 64-token chunk and adapting a sparse lexical repair budget over it.

The matched comparison between EXP-093/095 and EXP-092 is the central result: a healthy fixed-rate no-lookup learned tokenizer at `20.0` bits/token reaches `0.72045` token accuracy and `17.89119` mean chunk deviation, and a longer convergence-defense run improves to `0.77324` and `14.51295`. Cost-aware DABE still reaches `0.89274` token accuracy and `6.86454` mean chunk deviation at `20.06207` observed bits/token. This shows that total bit budget and fixed-baseline convergence alone do not explain the gain.

The broader ablation story is consistent. Fixed sparse lookup improves over no-lookup codes; ranked halting recovers much of fixed K=32 quality at lower observed bitrate; the gist-residual architecture improves the repair allocation further; and variable token-window geometry fails to beat fixed chunks plus repair. The architectural implication is clear: for learned tokenizers, adapt the repair budget before adapting the token window.

Future work should connect this tokenizer-autoencoder evidence to full LM training, stronger multi-domain evaluation, and variance-controlled runs. The immediate next step is not another architecture detour, but a paper-quality implementation of the current evidence stack.
