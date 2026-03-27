| metric | view | mode | test | alternative | n_pairs | statistic | p_value | candidate_mean | baseline_mean | delta_candidate_minus_baseline | p_value_holm | significant_0_05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hidden_cosine_similarity | vector | key_only_block_so8_triality_vector | wilcoxon | greater | 4 | 1.0 | 0.5 | 0.9990234375 | 0.99609375 | 0.0029296875 | 0.5 | False |
| hidden_cosine_similarity | spinor_plus_proxy | key_only_block_so8_triality_plus | wilcoxon | greater | 4 | 6.0 | 0.125 | 1.001953125 | 0.99609375 | 0.005859375 | 0.375 | False |
| hidden_cosine_similarity | spinor_minus_proxy | key_only_block_so8_triality_minus | wilcoxon | greater | 4 | 6.0 | 0.125 | 1.001953125 | 0.99609375 | 0.005859375 | 0.375 | False |
| next_logit_kl | vector | key_only_block_so8_triality_vector | wilcoxon | less | 4 | 4.0 | 0.4375 | -2.1347448108436098e-20 | 5.5 | -5.5 | 0.9375 | False |
| next_logit_kl | spinor_plus_proxy | key_only_block_so8_triality_plus | wilcoxon | less | 4 | 3.0 | 0.3125 | 4.743425144213564e-20 | 5.5 | -5.5 | 0.9375 | False |
| next_logit_kl | spinor_minus_proxy | key_only_block_so8_triality_minus | wilcoxon | less | 4 | 3.0 | 0.3125 | 5.373497434366859e-21 | 5.5 | -5.5 | 0.9375 | False |