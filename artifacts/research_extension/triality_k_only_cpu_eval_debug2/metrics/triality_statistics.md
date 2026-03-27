| metric | view | mode | test | alternative | n_pairs | statistic | p_value | candidate_mean | baseline_mean | delta_candidate_minus_baseline | p_value_holm | significant_0_05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hidden_cosine_similarity | vector | key_only_block_so8_triality_vector | wilcoxon | greater | 4 | 2.0 | 0.5 | 0.9970703125 | 0.99609375 | 0.0009765625 | 1.0 | False |
| hidden_cosine_similarity | spinor_plus_proxy | key_only_block_so8_triality_plus | wilcoxon | greater | 4 | 4.0 | 0.375 | 0.998046875 | 0.99609375 | 0.001953125 | 1.0 | False |
| hidden_cosine_similarity | spinor_minus_proxy | key_only_block_so8_triality_minus | wilcoxon | greater | 4 | 6.0 | 0.5 | 0.9970703125 | 0.99609375 | 0.0009765625 | 1.0 | False |
| next_logit_kl | vector | key_only_block_so8_triality_vector | wilcoxon | less | 4 | 6.0 | 0.6875 | 7.821812238652222e-20 | 5.5 | -5.5 | 1.0 | False |
| next_logit_kl | spinor_plus_proxy | key_only_block_so8_triality_plus | wilcoxon | less | 4 | 3.0 | 0.3125 | -5.747885296696238e-20 | 5.5 | -5.5 | 0.9375 | False |
| next_logit_kl | spinor_minus_proxy | key_only_block_so8_triality_minus | wilcoxon | less | 4 | 5.0 | 0.5625 | 0.0311279296875 | 5.5 | -5.4688720703125 | 1.0 | False |