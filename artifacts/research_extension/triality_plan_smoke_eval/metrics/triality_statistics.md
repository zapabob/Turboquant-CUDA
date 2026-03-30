| metric | view | mode | test | alternative | n_pairs | statistic | p_value | candidate_mean | baseline_mean | delta_candidate_minus_baseline | p_value_holm | significant_0_05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hidden_cosine_similarity | vector | key_only_block_so8_triality_vector | wilcoxon | greater | 8 | 6.0 | 0.5 | 0.99951171875 | 0.99951171875 | 0.0 | 1.0 | False |
| hidden_cosine_similarity | spinor_plus_proxy | key_only_block_so8_triality_plus | wilcoxon | greater | 8 | 3.0 | 0.625 | 0.9990234375 | 0.99951171875 | -0.00048828125 | 1.0 | False |
| hidden_cosine_similarity | spinor_minus_proxy | key_only_block_so8_triality_minus | wilcoxon | greater | 8 | 6.0 | 0.71875 | 0.9990234375 | 0.99951171875 | -0.00048828125 | 1.0 | False |
| next_logit_kl | vector | key_only_block_so8_triality_vector | wilcoxon | less | 8 | 23.0 | 0.76953125 | 1.105118821659359e-20 | 1.4842867923237224e-20 | -3.7916797066436345e-21 | 1.0 | False |
| next_logit_kl | spinor_plus_proxy | key_only_block_so8_triality_plus | wilcoxon | less | 8 | 16.0 | 0.421875 | -2.5946901894006246e-20 | 1.4842867923237224e-20 | -4.078976981724347e-20 | 1.0 | False |
| next_logit_kl | spinor_minus_proxy | key_only_block_so8_triality_minus | wilcoxon | less | 8 | 24.0 | 0.80859375 | 4.1961669921875e-05 | 1.4842867923237224e-20 | 4.1961669921874986e-05 | 1.0 | False |