# Attention Replay Summary

- Model: H:\Qwen3.5-9B-official-hf
- Query source: captured

Current mathematical bottleneck: value quantization amplifies attention-output drift more than key quantization alone. At 2 bits, full-KV trails block-SO(8) key-only by 0.0605 hidden-state cosine.
Runtime recommendation: protected-V is promising but not ready.

## Primary Pareto Table

```
                      mode bit_setting                   metric     mean      std      sem  ci95_low  ci95_high
                     exact       exact    memory_ratio_vs_exact 1.000000 0.000000 0.000000  1.000000   1.000000
           key_only_random           2    memory_ratio_vs_exact 0.566406 0.000000 0.000000  0.566406   0.566406
           key_only_random         2.5    memory_ratio_vs_exact 0.574219 0.000000 0.000000  0.574219   0.574219
           key_only_random         3.5    memory_ratio_vs_exact 0.605469 0.000000 0.000000  0.605469   0.605469
           key_only_random           4    memory_ratio_vs_exact 0.628906 0.000000 0.000000  0.628906   0.628906
 key_only_block_so8_static           2    memory_ratio_vs_exact 0.566406 0.000000 0.000000  0.566406   0.566406
 key_only_block_so8_static         2.5    memory_ratio_vs_exact 0.574219 0.000000 0.000000  0.574219   0.574219
 key_only_block_so8_static         3.5    memory_ratio_vs_exact 0.605469 0.000000 0.000000  0.605469   0.605469
 key_only_block_so8_static           4    memory_ratio_vs_exact 0.628906 0.000000 0.000000  0.628906   0.628906
key_only_block_so8_learned           2    memory_ratio_vs_exact 0.566406 0.000000 0.000000  0.566406   0.566406
key_only_block_so8_learned         2.5    memory_ratio_vs_exact 0.574219 0.000000 0.000000  0.574219   0.574219
key_only_block_so8_learned         3.5    memory_ratio_vs_exact 0.605469 0.000000 0.000000  0.605469   0.605469
key_only_block_so8_learned           4    memory_ratio_vs_exact 0.628906 0.000000 0.000000  0.628906   0.628906
               protected_v           2    memory_ratio_vs_exact 0.204352 0.001157 0.000579  0.202510   0.206193
               protected_v         2.5    memory_ratio_vs_exact 0.212164 0.001157 0.000579  0.210323   0.214006
               protected_v         3.5    memory_ratio_vs_exact 0.271490 0.001157 0.000579  0.269649   0.273332
               protected_v           4    memory_ratio_vs_exact 0.323004 0.001157 0.000579  0.321163   0.324845
       protected_v_lowrank           2    memory_ratio_vs_exact 0.212164 0.001157 0.000579  0.210323   0.214006
       protected_v_lowrank         2.5    memory_ratio_vs_exact 0.219977 0.001157 0.000579  0.218135   0.221818
       protected_v_lowrank         3.5    memory_ratio_vs_exact 0.279303 0.001157 0.000579  0.277462   0.281144
       protected_v_lowrank           4    memory_ratio_vs_exact 0.330817 0.001157 0.000579  0.328975   0.332658
                   full_kv           2    memory_ratio_vs_exact 0.130859 0.000000 0.000000  0.130859   0.130859
                   full_kv         2.5    memory_ratio_vs_exact 0.146484 0.000000 0.000000  0.146484   0.146484
                   full_kv         3.5    memory_ratio_vs_exact 0.208984 0.000000 0.000000  0.208984   0.208984
                   full_kv           4    memory_ratio_vs_exact 0.255859 0.000000 0.000000  0.255859   0.255859
                     exact       exact hidden_cosine_similarity 1.000000 0.000000 0.000000  1.000000   1.000000
           key_only_random           2 hidden_cosine_similarity 0.997070 0.003740 0.001870  0.991119   1.003021
           key_only_random         2.5 hidden_cosine_similarity 1.000000 0.006379 0.003189  0.989850   1.010150
           key_only_random         3.5 hidden_cosine_similarity 0.999023 0.001953 0.000977  0.995916   1.002131
           key_only_random           4 hidden_cosine_similarity 0.999023 0.001953 0.000977  0.995916   1.002131
 key_only_block_so8_static           2 hidden_cosine_similarity 1.000000 0.000000 0.000000  1.000000   1.000000
 key_only_block_so8_static         2.5 hidden_cosine_similarity 0.998047 0.009831 0.004915  0.982404   1.013689
 key_only_block_so8_static         3.5 hidden_cosine_similarity 0.999023 0.001953 0.000977  0.995916   1.002131
 key_only_block_so8_static           4 hidden_cosine_similarity 1.000000 0.000000 0.000000  1.000000   1.000000
key_only_block_so8_learned           2 hidden_cosine_similarity 1.000977 0.004915 0.002458  0.993155   1.008798
key_only_block_so8_learned         2.5 hidden_cosine_similarity 0.998047 0.006766 0.003383  0.987281   1.008813
key_only_block_so8_learned         3.5 hidden_cosine_similarity 0.998047 0.003906 0.001953  0.991831   1.004263
key_only_block_so8_learned           4 hidden_cosine_similarity 0.998047 0.003906 0.001953  0.991831   1.004263
               protected_v           2 hidden_cosine_similarity 0.955078 0.005043 0.002521  0.947054   0.963103
               protected_v         2.5 hidden_cosine_similarity 0.957031 0.003189 0.001595  0.951956   0.962106
               protected_v         3.5 hidden_cosine_similarity 0.985352 0.003740 0.001870  0.979400   0.991303
               protected_v           4 hidden_cosine_similarity 0.995117 0.004915 0.002458  0.987296   1.002938
       protected_v_lowrank           2 hidden_cosine_similarity 0.964844 0.003189 0.001595  0.959769   0.969919
       protected_v_lowrank         2.5 hidden_cosine_similarity 0.961914 0.004915 0.002458  0.954093   0.969735
       protected_v_lowrank         3.5 hidden_cosine_similarity 0.985352 0.003740 0.001870  0.979400   0.991303
       protected_v_lowrank           4 hidden_cosine_similarity 0.998047 0.002255 0.001128  0.994458   1.001636
                   full_kv           2 hidden_cosine_similarity 0.940430 0.006671 0.003336  0.929814   0.951045
                   full_kv         2.5 hidden_cosine_similarity 0.958984 0.003906 0.001953  0.952769   0.965200
                   full_kv         3.5 hidden_cosine_similarity 0.987305 0.001953 0.000977  0.984197   0.990413
                   full_kv           4 hidden_cosine_similarity 0.995117 0.001953 0.000977  0.992009   0.998225
                     exact       exact               hidden_mse 0.000000 0.000000 0.000000  0.000000   0.000000
           key_only_random           2               hidden_mse 0.000512 0.000402 0.000201 -0.000127   0.001152
           key_only_random         2.5               hidden_mse 0.000334 0.000177 0.000088  0.000052   0.000615
           key_only_random         3.5               hidden_mse 0.000093 0.000047 0.000024  0.000018   0.000168
           key_only_random           4               hidden_mse 0.000053 0.000064 0.000032 -0.000050   0.000155
 key_only_block_so8_static           2               hidden_mse 0.001321 0.000819 0.000410  0.000018   0.002624
 key_only_block_so8_static         2.5               hidden_mse 0.002679 0.003329 0.001665 -0.002619   0.007977
 key_only_block_so8_static         3.5               hidden_mse 0.000152 0.000133 0.000067 -0.000060   0.000365
 key_only_block_so8_static           4               hidden_mse 0.000102 0.000085 0.000042 -0.000033   0.000237
key_only_block_so8_learned           2               hidden_mse 0.000395 0.000256 0.000128 -0.000012   0.000802
key_only_block_so8_learned         2.5               hidden_mse 0.000635 0.000414 0.000207 -0.000024   0.001294
key_only_block_so8_learned         3.5               hidden_mse 0.000181 0.000168 0.000084 -0.000086   0.000448
key_only_block_so8_learned           4               hidden_mse 0.000077 0.000061 0.000030 -0.000019   0.000173
               protected_v           2               hidden_mse 0.052917 0.012101 0.006051  0.033661   0.072173
               protected_v         2.5               hidden_mse 0.052979 0.012412 0.006206  0.033228   0.072729
               protected_v         3.5               hidden_mse 0.016251 0.003709 0.001855  0.010348   0.022153
               protected_v           4               hidden_mse 0.004528 0.000909 0.000455  0.003081   0.005975
       protected_v_lowrank           2               hidden_mse 0.045288 0.007955 0.003977  0.032630   0.057946
       protected_v_lowrank         2.5               hidden_mse 0.045410 0.008063 0.004031  0.032580   0.058240
       protected_v_lowrank         3.5               hidden_mse 0.014023 0.002315 0.001158  0.010339   0.017707
       protected_v_lowrank           4               hidden_mse 0.003925 0.000618 0.000309  0.002943   0.004908
                   full_kv           2               hidden_mse 0.070251 0.017002 0.008501  0.043198   0.097305
                   full_kv         2.5               hidden_mse 0.049316 0.011845 0.005923  0.030468   0.068165
                   full_kv         3.5               hidden_mse 0.013870 0.003163 0.001582  0.008837   0.018904
                   full_kv           4               hidden_mse 0.005638 0.001316 0.000658  0.003545   0.007731
                     exact       exact  logit_cosine_similarity 1.000000 0.000000 0.000000  1.000000   1.000000
           key_only_random           2  logit_cosine_similarity 0.997070 0.001953 0.000977  0.993962   1.000178
           key_only_random         2.5  logit_cosine_similarity 0.998047 0.003906 0.001953  0.991831   1.004263
           key_only_random         3.5  logit_cosine_similarity 0.997070 0.003740 0.001870  0.991119   1.003021
           key_only_random           4  logit_cosine_similarity 0.998047 0.003906 0.001953  0.991831   1.004263
 key_only_block_so8_static           2  logit_cosine_similarity 0.997070 0.003740 0.001870  0.991119   1.003021
 key_only_block_so8_static         2.5  logit_cosine_similarity 0.998047 0.002255 0.001128  0.994458   1.001636
 key_only_block_so8_static         3.5  logit_cosine_similarity 1.000000 0.000000 0.000000  1.000000   1.000000
 key_only_block_so8_static           4  logit_cosine_similarity 0.999023 0.001953 0.000977  0.995916   1.002131
key_only_block_so8_learned           2  logit_cosine_similarity 0.996094 0.003189 0.001595  0.991019   1.001169
key_only_block_so8_learned         2.5  logit_cosine_similarity 0.997070 0.001953 0.000977  0.993962   1.000178
key_only_block_so8_learned         3.5  logit_cosine_similarity 1.000000 0.000000 0.000000  1.000000   1.000000
key_only_block_so8_learned           4  logit_cosine_similarity 0.999023 0.001953 0.000977  0.995916   1.002131
               protected_v           2  logit_cosine_similarity 0.996094 0.003189 0.001595  0.991019   1.001169
               protected_v         2.5  logit_cosine_similarity 0.997070 0.001953 0.000977  0.993962   1.000178
               protected_v         3.5  logit_cosine_similarity 1.000000 0.000000 0.000000  1.000000   1.000000
               protected_v           4  logit_cosine_similarity 0.999023 0.001953 0.000977  0.995916   1.002131
       protected_v_lowrank           2  logit_cosine_similarity 0.996094 0.003189 0.001595  0.991019   1.001169
       protected_v_lowrank         2.5  logit_cosine_similarity 0.997070 0.001953 0.000977  0.993962   1.000178
       protected_v_lowrank         3.5  logit_cosine_similarity 1.000000 0.000000 0.000000  1.000000   1.000000
       protected_v_lowrank           4  logit_cosine_similarity 0.999023 0.001953 0.000977  0.995916   1.002131
                   full_kv           2  logit_cosine_similarity 0.997070 0.001953 0.000977  0.993962   1.000178
                   full_kv         2.5  logit_cosine_similarity 0.998047 0.003906 0.001953  0.991831   1.004263
                   full_kv         3.5  logit_cosine_similarity 0.997070 0.003740 0.001870  0.991119   1.003021
                   full_kv           4  logit_cosine_similarity 0.998047 0.003906 0.001953  0.991831   1.004263
                     exact       exact         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
           key_only_random           2         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
           key_only_random         2.5         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
           key_only_random         3.5         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
           key_only_random           4         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
 key_only_block_so8_static           2         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
 key_only_block_so8_static         2.5         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
 key_only_block_so8_static         3.5         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
 key_only_block_so8_static           4         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
key_only_block_so8_learned           2         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
key_only_block_so8_learned         2.5         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
key_only_block_so8_learned         3.5         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
key_only_block_so8_learned           4         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
               protected_v           2         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
               protected_v         2.5         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
               protected_v         3.5         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
               protected_v           4         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
       protected_v_lowrank           2         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
       protected_v_lowrank         2.5         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
       protected_v_lowrank         3.5         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
       protected_v_lowrank           4         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
                   full_kv           2         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
                   full_kv         2.5         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
                   full_kv         3.5         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
                   full_kv           4         logit_top1_match 1.000000 0.000000 0.000000  1.000000   1.000000
                     exact       exact       logit_top5_overlap 1.000000 0.000000 0.000000  1.000000   1.000000
           key_only_random           2       logit_top5_overlap 0.834375 0.059839 0.029920  0.739158   0.929592
           key_only_random         2.5       logit_top5_overlap 0.856250 0.052540 0.026270  0.772648   0.939852
           key_only_random         3.5       logit_top5_overlap 0.887500 0.044488 0.022244  0.816710   0.958290
           key_only_random           4       logit_top5_overlap 0.934375 0.023662 0.011831  0.896723   0.972027
 key_only_block_so8_static           2       logit_top5_overlap 0.843750 0.075346 0.037673  0.723857   0.963643
 key_only_block_so8_static         2.5       logit_top5_overlap 0.834375 0.063225 0.031612  0.733770   0.934980
 key_only_block_so8_static         3.5       logit_top5_overlap 0.890625 0.051412 0.025706  0.808816   0.972433
 key_only_block_so8_static           4       logit_top5_overlap 0.906250 0.029756 0.014878  0.858902   0.953598
key_only_block_so8_learned           2       logit_top5_overlap 0.800000 0.044488 0.022244  0.729210   0.870790
key_only_block_so8_learned         2.5       logit_top5_overlap 0.850000 0.030619 0.015309  0.801279   0.898721
key_only_block_so8_learned         3.5       logit_top5_overlap 0.893750 0.042696 0.021348  0.825812   0.961688
key_only_block_so8_learned           4       logit_top5_overlap 0.931250 0.033072 0.016536  0.878625   0.983875
               protected_v           2       logit_top5_overlap 0.800000 0.044488 0.022244  0.729210   0.870790
               protected_v         2.5       logit_top5_overlap 0.850000 0.030619 0.015309  0.801279   0.898721
               protected_v         3.5       logit_top5_overlap 0.893750 0.042696 0.021348  0.825812   0.961688
               protected_v           4       logit_top5_overlap 0.931250 0.033072 0.016536  0.878625   0.983875
       protected_v_lowrank           2       logit_top5_overlap 0.800000 0.044488 0.022244  0.729210   0.870790
       protected_v_lowrank         2.5       logit_top5_overlap 0.850000 0.030619 0.015309  0.801279   0.898721
       protected_v_lowrank         3.5       logit_top5_overlap 0.893750 0.042696 0.021348  0.825812   0.961688
       protected_v_lowrank           4       logit_top5_overlap 0.931250 0.033072 0.016536  0.878625   0.983875
                   full_kv           2       logit_top5_overlap 0.834375 0.059839 0.029920  0.739158   0.929592
                   full_kv         2.5       logit_top5_overlap 0.856250 0.052540 0.026270  0.772648   0.939852
                   full_kv         3.5       logit_top5_overlap 0.887500 0.044488 0.022244  0.816710   0.958290
                   full_kv           4       logit_top5_overlap 0.934375 0.023662 0.011831  0.896723   0.972027
```

## Secondary Runtime Table

```
                      mode bit_setting          metric      mean      std      sem  ci95_low  ci95_high
                     exact       exact prefill_seconds  0.000640 0.000509 0.000254 -0.000170   0.001449
           key_only_random           2 prefill_seconds  0.173820 0.328519 0.164260 -0.348927   0.696567
           key_only_random         2.5 prefill_seconds  0.012515 0.003297 0.001648  0.007270   0.017761
           key_only_random         3.5 prefill_seconds  0.010783 0.003170 0.001585  0.005739   0.015827
           key_only_random           4 prefill_seconds  0.010262 0.005105 0.002553  0.002139   0.018385
 key_only_block_so8_static           2 prefill_seconds  0.023188 0.032077 0.016038 -0.027854   0.074229
 key_only_block_so8_static         2.5 prefill_seconds  0.012040 0.003237 0.001619  0.006889   0.017191
 key_only_block_so8_static         3.5 prefill_seconds  0.011170 0.004151 0.002076  0.004564   0.017775
 key_only_block_so8_static           4 prefill_seconds  0.011457 0.003871 0.001936  0.005297   0.017617
key_only_block_so8_learned           2 prefill_seconds  0.006585 0.003272 0.001636  0.001379   0.011791
key_only_block_so8_learned         2.5 prefill_seconds  0.015370 0.004935 0.002467  0.007517   0.023222
key_only_block_so8_learned         3.5 prefill_seconds  0.012813 0.006040 0.003020  0.003202   0.022423
key_only_block_so8_learned           4 prefill_seconds  0.008920 0.003856 0.001928  0.002783   0.015056
               protected_v           2 prefill_seconds  0.700848 1.370550 0.685275 -1.480003   2.881699
               protected_v         2.5 prefill_seconds  0.022296 0.002812 0.001406  0.017822   0.026770
               protected_v         3.5 prefill_seconds  0.018746 0.006430 0.003215  0.008514   0.028977
               protected_v           4 prefill_seconds  0.017917 0.003580 0.001790  0.012220   0.023614
       protected_v_lowrank           2 prefill_seconds  0.106277 0.099929 0.049964 -0.052732   0.265287
       protected_v_lowrank         2.5 prefill_seconds  0.026394 0.007880 0.003940  0.013856   0.038932
       protected_v_lowrank         3.5 prefill_seconds  0.045239 0.015322 0.007661  0.020858   0.069620
       protected_v_lowrank           4 prefill_seconds  0.025003 0.009097 0.004549  0.010527   0.039479
                   full_kv           2 prefill_seconds  0.015417 0.002808 0.001404  0.010949   0.019886
                   full_kv         2.5 prefill_seconds  0.025555 0.013027 0.006514  0.004826   0.046283
                   full_kv         3.5 prefill_seconds  0.053424 0.052898 0.026449 -0.030749   0.137596
                   full_kv           4 prefill_seconds  0.014265 0.003426 0.001713  0.008815   0.019716
                     exact       exact  decode_seconds  0.032854 0.062734 0.031367 -0.066969   0.132677
           key_only_random           2  decode_seconds  0.005215 0.001630 0.000815  0.002621   0.007808
           key_only_random         2.5  decode_seconds  0.003937 0.000664 0.000332  0.002880   0.004994
           key_only_random         3.5  decode_seconds  0.003728 0.000934 0.000467  0.002241   0.005214
           key_only_random           4  decode_seconds  0.003355 0.001462 0.000731  0.001029   0.005682
 key_only_block_so8_static           2  decode_seconds  0.003754 0.002313 0.001156  0.000074   0.007434
 key_only_block_so8_static         2.5  decode_seconds  0.004765 0.001613 0.000806  0.002199   0.007331
 key_only_block_so8_static         3.5  decode_seconds  0.003948 0.001341 0.000671  0.001814   0.006083
 key_only_block_so8_static           4  decode_seconds  0.003816 0.001471 0.000735  0.001476   0.006156
key_only_block_so8_learned           2  decode_seconds  0.003195 0.002300 0.001150 -0.000465   0.006855
key_only_block_so8_learned         2.5  decode_seconds  0.005054 0.002179 0.001089  0.001588   0.008521
key_only_block_so8_learned         3.5  decode_seconds  0.004770 0.002252 0.001126  0.001186   0.008355
key_only_block_so8_learned           4  decode_seconds  0.002601 0.000770 0.000385  0.001375   0.003826
               protected_v           2  decode_seconds  0.003419 0.002294 0.001147 -0.000232   0.007070
               protected_v         2.5  decode_seconds  0.005918 0.001431 0.000715  0.003642   0.008195
               protected_v         3.5  decode_seconds  0.004726 0.001946 0.000973  0.001630   0.007822
               protected_v           4  decode_seconds  0.003008 0.001315 0.000658  0.000915   0.005101
       protected_v_lowrank           2  decode_seconds  0.002853 0.000968 0.000484  0.001313   0.004394
       protected_v_lowrank         2.5  decode_seconds  0.003194 0.000768 0.000384  0.001971   0.004416
       protected_v_lowrank         3.5  decode_seconds  0.004337 0.001517 0.000758  0.001923   0.006750
       protected_v_lowrank           4  decode_seconds  0.003690 0.001786 0.000893  0.000848   0.006532
                   full_kv           2  decode_seconds  0.003415 0.000699 0.000350  0.002302   0.004528
                   full_kv         2.5  decode_seconds  0.004130 0.001479 0.000740  0.001776   0.006484
                   full_kv         3.5  decode_seconds  0.004816 0.001329 0.000665  0.002701   0.006932
                   full_kv           4  decode_seconds  0.002800 0.000410 0.000205  0.002148   0.003452
                     exact       exact    peak_vram_mb 14.317871 4.068733 2.034367  7.843608  20.792134
           key_only_random           2    peak_vram_mb 15.610352 4.141278 2.070639  9.020654  22.200049
           key_only_random         2.5    peak_vram_mb 17.490112 0.205461 0.102731 17.163178  17.817047
           key_only_random         3.5    peak_vram_mb 17.490112 0.205461 0.102731 17.163178  17.817047
           key_only_random           4    peak_vram_mb 17.641602 0.248527 0.124263 17.246140  18.037063
 key_only_block_so8_static           2    peak_vram_mb 15.610352 4.141278 2.070639  9.020654  22.200049
 key_only_block_so8_static         2.5    peak_vram_mb 17.490112 0.205461 0.102731 17.163178  17.817047
 key_only_block_so8_static         3.5    peak_vram_mb 17.490112 0.205461 0.102731 17.163178  17.817047
 key_only_block_so8_static           4    peak_vram_mb 17.641602 0.248527 0.124263 17.246140  18.037063
key_only_block_so8_learned           2    peak_vram_mb 17.644531 0.248527 0.124263 17.249070  18.039993
key_only_block_so8_learned         2.5    peak_vram_mb 17.493042 0.205461 0.102731 17.166107  17.819977
key_only_block_so8_learned         3.5    peak_vram_mb 17.493042 0.205461 0.102731 17.166107  17.819977
key_only_block_so8_learned           4    peak_vram_mb 17.644531 0.248527 0.124263 17.249070  18.039993
               protected_v           2    peak_vram_mb 17.666504 0.253860 0.126930 17.262556  18.070452
               protected_v         2.5    peak_vram_mb 17.570679 0.226261 0.113130 17.210647  17.930710
               protected_v         3.5    peak_vram_mb 17.570679 0.226261 0.113130 17.210647  17.930710
               protected_v           4    peak_vram_mb 17.666504 0.253860 0.126930 17.262556  18.070452
       protected_v_lowrank           2    peak_vram_mb 21.063477 0.158838 0.079419 20.810729  21.316224
       protected_v_lowrank         2.5    peak_vram_mb 21.064453 0.158838 0.079419 20.811706  21.317200
       protected_v_lowrank         3.5    peak_vram_mb 21.064453 0.158838 0.079419 20.811706  21.317200
       protected_v_lowrank           4    peak_vram_mb 21.063477 0.158838 0.079419 20.810729  21.316224
                   full_kv           2    peak_vram_mb 17.697754 0.263993 0.131996 17.277682  18.117826
                   full_kv         2.5    peak_vram_mb 17.547241 0.220927 0.110464 17.195696  17.898786
                   full_kv         3.5    peak_vram_mb 17.547241 0.220927 0.110464 17.195696  17.898786
                   full_kv           4    peak_vram_mb 17.697754 0.263993 0.131996 17.277682  18.117826
```

## Summary Statistics

```
 dataset                       mode bit_setting  bits                   metric  n          mean           std          sem      ci95_low     ci95_high                  model_id query_source
captured                      exact       exact   NaN           decode_seconds  4      0.032854      0.062734     0.031367     -0.066969      0.132677 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN hidden_cosine_similarity  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN               hidden_mae  4      0.000000      0.000000     0.000000      0.000000      0.000000 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN               hidden_mse  4      0.000000      0.000000     0.000000      0.000000      0.000000 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN  logit_cosine_similarity  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN                logit_mae  4      0.000000      0.000000     0.000000      0.000000      0.000000 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN                logit_mse  4      0.000000      0.000000     0.000000      0.000000      0.000000 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN           logit_spearman  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN       logit_top5_overlap  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN              memory_bits  4 598016.000000 170004.561005 85002.280503 327500.806473 868531.193527 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN    memory_ratio_vs_exact  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN             peak_vram_mb  4     14.317871      4.068733     2.034367      7.843608     20.792134 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN          prefill_seconds  4      0.000640      0.000509     0.000254     -0.000170      0.001449 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0           decode_seconds  4      0.003415      0.000699     0.000350      0.002302      0.004528 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0 hidden_cosine_similarity  4      0.940430      0.006671     0.003336      0.929814      0.951045 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0               hidden_mae  4      0.204346      0.024864     0.012432      0.164782      0.243910 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0               hidden_mse  4      0.070251      0.017002     0.008501      0.043198      0.097305 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0  logit_cosine_similarity  4      0.997070      0.001953     0.000977      0.993962      1.000178 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0                logit_mae  4     14.546875      0.802624     0.401312     13.269721     15.824029 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0                logit_mse  4    336.000000     29.439203    14.719601    289.155659    382.844341 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0           logit_spearman  4      0.978343      0.007001     0.003501      0.967203      0.989484 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0       logit_top5_overlap  4      0.834375      0.059839     0.029920      0.739158      0.929592 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0              memory_bits  4  78256.000000  22246.690600 11123.345300  42856.550847 113655.449153 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0    memory_ratio_vs_exact  4      0.130859      0.000000     0.000000      0.130859      0.130859 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0             peak_vram_mb  4     17.697754      0.263993     0.131996     17.277682     18.117826 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0          prefill_seconds  4      0.015417      0.002808     0.001404      0.010949      0.019886 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5           decode_seconds  4      0.004130      0.001479     0.000740      0.001776      0.006484 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5 hidden_cosine_similarity  4      0.958984      0.003906     0.001953      0.952769      0.965200 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5               hidden_mae  4      0.171875      0.021021     0.010510      0.138426      0.205324 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5               hidden_mse  4      0.049316      0.011845     0.005923      0.030468      0.068165 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5  logit_cosine_similarity  4      0.998047      0.003906     0.001953      0.991831      1.004263 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5                logit_mae  4     11.234375      0.766103     0.383052     10.015334     12.453416 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5                logit_mse  4    202.000000     26.495283    13.247641    159.840093    244.159907 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5           logit_spearman  4      0.982502      0.005253     0.002627      0.974143      0.990861 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5       logit_top5_overlap  4      0.856250      0.052540     0.026270      0.772648      0.939852 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5              memory_bits  4  87600.000000  24903.011866 12451.505933  47973.750948 127226.249052 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5    memory_ratio_vs_exact  4      0.146484      0.000000     0.000000      0.146484      0.146484 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5             peak_vram_mb  4     17.547241      0.220927     0.110464     17.195696     17.898786 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5          prefill_seconds  4      0.025555      0.013027     0.006514      0.004826      0.046283 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5           decode_seconds  4      0.004816      0.001329     0.000665      0.002701      0.006932 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5 hidden_cosine_similarity  4      0.987305      0.001953     0.000977      0.984197      0.990413 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5               hidden_mae  4      0.090942      0.010596     0.005298      0.074082      0.107803 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5               hidden_mse  4      0.013870      0.003163     0.001582      0.008837      0.018904 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5  logit_cosine_similarity  4      0.997070      0.003740     0.001870      0.991119      1.003021 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5                logit_mae  4      7.218750      0.319709     0.159854      6.710022      7.727478 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5                logit_mse  4     85.500000     12.369317     6.184658     65.817657    105.182343 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5           logit_spearman  4      0.990668      0.002271     0.001136      0.987054      0.994282 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5       logit_top5_overlap  4      0.887500      0.044488     0.022244      0.816710      0.958290 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5              memory_bits  4 124976.000000  35528.296929 17764.148464  68442.551353 181509.448647 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5    memory_ratio_vs_exact  4      0.208984      0.000000     0.000000      0.208984      0.208984 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5             peak_vram_mb  4     17.547241      0.220927     0.110464     17.195696     17.898786 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5          prefill_seconds  4      0.053424      0.052898     0.026449     -0.030749      0.137596 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0           decode_seconds  4      0.002800      0.000410     0.000205      0.002148      0.003452 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0 hidden_cosine_similarity  4      0.995117      0.001953     0.000977      0.992009      0.998225 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0               hidden_mae  4      0.057800      0.006663     0.003332      0.047197      0.068403 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0               hidden_mse  4      0.005638      0.001316     0.000658      0.003545      0.007731 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0  logit_cosine_similarity  4      0.998047      0.003906     0.001953      0.991831      1.004263 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0                logit_mae  4      4.820312      0.284558     0.142279      4.367517      5.273108 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0                logit_mse  4     37.187500      4.394006     2.197003     30.195655     44.179345 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0           logit_spearman  4      0.993916      0.001509     0.000754      0.991516      0.996317 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0       logit_top5_overlap  4      0.934375      0.023662     0.011831      0.896723      0.972027 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0              memory_bits  4 153008.000000  43497.260726 21748.630363  83794.151656 222221.848344 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0    memory_ratio_vs_exact  4      0.255859      0.000000     0.000000      0.255859      0.255859 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0             peak_vram_mb  4     17.697754      0.263993     0.131996     17.277682     18.117826 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0          prefill_seconds  4      0.014265      0.003426     0.001713      0.008815      0.019716 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0           decode_seconds  4      0.003195      0.002300     0.001150     -0.000465      0.006855 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0 hidden_cosine_similarity  4      1.000977      0.004915     0.002458      0.993155      1.008798 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0               hidden_mae  4      0.008858      0.002048     0.001024      0.005598      0.012117 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0               hidden_mse  4      0.000395      0.000256     0.000128     -0.000012      0.000802 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0  logit_cosine_similarity  4      0.996094      0.003189     0.001595      0.991019      1.001169 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0                logit_mae  4     16.750000      0.444878     0.222439     16.042099     17.457901 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0                logit_mse  4    448.500000     29.137605    14.568802    402.135569    494.864431 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0           logit_spearman  4      0.974887      0.002623     0.001311      0.970714      0.979061 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0       logit_top5_overlap  4      0.800000      0.044488     0.022244      0.729210      0.870790 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0              memory_bits  4 338720.000000  96291.645882 48145.822941 185498.503667 491941.496333 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0    memory_ratio_vs_exact  4      0.566406      0.000000     0.000000      0.566406      0.566406 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0             peak_vram_mb  4     17.644531      0.248527     0.124263     17.249070     18.039993 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0          prefill_seconds  4      0.006585      0.003272     0.001636      0.001379      0.011791 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5           decode_seconds  4      0.005054      0.002179     0.001089      0.001588      0.008521 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5 hidden_cosine_similarity  4      0.998047      0.006766     0.003383      0.987281      1.008813 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5               hidden_mae  4      0.010254      0.002111     0.001056      0.006894      0.013614 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5               hidden_mse  4      0.000635      0.000414     0.000207     -0.000024      0.001294 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5  logit_cosine_similarity  4      0.997070      0.001953     0.000977      0.993962      1.000178 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5                logit_mae  4     13.625000      0.517908     0.258954     12.800892     14.449108 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5                logit_mse  4    303.000000     22.181073    11.090537    267.704963    338.295037 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5           logit_spearman  4      0.978083      0.005292     0.002646      0.969662      0.986504 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5       logit_top5_overlap  4      0.850000      0.030619     0.015309      0.801279      0.898721 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5              memory_bits  4 343392.000000  97619.806515 48809.903257 188057.103717 498726.896283 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5    memory_ratio_vs_exact  4      0.574219      0.000000     0.000000      0.574219      0.574219 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5             peak_vram_mb  4     17.493042      0.205461     0.102731     17.166107     17.819977 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5          prefill_seconds  4      0.015370      0.004935     0.002467      0.007517      0.023222 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5           decode_seconds  4      0.004770      0.002252     0.001126      0.001186      0.008355 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5 hidden_cosine_similarity  4      0.998047      0.003906     0.001953      0.991831      1.004263 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5               hidden_mae  4      0.005604      0.002719     0.001359      0.001277      0.009930 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5               hidden_mse  4      0.000181      0.000168     0.000084     -0.000086      0.000448 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5  logit_cosine_similarity  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5                logit_mae  4      8.554688      0.638589     0.319295      7.538550      9.570825 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5                logit_mse  4    111.875000     14.285045     7.142522     89.144306    134.605694 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5           logit_spearman  4      0.989026      0.001437     0.000719      0.986740      0.991313 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5       logit_top5_overlap  4      0.893750      0.042696     0.021348      0.825812      0.961688 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5              memory_bits  4 362080.000000 102932.449046 51466.224523 198291.503919 525868.496081 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5    memory_ratio_vs_exact  4      0.605469      0.000000     0.000000      0.605469      0.605469 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5             peak_vram_mb  4     17.493042      0.205461     0.102731     17.166107     17.819977 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5          prefill_seconds  4      0.012813      0.006040     0.003020      0.003202      0.022423 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0           decode_seconds  4      0.002601      0.000770     0.000385      0.001375      0.003826 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0 hidden_cosine_similarity  4      0.998047      0.003906     0.001953      0.991831      1.004263 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0               hidden_mae  4      0.003571      0.001175     0.000587      0.001701      0.005440 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0               hidden_mse  4      0.000077      0.000061     0.000030     -0.000019      0.000173 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0  logit_cosine_similarity  4      0.999023      0.001953     0.000977      0.995916      1.002131 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0                logit_mae  4      6.257812      0.428763     0.214382      5.575555      6.940070 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0                logit_mse  4     61.937500     10.774923     5.387461     44.792194     79.082806 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0           logit_spearman  4      0.992944      0.001386     0.000693      0.990738      0.995149 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0       logit_top5_overlap  4      0.931250      0.033072     0.016536      0.878625      0.983875 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0              memory_bits  4 376096.000000 106916.930945 53458.465472 205967.304071 546224.695929 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0    memory_ratio_vs_exact  4      0.628906      0.000000     0.000000      0.628906      0.628906 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0             peak_vram_mb  4     17.644531      0.248527     0.124263     17.249070     18.039993 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0          prefill_seconds  4      0.008920      0.003856     0.001928      0.002783      0.015056 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0           decode_seconds  4      0.003754      0.002313     0.001156      0.000074      0.007434 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0 hidden_cosine_similarity  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0               hidden_mae  4      0.011925      0.003937     0.001968      0.005660      0.018189 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0               hidden_mse  4      0.001321      0.000819     0.000410      0.000018      0.002624 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0  logit_cosine_similarity  4      0.997070      0.003740     0.001870      0.991119      1.003021 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0                logit_mae  4     15.984375      1.452705     0.726352     13.672797     18.295953 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0                logit_mse  4    398.500000     62.660461    31.330230    298.793224    498.206776 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0           logit_spearman  4      0.975417      0.007410     0.003705      0.963626      0.987207 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0       logit_top5_overlap  4      0.843750      0.075346     0.037673      0.723857      0.963643 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0              memory_bits  4 338720.000000  96291.645882 48145.822941 185498.503667 491941.496333 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0    memory_ratio_vs_exact  4      0.566406      0.000000     0.000000      0.566406      0.566406 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0             peak_vram_mb  4     15.610352      4.141278     2.070639      9.020654     22.200049 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0          prefill_seconds  4      0.023188      0.032077     0.016038     -0.027854      0.074229 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5           decode_seconds  4      0.004765      0.001613     0.000806      0.002199      0.007331 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5 hidden_cosine_similarity  4      0.998047      0.009831     0.004915      0.982404      1.013689 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5               hidden_mae  4      0.017220      0.009854     0.004927      0.001540      0.032899 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5               hidden_mse  4      0.002679      0.003329     0.001665     -0.002619      0.007977 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5  logit_cosine_similarity  4      0.998047      0.002255     0.001128      0.994458      1.001636 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5                logit_mae  4     13.953125      1.679669     0.839835     11.280396     16.625854 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5                logit_mse  4    319.000000     62.875538    31.437769    218.950988    419.049012 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5           logit_spearman  4      0.977573      0.007325     0.003663      0.965917      0.989229 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5       logit_top5_overlap  4      0.834375      0.063225     0.031612      0.733770      0.934980 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5              memory_bits  4 343392.000000  97619.806515 48809.903257 188057.103717 498726.896283 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5    memory_ratio_vs_exact  4      0.574219      0.000000     0.000000      0.574219      0.574219 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5             peak_vram_mb  4     17.490112      0.205461     0.102731     17.163178     17.817047 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5          prefill_seconds  4      0.012040      0.003237     0.001619      0.006889      0.017191 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5           decode_seconds  4      0.003948      0.001341     0.000671      0.001814      0.006083 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5 hidden_cosine_similarity  4      0.999023      0.001953     0.000977      0.995916      1.002131 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5               hidden_mae  4      0.005634      0.001900     0.000950      0.002612      0.008657 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5               hidden_mse  4      0.000152      0.000133     0.000067     -0.000060      0.000365 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5  logit_cosine_similarity  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5                logit_mae  4      8.578125      0.606765     0.303383      7.612626      9.543624 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5                logit_mse  4    119.250000     16.291613     8.145807     93.326408    145.173592 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5           logit_spearman  4      0.988193      0.002926     0.001463      0.983537      0.992849 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5       logit_top5_overlap  4      0.890625      0.051412     0.025706      0.808816      0.972433 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5              memory_bits  4 362080.000000 102932.449046 51466.224523 198291.503919 525868.496081 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5    memory_ratio_vs_exact  4      0.605469      0.000000     0.000000      0.605469      0.605469 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5             peak_vram_mb  4     17.490112      0.205461     0.102731     17.163178     17.817047 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5          prefill_seconds  4      0.011170      0.004151     0.002076      0.004564      0.017775 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0           decode_seconds  4      0.003816      0.001471     0.000735      0.001476      0.006156 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0 hidden_cosine_similarity  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0               hidden_mae  4      0.004349      0.000783     0.000392      0.003102      0.005595 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0               hidden_mse  4      0.000102      0.000085     0.000042     -0.000033      0.000237 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0  logit_cosine_similarity  4      0.999023      0.001953     0.000977      0.995916      1.002131 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0                logit_mae  4      6.789062      0.926454     0.463227      5.314867      8.263258 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0                logit_mse  4     77.687500     25.140750    12.570375     37.682957    117.692043 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0           logit_spearman  4      0.992662      0.002050     0.001025      0.989399      0.995924 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0       logit_top5_overlap  4      0.906250      0.029756     0.014878      0.858902      0.953598 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0              memory_bits  4 376096.000000 106916.930945 53458.465472 205967.304071 546224.695929 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0    memory_ratio_vs_exact  4      0.628906      0.000000     0.000000      0.628906      0.628906 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0             peak_vram_mb  4     17.641602      0.248527     0.124263     17.246140     18.037063 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0          prefill_seconds  4      0.011457      0.003871     0.001936      0.005297      0.017617 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0           decode_seconds  4      0.005215      0.001630     0.000815      0.002621      0.007808 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0 hidden_cosine_similarity  4      0.997070      0.003740     0.001870      0.991119      1.003021 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0               hidden_mae  4      0.009300      0.003330     0.001665      0.004002      0.014599 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0               hidden_mse  4      0.000512      0.000402     0.000201     -0.000127      0.001152 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0  logit_cosine_similarity  4      0.997070      0.001953     0.000977      0.993962      1.000178 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0                logit_mae  4     14.546875      0.802624     0.401312     13.269721     15.824029 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0                logit_mse  4    336.000000     29.439203    14.719601    289.155659    382.844341 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0           logit_spearman  4      0.978343      0.007001     0.003501      0.967203      0.989484 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0       logit_top5_overlap  4      0.834375      0.059839     0.029920      0.739158      0.929592 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0              memory_bits  4 338720.000000  96291.645882 48145.822941 185498.503667 491941.496333 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0    memory_ratio_vs_exact  4      0.566406      0.000000     0.000000      0.566406      0.566406 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0             peak_vram_mb  4     15.610352      4.141278     2.070639      9.020654     22.200049 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0          prefill_seconds  4      0.173820      0.328519     0.164260     -0.348927      0.696567 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5           decode_seconds  4      0.003937      0.000664     0.000332      0.002880      0.004994 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5 hidden_cosine_similarity  4      1.000000      0.006379     0.003189      0.989850      1.010150 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5               hidden_mae  4      0.008034      0.002680     0.001340      0.003769      0.012298 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5               hidden_mse  4      0.000334      0.000177     0.000088      0.000052      0.000615 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5  logit_cosine_similarity  4      0.998047      0.003906     0.001953      0.991831      1.004263 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5                logit_mae  4     11.234375      0.766103     0.383052     10.015334     12.453416 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5                logit_mse  4    202.000000     26.495283    13.247641    159.840093    244.159907 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5           logit_spearman  4      0.982502      0.005253     0.002627      0.974143      0.990861 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5       logit_top5_overlap  4      0.856250      0.052540     0.026270      0.772648      0.939852 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5              memory_bits  4 343392.000000  97619.806515 48809.903257 188057.103717 498726.896283 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5    memory_ratio_vs_exact  4      0.574219      0.000000     0.000000      0.574219      0.574219 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5             peak_vram_mb  4     17.490112      0.205461     0.102731     17.163178     17.817047 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5          prefill_seconds  4      0.012515      0.003297     0.001648      0.007270      0.017761 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5           decode_seconds  4      0.003728      0.000934     0.000467      0.002241      0.005214 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5 hidden_cosine_similarity  4      0.999023      0.001953     0.000977      0.995916      1.002131 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5               hidden_mae  4      0.004642      0.001182     0.000591      0.002762      0.006523 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5               hidden_mse  4      0.000093      0.000047     0.000024      0.000018      0.000168 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5  logit_cosine_similarity  4      0.997070      0.003740     0.001870      0.991119      1.003021 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5                logit_mae  4      7.218750      0.319709     0.159854      6.710022      7.727478 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5                logit_mse  4     85.500000     12.369317     6.184658     65.817657    105.182343 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5           logit_spearman  4      0.990668      0.002271     0.001136      0.987054      0.994282 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5       logit_top5_overlap  4      0.887500      0.044488     0.022244      0.816710      0.958290 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5              memory_bits  4 362080.000000 102932.449046 51466.224523 198291.503919 525868.496081 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5    memory_ratio_vs_exact  4      0.605469      0.000000     0.000000      0.605469      0.605469 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5             peak_vram_mb  4     17.490112      0.205461     0.102731     17.163178     17.817047 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5          prefill_seconds  4      0.010783      0.003170     0.001585      0.005739      0.015827 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0           decode_seconds  4      0.003355      0.001462     0.000731      0.001029      0.005682 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0 hidden_cosine_similarity  4      0.999023      0.001953     0.000977      0.995916      1.002131 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0               hidden_mae  4      0.002890      0.001403     0.000701      0.000658      0.005121 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0               hidden_mse  4      0.000053      0.000064     0.000032     -0.000050      0.000155 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0  logit_cosine_similarity  4      0.998047      0.003906     0.001953      0.991831      1.004263 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0                logit_mae  4      4.820312      0.284558     0.142279      4.367517      5.273108 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0                logit_mse  4     37.187500      4.394006     2.197003     30.195655     44.179345 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0           logit_spearman  4      0.993916      0.001509     0.000754      0.991516      0.996317 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0       logit_top5_overlap  4      0.934375      0.023662     0.011831      0.896723      0.972027 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0              memory_bits  4 376096.000000 106916.930945 53458.465472 205967.304071 546224.695929 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0    memory_ratio_vs_exact  4      0.628906      0.000000     0.000000      0.628906      0.628906 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0             peak_vram_mb  4     17.641602      0.248527     0.124263     17.246140     18.037063 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0          prefill_seconds  4      0.010262      0.005105     0.002553      0.002139      0.018385 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0           decode_seconds  4      0.003419      0.002294     0.001147     -0.000232      0.007070 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0 hidden_cosine_similarity  4      0.955078      0.005043     0.002521      0.947054      0.963103 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0               hidden_mae  4      0.168457      0.019162     0.009581      0.137967      0.198947 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0               hidden_mse  4      0.052917      0.012101     0.006051      0.033661      0.072173 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0  logit_cosine_similarity  4      0.996094      0.003189     0.001595      0.991019      1.001169 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0                logit_mae  4     16.750000      0.444878     0.222439     16.042099     17.457901 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0                logit_mse  4    448.500000     29.137605    14.568802    402.135569    494.864431 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0           logit_spearman  4      0.974887      0.002623     0.001311      0.970714      0.979061 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0       logit_top5_overlap  4      0.800000      0.044488     0.022244      0.729210      0.870790 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0              memory_bits  4 122060.000000  34117.126256 17058.563128  67772.038799 176347.961201 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0    memory_ratio_vs_exact  4      0.204352      0.001157     0.000579      0.202510      0.206193 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0             peak_vram_mb  4     17.666504      0.253860     0.126930     17.262556     18.070452 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0          prefill_seconds  4      0.700848      1.370550     0.685275     -1.480003      2.881699 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5           decode_seconds  4      0.005918      0.001431     0.000715      0.003642      0.008195 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5 hidden_cosine_similarity  4      0.957031      0.003189     0.001595      0.951956      0.962106 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5               hidden_mae  4      0.168701      0.019358     0.009679      0.137899      0.199503 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5               hidden_mse  4      0.052979      0.012412     0.006206      0.033228      0.072729 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5  logit_cosine_similarity  4      0.997070      0.001953     0.000977      0.993962      1.000178 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5                logit_mae  4     13.625000      0.517908     0.258954     12.800892     14.449108 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5                logit_mse  4    303.000000     22.181073    11.090537    267.704963    338.295037 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5           logit_spearman  4      0.978083      0.005292     0.002646      0.969662      0.986504 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5       logit_top5_overlap  4      0.850000      0.030619     0.015309      0.801279      0.898721 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5              memory_bits  4 126732.000000  35445.286889 17722.643445  70330.638850 183133.361150 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5    memory_ratio_vs_exact  4      0.212164      0.001157     0.000579      0.210323      0.214006 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5             peak_vram_mb  4     17.570679      0.226261     0.113130     17.210647     17.930710 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5          prefill_seconds  4      0.022296      0.002812     0.001406      0.017822      0.026770 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5           decode_seconds  4      0.004726      0.001946     0.000973      0.001630      0.007822 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5 hidden_cosine_similarity  4      0.985352      0.003740     0.001870      0.979400      0.991303 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5               hidden_mae  4      0.092651      0.010057     0.005029      0.076648      0.108655 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5               hidden_mse  4      0.016251      0.003709     0.001855      0.010348      0.022153 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5  logit_cosine_similarity  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5                logit_mae  4      8.554688      0.638589     0.319295      7.538550      9.570825 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5                logit_mse  4    111.875000     14.285045     7.142522     89.144306    134.605694 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5           logit_spearman  4      0.989026      0.001437     0.000719      0.986740      0.991313 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5       logit_top5_overlap  4      0.893750      0.042696     0.021348      0.825812      0.961688 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5              memory_bits  4 162210.000000  45531.006695 22765.503348  89760.007984 234659.992016 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5    memory_ratio_vs_exact  4      0.271490      0.001157     0.000579      0.269649      0.273332 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5             peak_vram_mb  4     17.570679      0.226261     0.113130     17.210647     17.930710 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5          prefill_seconds  4      0.018746      0.006430     0.003215      0.008514      0.028977 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0           decode_seconds  4      0.003008      0.001315     0.000658      0.000915      0.005101 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0 hidden_cosine_similarity  4      0.995117      0.004915     0.002458      0.987296      1.002938 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0               hidden_mae  4      0.048889      0.005107     0.002554      0.040763      0.057016 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0               hidden_mse  4      0.004528      0.000909     0.000455      0.003081      0.005975 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0  logit_cosine_similarity  4      0.999023      0.001953     0.000977      0.995916      1.002131 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0                logit_mae  4      6.257812      0.428763     0.214382      5.575555      6.940070 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0                logit_mse  4     61.937500     10.774923     5.387461     44.792194     79.082806 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0           logit_spearman  4      0.992944      0.001386     0.000693      0.990738      0.995149 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0       logit_top5_overlap  4      0.931250      0.033072     0.016536      0.878625      0.983875 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0              memory_bits  4 193016.000000  54288.565868 27144.282934 106630.777067 279401.222933 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0    memory_ratio_vs_exact  4      0.323004      0.001157     0.000579      0.321163      0.324845 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0             peak_vram_mb  4     17.666504      0.253860     0.126930     17.262556     18.070452 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0          prefill_seconds  4      0.017917      0.003580     0.001790      0.012220      0.023614 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0           decode_seconds  4      0.002853      0.000968     0.000484      0.001313      0.004394 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0 hidden_cosine_similarity  4      0.964844      0.003189     0.001595      0.959769      0.969919 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0               hidden_mae  4      0.159912      0.014624     0.007312      0.136642      0.183182 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0               hidden_mse  4      0.045288      0.007955     0.003977      0.032630      0.057946 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0  logit_cosine_similarity  4      0.996094      0.003189     0.001595      0.991019      1.001169 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0                logit_mae  4     16.750000      0.444878     0.222439     16.042099     17.457901 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0                logit_mse  4    448.500000     29.137605    14.568802    402.135569    494.864431 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0           logit_spearman  4      0.974887      0.002623     0.001311      0.970714      0.979061 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0       logit_top5_overlap  4      0.800000      0.044488     0.022244      0.729210      0.870790 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0              memory_bits  4 126732.000000  35445.286889 17722.643445  70330.638850 183133.361150 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0    memory_ratio_vs_exact  4      0.212164      0.001157     0.000579      0.210323      0.214006 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0             peak_vram_mb  4     21.063477      0.158838     0.079419     20.810729     21.316224 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0          prefill_seconds  4      0.106277      0.099929     0.049964     -0.052732      0.265287 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5           decode_seconds  4      0.003194      0.000768     0.000384      0.001971      0.004416 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5 hidden_cosine_similarity  4      0.961914      0.004915     0.002458      0.954093      0.969735 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5               hidden_mae  4      0.159912      0.014775     0.007388      0.136401      0.183423 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5               hidden_mse  4      0.045410      0.008063     0.004031      0.032580      0.058240 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5  logit_cosine_similarity  4      0.997070      0.001953     0.000977      0.993962      1.000178 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5                logit_mae  4     13.625000      0.517908     0.258954     12.800892     14.449108 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5                logit_mse  4    303.000000     22.181073    11.090537    267.704963    338.295037 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5           logit_spearman  4      0.978083      0.005292     0.002646      0.969662      0.986504 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5       logit_top5_overlap  4      0.850000      0.030619     0.015309      0.801279      0.898721 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5              memory_bits  4 131404.000000  36773.447522 18386.723761  72889.238900 189918.761100 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5    memory_ratio_vs_exact  4      0.219977      0.001157     0.000579      0.218135      0.221818 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5             peak_vram_mb  4     21.064453      0.158838     0.079419     20.811706     21.317200 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5          prefill_seconds  4      0.026394      0.007880     0.003940      0.013856      0.038932 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5           decode_seconds  4      0.004337      0.001517     0.000758      0.001923      0.006750 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5 hidden_cosine_similarity  4      0.985352      0.003740     0.001870      0.979400      0.991303 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5               hidden_mae  4      0.088379      0.007098     0.003549      0.077084      0.099674 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5               hidden_mse  4      0.014023      0.002315     0.001158      0.010339      0.017707 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5  logit_cosine_similarity  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5                logit_mae  4      8.554688      0.638589     0.319295      7.538550      9.570825 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5                logit_mse  4    111.875000     14.285045     7.142522     89.144306    134.605694 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5           logit_spearman  4      0.989026      0.001437     0.000719      0.986740      0.991313 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5       logit_top5_overlap  4      0.893750      0.042696     0.021348      0.825812      0.961688 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5              memory_bits  4 166882.000000  46859.167328 23429.583664  92318.608034 241445.391966 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5    memory_ratio_vs_exact  4      0.279303      0.001157     0.000579      0.277462      0.281144 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5             peak_vram_mb  4     21.064453      0.158838     0.079419     20.811706     21.317200 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5          prefill_seconds  4      0.045239      0.015322     0.007661      0.020858      0.069620 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0           decode_seconds  4      0.003690      0.001786     0.000893      0.000848      0.006532 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0 hidden_cosine_similarity  4      0.998047      0.002255     0.001128      0.994458      1.001636 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0               hidden_mae  4      0.046875      0.004119     0.002060      0.040321      0.053429 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0               hidden_mse  4      0.003925      0.000618     0.000309      0.002943      0.004908 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0  logit_cosine_similarity  4      0.999023      0.001953     0.000977      0.995916      1.002131 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0                logit_mae  4      6.257812      0.428763     0.214382      5.575555      6.940070 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0                logit_mse  4     61.937500     10.774923     5.387461     44.792194     79.082806 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0           logit_spearman  4      0.992944      0.001386     0.000693      0.990738      0.995149 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0         logit_top1_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0         logit_top5_match  4      1.000000      0.000000     0.000000      1.000000      1.000000 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0       logit_top5_overlap  4      0.931250      0.033072     0.016536      0.878625      0.983875 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0              memory_bits  4 197688.000000  55616.726501 27808.363250 109189.377118 286186.622882 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0    memory_ratio_vs_exact  4      0.330817      0.001157     0.000579      0.328975      0.332658 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0             peak_vram_mb  4     21.063477      0.158838     0.079419     20.810729     21.316224 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0          prefill_seconds  4      0.025003      0.009097     0.004549      0.010527      0.039479 H:\Qwen3.5-9B-official-hf     captured
```

## First-Layer Thresholds

```
 dataset                       mode bit_setting  bits         capture_id prompt_label                                                      prompt_hash                   metric  threshold  n  mean  std  sem  ci95_low  ci95_high                  model_id query_source
captured                      exact       exact   NaN    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                      exact       exact   NaN   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           2   2.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         2.5   2.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv         3.5   3.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                    full_kv           4   4.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           2   2.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         2.5   2.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned         3.5   3.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured key_only_block_so8_learned           4   4.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           2   2.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         2.5   2.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static         3.5   3.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured  key_only_block_so8_static           4   4.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           2   2.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         2.5   2.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random         3.5   3.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured            key_only_random           4   4.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         3.5   3.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           4   4.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           2   2.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         2.5   2.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank         3.5   3.5   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured        protected_v_lowrank           4   4.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
```
