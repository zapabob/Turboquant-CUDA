# Attention Replay Summary

- Model: Qwen/Qwen3.5-9B
- Query source: synthetic

Current mathematical bottleneck: value quantization amplifies attention-output drift more than key quantization alone. At 2 bits, full-KV trails block-SO(8) key-only by 0.0578 hidden-state cosine.


## Primary Pareto Table

```
                      mode bit_setting                   metric         mean          std          sem      ci95_low    ci95_high
                     exact       exact    memory_ratio_vs_exact 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
           key_only_random           2    memory_ratio_vs_exact 5.390625e-01 0.000000e+00 0.000000e+00  5.390625e-01 5.390625e-01
           key_only_random         2.5    memory_ratio_vs_exact 5.429688e-01 0.000000e+00 0.000000e+00  5.429688e-01 5.429688e-01
           key_only_random           3    memory_ratio_vs_exact 5.546875e-01 0.000000e+00 0.000000e+00  5.546875e-01 5.546875e-01
           key_only_random         3.5    memory_ratio_vs_exact 5.585938e-01 0.000000e+00 0.000000e+00  5.585938e-01 5.585938e-01
           key_only_random           4    memory_ratio_vs_exact 5.703125e-01 0.000000e+00 0.000000e+00  5.703125e-01 5.703125e-01
 key_only_block_so8_static           2    memory_ratio_vs_exact 5.390625e-01 0.000000e+00 0.000000e+00  5.390625e-01 5.390625e-01
 key_only_block_so8_static         2.5    memory_ratio_vs_exact 5.429688e-01 0.000000e+00 0.000000e+00  5.429688e-01 5.429688e-01
 key_only_block_so8_static           3    memory_ratio_vs_exact 5.546875e-01 0.000000e+00 0.000000e+00  5.546875e-01 5.546875e-01
 key_only_block_so8_static         3.5    memory_ratio_vs_exact 5.585938e-01 0.000000e+00 0.000000e+00  5.585938e-01 5.585938e-01
 key_only_block_so8_static           4    memory_ratio_vs_exact 5.703125e-01 0.000000e+00 0.000000e+00  5.703125e-01 5.703125e-01
key_only_block_so8_learned           2    memory_ratio_vs_exact 5.390625e-01 0.000000e+00 0.000000e+00  5.390625e-01 5.390625e-01
key_only_block_so8_learned         2.5    memory_ratio_vs_exact 5.429688e-01 0.000000e+00 0.000000e+00  5.429688e-01 5.429688e-01
key_only_block_so8_learned           3    memory_ratio_vs_exact 5.546875e-01 0.000000e+00 0.000000e+00  5.546875e-01 5.546875e-01
key_only_block_so8_learned         3.5    memory_ratio_vs_exact 5.585938e-01 0.000000e+00 0.000000e+00  5.585938e-01 5.585938e-01
key_only_block_so8_learned           4    memory_ratio_vs_exact 5.703125e-01 0.000000e+00 0.000000e+00  5.703125e-01 5.703125e-01
               protected_v           2    memory_ratio_vs_exact 1.354980e-01 0.000000e+00 0.000000e+00  1.354980e-01 1.354980e-01
               protected_v         2.5    memory_ratio_vs_exact 1.394043e-01 0.000000e+00 0.000000e+00  1.394043e-01 1.394043e-01
               protected_v           3    memory_ratio_vs_exact 1.651611e-01 0.000000e+00 0.000000e+00  1.651611e-01 1.651611e-01
               protected_v         3.5    memory_ratio_vs_exact 1.690674e-01 0.000000e+00 0.000000e+00  1.690674e-01 1.690674e-01
               protected_v           4    memory_ratio_vs_exact 1.948242e-01 0.000000e+00 0.000000e+00  1.948242e-01 1.948242e-01
       protected_v_lowrank           2    memory_ratio_vs_exact 1.511230e-01 0.000000e+00 0.000000e+00  1.511230e-01 1.511230e-01
       protected_v_lowrank         2.5    memory_ratio_vs_exact 1.550293e-01 0.000000e+00 0.000000e+00  1.550293e-01 1.550293e-01
       protected_v_lowrank           3    memory_ratio_vs_exact 1.807861e-01 0.000000e+00 0.000000e+00  1.807861e-01 1.807861e-01
       protected_v_lowrank         3.5    memory_ratio_vs_exact 1.846924e-01 0.000000e+00 0.000000e+00  1.846924e-01 1.846924e-01
       protected_v_lowrank           4    memory_ratio_vs_exact 2.104492e-01 0.000000e+00 0.000000e+00  2.104492e-01 2.104492e-01
                   full_kv           2    memory_ratio_vs_exact 7.421875e-02 0.000000e+00 0.000000e+00  7.421875e-02 7.421875e-02
                   full_kv         2.5    memory_ratio_vs_exact 8.203125e-02 0.000000e+00 0.000000e+00  8.203125e-02 8.203125e-02
                   full_kv           3    memory_ratio_vs_exact 1.054688e-01 0.000000e+00 0.000000e+00  1.054688e-01 1.054688e-01
                   full_kv         3.5    memory_ratio_vs_exact 1.132812e-01 0.000000e+00 0.000000e+00  1.132812e-01 1.132812e-01
                   full_kv           4    memory_ratio_vs_exact 1.367188e-01 0.000000e+00 0.000000e+00  1.367188e-01 1.367188e-01
                     exact       exact hidden_cosine_similarity 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
           key_only_random           2 hidden_cosine_similarity 9.999993e-01 6.124601e-07 3.062300e-07  9.999983e-01 1.000000e+00
           key_only_random         2.5 hidden_cosine_similarity 9.999995e-01 7.412735e-07 3.706367e-07  9.999983e-01 1.000001e+00
           key_only_random           3 hidden_cosine_similarity 9.999999e-01 1.141342e-07 5.706710e-08  9.999997e-01 1.000000e+00
           key_only_random         3.5 hidden_cosine_similarity 9.999998e-01 3.424026e-07 1.712013e-07  9.999992e-01 1.000000e+00
           key_only_random           4 hidden_cosine_similarity 9.999999e-01 2.064765e-07 1.032383e-07  9.999996e-01 1.000000e+00
 key_only_block_so8_static           2 hidden_cosine_similarity 9.999995e-01 3.576279e-07 1.788139e-07  9.999989e-01 1.000000e+00
 key_only_block_so8_static         2.5 hidden_cosine_similarity 9.999998e-01 2.035886e-07 1.017943e-07  9.999995e-01 1.000000e+00
 key_only_block_so8_static           3 hidden_cosine_similarity 9.999996e-01 1.946680e-07 9.733398e-08  9.999993e-01 1.000000e+00
 key_only_block_so8_static         3.5 hidden_cosine_similarity 9.999998e-01 3.706367e-07 1.853184e-07  9.999992e-01 1.000000e+00
 key_only_block_so8_static           4 hidden_cosine_similarity 1.000000e+00 1.500017e-07 7.500087e-08  9.999998e-01 1.000000e+00
key_only_block_so8_learned           2 hidden_cosine_similarity 9.999989e-01 1.552774e-06 7.763872e-07  9.999964e-01 1.000001e+00
key_only_block_so8_learned         2.5 hidden_cosine_similarity 9.999996e-01 4.802399e-07 2.401199e-07  9.999988e-01 1.000000e+00
key_only_block_so8_learned           3 hidden_cosine_similarity 9.999998e-01 1.192093e-07 5.960464e-08  9.999996e-01 1.000000e+00
key_only_block_so8_learned         3.5 hidden_cosine_similarity 9.999998e-01 1.788139e-07 8.940697e-08  9.999995e-01 1.000000e+00
key_only_block_so8_learned           4 hidden_cosine_similarity 9.999999e-01 1.500017e-07 7.500087e-08  9.999997e-01 1.000000e+00
               protected_v           2 hidden_cosine_similarity 9.556855e-01 2.592696e-03 1.296348e-03  9.515600e-01 9.598111e-01
               protected_v         2.5 hidden_cosine_similarity 9.556810e-01 2.591929e-03 1.295964e-03  9.515567e-01 9.598054e-01
               protected_v           3 hidden_cosine_similarity 9.872561e-01 4.479650e-04 2.239825e-04  9.865432e-01 9.879689e-01
               protected_v         3.5 hidden_cosine_similarity 9.872558e-01 4.475463e-04 2.237731e-04  9.865437e-01 9.879680e-01
               protected_v           4 hidden_cosine_similarity 9.964459e-01 2.164747e-04 1.082374e-04  9.961015e-01 9.967904e-01
       protected_v_lowrank           2 hidden_cosine_similarity 9.624704e-01 2.045371e-03 1.022686e-03  9.592157e-01 9.657250e-01
       protected_v_lowrank         2.5 hidden_cosine_similarity 9.624688e-01 2.045139e-03 1.022570e-03  9.592145e-01 9.657231e-01
       protected_v_lowrank           3 hidden_cosine_similarity 9.890233e-01 4.809704e-04 2.404852e-04  9.882580e-01 9.897886e-01
       protected_v_lowrank         3.5 hidden_cosine_similarity 9.890235e-01 4.813546e-04 2.406773e-04  9.882575e-01 9.897894e-01
       protected_v_lowrank           4 hidden_cosine_similarity 9.969856e-01 1.808270e-04 9.041349e-05  9.966979e-01 9.972734e-01
                   full_kv           2 hidden_cosine_similarity 9.421799e-01 3.195437e-03 1.597718e-03  9.370952e-01 9.472645e-01
                   full_kv         2.5 hidden_cosine_similarity 9.594020e-01 1.658602e-03 8.293008e-04  9.567628e-01 9.620412e-01
                   full_kv           3 hidden_cosine_similarity 9.836487e-01 1.245004e-03 6.225019e-04  9.816676e-01 9.856298e-01
                   full_kv         3.5 hidden_cosine_similarity 9.888864e-01 4.129969e-04 2.064984e-04  9.882292e-01 9.895436e-01
                   full_kv           4 hidden_cosine_similarity 9.954909e-01 3.354689e-04 1.677344e-04  9.949571e-01 9.960247e-01
                     exact       exact               hidden_mse 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00
           key_only_random           2               hidden_mse 1.888114e-06 1.422744e-06 7.113718e-07 -3.757885e-07 4.152016e-06
           key_only_random         2.5               hidden_mse 1.319156e-06 1.652239e-06 8.261196e-07 -1.309925e-06 3.948238e-06
           key_only_random           3               hidden_mse 1.526920e-07 8.718085e-08 4.359043e-08  1.396777e-08 2.914161e-07
           key_only_random         3.5               hidden_mse 5.388360e-07 4.955503e-07 2.477752e-07 -2.496951e-07 1.327367e-06
           key_only_random           4               hidden_mse 2.696188e-07 3.595475e-07 1.797737e-07 -3.025015e-07 8.417391e-07
 key_only_block_so8_static           2               hidden_mse 1.213255e-06 8.126604e-07 4.063302e-07 -7.986906e-08 2.506379e-06
 key_only_block_so8_static         2.5               hidden_mse 5.949489e-07 5.328378e-07 2.664189e-07 -2.529150e-07 1.442813e-06
 key_only_block_so8_static           3               hidden_mse 8.847929e-07 6.137360e-07 3.068680e-07 -9.179808e-08 1.861384e-06
 key_only_block_so8_static         3.5               hidden_mse 5.674531e-07 6.763651e-07 3.381825e-07 -5.087947e-07 1.643701e-06
 key_only_block_so8_static           4               hidden_mse 9.847738e-08 1.013840e-07 5.069201e-08 -6.284721e-08 2.598020e-07
key_only_block_so8_learned           2               hidden_mse 2.460368e-06 3.642159e-06 1.821080e-06 -3.335121e-06 8.255856e-06
key_only_block_so8_learned         2.5               hidden_mse 9.750123e-07 1.067145e-06 5.335723e-07 -7.230528e-07 2.673077e-06
key_only_block_so8_learned           3               hidden_mse 2.682933e-07 1.638364e-07 8.191820e-08  7.593059e-09 5.289936e-07
key_only_block_so8_learned         3.5               hidden_mse 3.422938e-07 1.676364e-07 8.381822e-08  7.554676e-08 6.090407e-07
key_only_block_so8_learned           4               hidden_mse 9.366999e-08 7.777750e-08 3.888875e-08 -3.009138e-08 2.174313e-07
               protected_v           2               hidden_mse 8.984431e-02 5.017488e-03 2.508744e-03  8.186037e-02 9.782825e-02
               protected_v         2.5               hidden_mse 8.985602e-02 5.018472e-03 2.509236e-03  8.187051e-02 9.784153e-02
               protected_v           3               hidden_mse 2.609728e-02 8.816241e-04 4.408120e-04  2.469442e-02 2.750014e-02
               protected_v         3.5               hidden_mse 2.609809e-02 8.820601e-04 4.410300e-04  2.469454e-02 2.750165e-02
               protected_v           4               hidden_mse 7.307117e-03 3.978703e-04 1.989351e-04  6.674017e-03 7.940218e-03
       protected_v_lowrank           2               hidden_mse 7.624669e-02 3.510824e-03 1.755412e-03  7.066019e-02 8.183320e-02
       protected_v_lowrank         2.5               hidden_mse 7.625137e-02 3.511482e-03 1.755741e-03  7.066381e-02 8.183892e-02
       protected_v_lowrank           3               hidden_mse 2.248513e-02 1.117404e-03 5.587020e-04  2.070709e-02 2.426317e-02
       protected_v_lowrank         3.5               hidden_mse 2.248520e-02 1.117531e-03 5.587655e-04  2.070696e-02 2.426345e-02
       protected_v_lowrank           4               hidden_mse 6.201442e-03 3.118335e-04 1.559168e-04  5.705245e-03 6.697638e-03
                   full_kv           2               hidden_mse 1.155447e-01 7.627250e-03 3.813625e-03  1.034080e-01 1.276813e-01
                   full_kv         2.5               hidden_mse 8.335987e-02 4.118469e-03 2.059234e-03  7.680646e-02 8.991327e-02
                   full_kv           3               hidden_mse 3.341216e-02 2.848775e-03 1.424387e-03  2.887912e-02 3.794520e-02
                   full_kv         3.5               hidden_mse 2.294051e-02 9.269252e-04 4.634626e-04  2.146557e-02 2.441546e-02
                   full_kv           4               hidden_mse 9.275340e-03 7.288967e-04 3.644484e-04  8.115502e-03 1.043518e-02
                     exact       exact  logit_cosine_similarity 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
           key_only_random           2  logit_cosine_similarity 9.476702e-01 3.149325e-03 1.574662e-03  9.426589e-01 9.526814e-01
           key_only_random         2.5  logit_cosine_similarity 9.704896e-01 4.456521e-03 2.228261e-03  9.633983e-01 9.775809e-01
           key_only_random           3  logit_cosine_similarity 9.814544e-01 1.876528e-03 9.382640e-04  9.784684e-01 9.844404e-01
           key_only_random         3.5  logit_cosine_similarity 9.865526e-01 1.111068e-03 5.555342e-04  9.847847e-01 9.883206e-01
           key_only_random           4  logit_cosine_similarity 9.948314e-01 9.808550e-04 4.904275e-04  9.932706e-01 9.963922e-01
 key_only_block_so8_static           2  logit_cosine_similarity 9.520647e-01 4.126013e-03 2.063007e-03  9.454993e-01 9.586301e-01
 key_only_block_so8_static         2.5  logit_cosine_similarity 9.715194e-01 1.929699e-03 9.648496e-04  9.684488e-01 9.745900e-01
 key_only_block_so8_static           3  logit_cosine_similarity 9.827781e-01 9.279485e-04 4.639743e-04  9.813015e-01 9.842546e-01
 key_only_block_so8_static         3.5  logit_cosine_similarity 9.871237e-01 8.824849e-04 4.412425e-04  9.857195e-01 9.885279e-01
 key_only_block_so8_static           4  logit_cosine_similarity 9.948779e-01 7.171704e-04 3.585852e-04  9.937368e-01 9.960191e-01
key_only_block_so8_learned           2  logit_cosine_similarity 9.475698e-01 3.498302e-03 1.749151e-03  9.420032e-01 9.531363e-01
key_only_block_so8_learned         2.5  logit_cosine_similarity 9.737917e-01 1.450250e-03 7.251250e-04  9.714840e-01 9.760994e-01
key_only_block_so8_learned           3  logit_cosine_similarity 9.830729e-01 2.447270e-03 1.223635e-03  9.791788e-01 9.869671e-01
key_only_block_so8_learned         3.5  logit_cosine_similarity 9.874147e-01 1.071787e-03 5.358933e-04  9.857093e-01 9.891202e-01
key_only_block_so8_learned           4  logit_cosine_similarity 9.944991e-01 2.262614e-04 1.131307e-04  9.941390e-01 9.948591e-01
               protected_v           2  logit_cosine_similarity 9.475698e-01 3.498302e-03 1.749151e-03  9.420032e-01 9.531363e-01
               protected_v         2.5  logit_cosine_similarity 9.737917e-01 1.450250e-03 7.251250e-04  9.714840e-01 9.760994e-01
               protected_v           3  logit_cosine_similarity 9.830729e-01 2.447270e-03 1.223635e-03  9.791788e-01 9.869671e-01
               protected_v         3.5  logit_cosine_similarity 9.874147e-01 1.071787e-03 5.358933e-04  9.857093e-01 9.891202e-01
               protected_v           4  logit_cosine_similarity 9.944991e-01 2.262614e-04 1.131307e-04  9.941390e-01 9.948591e-01
       protected_v_lowrank           2  logit_cosine_similarity 9.475698e-01 3.498302e-03 1.749151e-03  9.420032e-01 9.531363e-01
       protected_v_lowrank         2.5  logit_cosine_similarity 9.737917e-01 1.450250e-03 7.251250e-04  9.714840e-01 9.760994e-01
       protected_v_lowrank           3  logit_cosine_similarity 9.830729e-01 2.447270e-03 1.223635e-03  9.791788e-01 9.869671e-01
       protected_v_lowrank         3.5  logit_cosine_similarity 9.874147e-01 1.071787e-03 5.358933e-04  9.857093e-01 9.891202e-01
       protected_v_lowrank           4  logit_cosine_similarity 9.944991e-01 2.262614e-04 1.131307e-04  9.941390e-01 9.948591e-01
                   full_kv           2  logit_cosine_similarity 9.476702e-01 3.149325e-03 1.574662e-03  9.426589e-01 9.526814e-01
                   full_kv         2.5  logit_cosine_similarity 9.704896e-01 4.456521e-03 2.228261e-03  9.633983e-01 9.775809e-01
                   full_kv           3  logit_cosine_similarity 9.814544e-01 1.876528e-03 9.382640e-04  9.784684e-01 9.844404e-01
                   full_kv         3.5  logit_cosine_similarity 9.865526e-01 1.111068e-03 5.555342e-04  9.847847e-01 9.883206e-01
                   full_kv           4  logit_cosine_similarity 9.948314e-01 9.808550e-04 4.904275e-04  9.932706e-01 9.963922e-01
                     exact       exact         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
           key_only_random           2         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
           key_only_random         2.5         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
           key_only_random           3         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
           key_only_random         3.5         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
           key_only_random           4         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
 key_only_block_so8_static           2         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
 key_only_block_so8_static         2.5         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
 key_only_block_so8_static           3         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
 key_only_block_so8_static         3.5         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
 key_only_block_so8_static           4         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
key_only_block_so8_learned           2         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
key_only_block_so8_learned         2.5         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
key_only_block_so8_learned           3         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
key_only_block_so8_learned         3.5         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
key_only_block_so8_learned           4         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
               protected_v           2         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
               protected_v         2.5         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
               protected_v           3         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
               protected_v         3.5         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
               protected_v           4         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
       protected_v_lowrank           2         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
       protected_v_lowrank         2.5         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
       protected_v_lowrank           3         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
       protected_v_lowrank         3.5         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
       protected_v_lowrank           4         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
                   full_kv           2         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
                   full_kv         2.5         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
                   full_kv           3         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
                   full_kv         3.5         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
                   full_kv           4         logit_top1_match 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
                     exact       exact       logit_top5_overlap 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00
           key_only_random           2       logit_top5_overlap 6.250000e-01 4.082479e-02 2.041240e-02  5.600387e-01 6.899614e-01
           key_only_random         2.5       logit_top5_overlap 7.000000e-01 1.099242e-01 5.496212e-02  5.250860e-01 8.749140e-01
           key_only_random           3       logit_top5_overlap 7.625000e-01 7.772813e-02 3.886406e-02  6.388172e-01 8.861828e-01
           key_only_random         3.5       logit_top5_overlap 8.000000e-01 5.400614e-02 2.700307e-02  7.140642e-01 8.859359e-01
           key_only_random           4       logit_top5_overlap 8.875000e-01 4.330125e-02 2.165062e-02  8.185981e-01 9.564020e-01
 key_only_block_so8_static           2       logit_top5_overlap 6.812500e-01 4.269562e-02 2.134781e-02  6.133118e-01 7.491883e-01
 key_only_block_so8_static         2.5       logit_top5_overlap 7.187500e-01 6.884465e-02 3.442233e-02  6.092028e-01 8.282972e-01
 key_only_block_so8_static           3       logit_top5_overlap 8.250000e-01 4.999997e-02 2.499999e-02  7.454389e-01 9.045611e-01
 key_only_block_so8_static         3.5       logit_top5_overlap 8.312500e-01 5.153882e-02 2.576941e-02  7.492403e-01 9.132598e-01
 key_only_block_so8_static           4       logit_top5_overlap 9.125000e-01 2.500002e-02 1.250001e-02  8.727194e-01 9.522806e-01
key_only_block_so8_learned           2       logit_top5_overlap 6.250000e-01 9.354142e-02 4.677071e-02  4.761547e-01 7.738453e-01
key_only_block_so8_learned         2.5       logit_top5_overlap 7.312500e-01 7.180705e-02 3.590353e-02  6.169890e-01 8.455111e-01
key_only_block_so8_learned           3       logit_top5_overlap 8.062500e-01 2.393570e-02 1.196785e-02  7.681630e-01 8.443370e-01
key_only_block_so8_learned         3.5       logit_top5_overlap 8.625000e-01 4.330128e-02 2.165064e-02  7.935980e-01 9.314020e-01
key_only_block_so8_learned           4       logit_top5_overlap 9.000000e-01 2.041240e-02 1.020620e-02  8.675193e-01 9.324807e-01
               protected_v           2       logit_top5_overlap 6.250000e-01 9.354142e-02 4.677071e-02  4.761547e-01 7.738453e-01
               protected_v         2.5       logit_top5_overlap 7.312500e-01 7.180705e-02 3.590353e-02  6.169890e-01 8.455111e-01
               protected_v           3       logit_top5_overlap 8.062500e-01 2.393570e-02 1.196785e-02  7.681630e-01 8.443370e-01
               protected_v         3.5       logit_top5_overlap 8.625000e-01 4.330128e-02 2.165064e-02  7.935980e-01 9.314020e-01
               protected_v           4       logit_top5_overlap 9.000000e-01 2.041240e-02 1.020620e-02  8.675193e-01 9.324807e-01
       protected_v_lowrank           2       logit_top5_overlap 6.250000e-01 9.354142e-02 4.677071e-02  4.761547e-01 7.738453e-01
       protected_v_lowrank         2.5       logit_top5_overlap 7.312500e-01 7.180705e-02 3.590353e-02  6.169890e-01 8.455111e-01
       protected_v_lowrank           3       logit_top5_overlap 8.062500e-01 2.393570e-02 1.196785e-02  7.681630e-01 8.443370e-01
       protected_v_lowrank         3.5       logit_top5_overlap 8.625000e-01 4.330128e-02 2.165064e-02  7.935980e-01 9.314020e-01
       protected_v_lowrank           4       logit_top5_overlap 9.000000e-01 2.041240e-02 1.020620e-02  8.675193e-01 9.324807e-01
                   full_kv           2       logit_top5_overlap 6.250000e-01 4.082479e-02 2.041240e-02  5.600387e-01 6.899614e-01
                   full_kv         2.5       logit_top5_overlap 7.000000e-01 1.099242e-01 5.496212e-02  5.250860e-01 8.749140e-01
                   full_kv           3       logit_top5_overlap 7.625000e-01 7.772813e-02 3.886406e-02  6.388172e-01 8.861828e-01
                   full_kv         3.5       logit_top5_overlap 8.000000e-01 5.400614e-02 2.700307e-02  7.140642e-01 8.859359e-01
                   full_kv           4       logit_top5_overlap 8.875000e-01 4.330125e-02 2.165062e-02  8.185981e-01 9.564020e-01
```

## Secondary Runtime Table

```
                      mode bit_setting          metric     mean      std      sem  ci95_low  ci95_high
                     exact       exact prefill_seconds 0.010617 0.019880 0.009940 -0.021017   0.042251
           key_only_random           2 prefill_seconds 0.090487 0.172390 0.086195 -0.183824   0.364797
           key_only_random         2.5 prefill_seconds 0.006110 0.001416 0.000708  0.003857   0.008363
           key_only_random           3 prefill_seconds 0.003786 0.000864 0.000432  0.002410   0.005161
           key_only_random         3.5 prefill_seconds 0.007403 0.001223 0.000612  0.005457   0.009349
           key_only_random           4 prefill_seconds 0.004172 0.001154 0.000577  0.002335   0.006009
 key_only_block_so8_static           2 prefill_seconds 0.065075 0.121353 0.060677 -0.128025   0.258175
 key_only_block_so8_static         2.5 prefill_seconds 0.006370 0.001995 0.000998  0.003195   0.009545
 key_only_block_so8_static           3 prefill_seconds 0.004258 0.002330 0.001165  0.000550   0.007966
 key_only_block_so8_static         3.5 prefill_seconds 0.009517 0.007548 0.003774 -0.002493   0.021527
 key_only_block_so8_static           4 prefill_seconds 0.004401 0.001463 0.000731  0.002074   0.006729
key_only_block_so8_learned           2 prefill_seconds 0.005389 0.002938 0.001469  0.000715   0.010064
key_only_block_so8_learned         2.5 prefill_seconds 0.005718 0.001828 0.000914  0.002809   0.008628
key_only_block_so8_learned           3 prefill_seconds 0.005898 0.003606 0.001803  0.000161   0.011636
key_only_block_so8_learned         3.5 prefill_seconds 0.007222 0.002982 0.001491  0.002476   0.011968
key_only_block_so8_learned           4 prefill_seconds 0.003474 0.001169 0.000584  0.001614   0.005333
               protected_v           2 prefill_seconds 0.923884 1.825741 0.912871 -1.981278   3.829046
               protected_v         2.5 prefill_seconds 0.020462 0.019359 0.009680 -0.010344   0.051267
               protected_v           3 prefill_seconds 0.007035 0.002392 0.001196  0.003229   0.010841
               protected_v         3.5 prefill_seconds 0.011223 0.006503 0.003251  0.000875   0.021570
               protected_v           4 prefill_seconds 0.006825 0.001631 0.000815  0.004230   0.009420
       protected_v_lowrank           2 prefill_seconds 0.012285 0.005041 0.002520  0.004264   0.020305
       protected_v_lowrank         2.5 prefill_seconds 0.015052 0.009500 0.004750 -0.000065   0.030168
       protected_v_lowrank           3 prefill_seconds 0.026487 0.029051 0.014525 -0.019739   0.072713
       protected_v_lowrank         3.5 prefill_seconds 0.011513 0.001331 0.000666  0.009394   0.013631
       protected_v_lowrank           4 prefill_seconds 0.014306 0.011574 0.005787 -0.004112   0.032723
                   full_kv           2 prefill_seconds 0.006591 0.001969 0.000984  0.003458   0.009723
                   full_kv         2.5 prefill_seconds 0.019583 0.022081 0.011041 -0.015553   0.054719
                   full_kv           3 prefill_seconds 0.013817 0.012893 0.006447 -0.006699   0.034333
                   full_kv         3.5 prefill_seconds 0.050371 0.084662 0.042331 -0.084345   0.185087
                   full_kv           4 prefill_seconds 0.005363 0.001790 0.000895  0.002515   0.008212
                     exact       exact  decode_seconds 0.011287 0.021819 0.010910 -0.023432   0.046006
           key_only_random           2  decode_seconds 0.031333 0.058316 0.029158 -0.061461   0.124127
           key_only_random         2.5  decode_seconds 0.002248 0.000900 0.000450  0.000816   0.003680
           key_only_random           3  decode_seconds 0.002812 0.002885 0.001443 -0.001779   0.007403
           key_only_random         3.5  decode_seconds 0.002012 0.000632 0.000316  0.001006   0.003018
           key_only_random           4  decode_seconds 0.001990 0.001097 0.000549  0.000244   0.003736
 key_only_block_so8_static           2  decode_seconds 0.052682 0.102708 0.051354 -0.110748   0.216113
 key_only_block_so8_static         2.5  decode_seconds 0.002130 0.001170 0.000585  0.000269   0.003992
 key_only_block_so8_static           3  decode_seconds 0.001607 0.000748 0.000374  0.000417   0.002798
 key_only_block_so8_static         3.5  decode_seconds 0.003918 0.002741 0.001371 -0.000443   0.008280
 key_only_block_so8_static           4  decode_seconds 0.001709 0.000604 0.000302  0.000748   0.002670
key_only_block_so8_learned           2  decode_seconds 0.001985 0.001072 0.000536  0.000280   0.003691
key_only_block_so8_learned         2.5  decode_seconds 0.001924 0.000947 0.000473  0.000417   0.003430
key_only_block_so8_learned           3  decode_seconds 0.001429 0.000220 0.000110  0.001079   0.001779
key_only_block_so8_learned         3.5  decode_seconds 0.002527 0.001304 0.000652  0.000452   0.004602
key_only_block_so8_learned           4  decode_seconds 0.006548 0.011063 0.005532 -0.011056   0.024152
               protected_v           2  decode_seconds 0.001983 0.000845 0.000422  0.000639   0.003328
               protected_v         2.5  decode_seconds 0.004782 0.004574 0.002287 -0.002496   0.012061
               protected_v           3  decode_seconds 0.001328 0.000471 0.000236  0.000578   0.002077
               protected_v         3.5  decode_seconds 0.002095 0.000920 0.000460  0.000631   0.003558
               protected_v           4  decode_seconds 0.001565 0.000500 0.000250  0.000769   0.002360
       protected_v_lowrank           2  decode_seconds 0.001908 0.000665 0.000332  0.000850   0.002965
       protected_v_lowrank         2.5  decode_seconds 0.002305 0.001056 0.000528  0.000626   0.003985
       protected_v_lowrank           3  decode_seconds 0.002334 0.001260 0.000630  0.000330   0.004339
       protected_v_lowrank         3.5  decode_seconds 0.001738 0.000268 0.000134  0.001311   0.002165
       protected_v_lowrank           4  decode_seconds 0.001808 0.000983 0.000492  0.000243   0.003372
                   full_kv           2  decode_seconds 0.001353 0.000370 0.000185  0.000765   0.001941
                   full_kv         2.5  decode_seconds 0.002235 0.001337 0.000668  0.000108   0.004363
                   full_kv           3  decode_seconds 0.001897 0.001260 0.000630 -0.000108   0.003901
                   full_kv         3.5  decode_seconds 0.002091 0.000789 0.000395  0.000835   0.003347
                   full_kv           4  decode_seconds 0.001213 0.000385 0.000193  0.000599   0.001826
                     exact       exact    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
           key_only_random           2    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
           key_only_random         2.5    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
           key_only_random           3    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
           key_only_random         3.5    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
           key_only_random           4    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
 key_only_block_so8_static           2    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
 key_only_block_so8_static         2.5    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
 key_only_block_so8_static           3    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
 key_only_block_so8_static         3.5    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
 key_only_block_so8_static           4    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
key_only_block_so8_learned           2    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
key_only_block_so8_learned         2.5    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
key_only_block_so8_learned           3    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
key_only_block_so8_learned         3.5    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
key_only_block_so8_learned           4    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
               protected_v           2    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
               protected_v         2.5    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
               protected_v           3    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
               protected_v         3.5    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
               protected_v           4    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
       protected_v_lowrank           2    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
       protected_v_lowrank         2.5    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
       protected_v_lowrank           3    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
       protected_v_lowrank         3.5    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
       protected_v_lowrank           4    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
                   full_kv           2    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
                   full_kv         2.5    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
                   full_kv           3    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
                   full_kv         3.5    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
                   full_kv           4    peak_vram_mb 0.000000 0.000000 0.000000  0.000000   0.000000
```

## Summary Statistics

```
  dataset                       mode bit_setting  bits                   metric  n         mean          std          sem      ci95_low    ci95_high        model_id query_source
synthetic                      exact       exact   NaN           decode_seconds  4 1.128675e-02 2.181916e-02 1.090958e-02 -2.343239e-02 4.600589e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                      exact       exact   NaN hidden_cosine_similarity  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                      exact       exact   NaN               hidden_mae  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                      exact       exact   NaN               hidden_mse  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                      exact       exact   NaN  logit_cosine_similarity  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                      exact       exact   NaN                logit_mae  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                      exact       exact   NaN                logit_mse  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                      exact       exact   NaN           logit_spearman  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                      exact       exact   NaN         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                      exact       exact   NaN         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                      exact       exact   NaN       logit_top5_overlap  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                      exact       exact   NaN              memory_bits  4 5.242880e+05 0.000000e+00 0.000000e+00  5.242880e+05 5.242880e+05 Qwen/Qwen3.5-9B    synthetic
synthetic                      exact       exact   NaN    memory_ratio_vs_exact  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                      exact       exact   NaN             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                      exact       exact   NaN          prefill_seconds  4 1.061710e-02 1.988015e-02 9.940073e-03 -2.101665e-02 4.225085e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0           decode_seconds  4 1.353275e-03 3.695212e-04 1.847606e-04  7.652843e-04 1.941266e-03 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0 hidden_cosine_similarity  4 9.421799e-01 3.195437e-03 1.597718e-03  9.370952e-01 9.472645e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0               hidden_mae  4 2.700665e-01 7.652308e-03 3.826154e-03  2.578900e-01 2.822430e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0               hidden_mse  4 1.155447e-01 7.627250e-03 3.813625e-03  1.034080e-01 1.276813e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0  logit_cosine_similarity  4 9.476702e-01 3.149325e-03 1.574662e-03  9.426589e-01 9.526814e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0                logit_mae  4 6.891741e+00 2.054686e-01 1.027343e-01  6.564795e+00 7.218687e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0                logit_mse  4 7.550483e+01 4.639777e+00 2.319889e+00  6.812191e+01 8.288775e+01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0           logit_spearman  4 9.444034e-01 4.863393e-03 2.431696e-03  9.366647e-01 9.521422e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0       logit_top5_overlap  4 6.250000e-01 4.082479e-02 2.041240e-02  5.600387e-01 6.899614e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0              memory_bits  4 3.891200e+04 0.000000e+00 0.000000e+00  3.891200e+04 3.891200e+04 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0    memory_ratio_vs_exact  4 7.421875e-02 0.000000e+00 0.000000e+00  7.421875e-02 7.421875e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0          prefill_seconds  4 6.590875e-03 1.968614e-03 9.843072e-04  3.458370e-03 9.723380e-03 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5           decode_seconds  4 2.235425e-03 1.336783e-03 6.683914e-04  1.083054e-04 4.362545e-03 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5 hidden_cosine_similarity  4 9.594020e-01 1.658602e-03 8.293008e-04  9.567628e-01 9.620412e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5               hidden_mae  4 2.307718e-01 4.798492e-03 2.399246e-03  2.231363e-01 2.384073e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5               hidden_mse  4 8.335987e-02 4.118469e-03 2.059234e-03  7.680646e-02 8.991327e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5  logit_cosine_similarity  4 9.704896e-01 4.456521e-03 2.228261e-03  9.633983e-01 9.775809e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5                logit_mae  4 5.063751e+00 4.049561e-01 2.024780e-01  4.419375e+00 5.708126e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5                logit_mse  4 4.014226e+01 6.043810e+00 3.021905e+00  3.052521e+01 4.975931e+01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5           logit_spearman  4 9.615615e-01 6.281442e-03 3.140721e-03  9.515664e-01 9.715567e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5       logit_top5_overlap  4 7.000000e-01 1.099242e-01 5.496212e-02  5.250860e-01 8.749140e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5              memory_bits  4 4.300800e+04 0.000000e+00 0.000000e+00  4.300800e+04 4.300800e+04 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5    memory_ratio_vs_exact  4 8.203125e-02 0.000000e+00 0.000000e+00  8.203125e-02 8.203125e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5          prefill_seconds  4 1.958302e-02 2.208142e-02 1.104071e-02 -1.555344e-02 5.471949e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0           decode_seconds  4 1.896800e-03 1.259747e-03 6.298737e-04 -1.077392e-04 3.901339e-03 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0 hidden_cosine_similarity  4 9.836487e-01 1.245004e-03 6.225019e-04  9.816676e-01 9.856298e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0               hidden_mae  4 1.451317e-01 6.018778e-03 3.009389e-03  1.355545e-01 1.547090e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0               hidden_mse  4 3.341216e-02 2.848775e-03 1.424387e-03  2.887912e-02 3.794520e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0  logit_cosine_similarity  4 9.814544e-01 1.876528e-03 9.382640e-04  9.784684e-01 9.844404e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0                logit_mae  4 4.023046e+00 1.886382e-01 9.431912e-02  3.722881e+00 4.323212e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0                logit_mse  4 2.549393e+01 2.544385e+00 1.272193e+00  2.144524e+01 2.954261e+01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0           logit_spearman  4 9.742644e-01 4.804404e-03 2.402202e-03  9.666195e-01 9.819092e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0       logit_top5_overlap  4 7.625000e-01 7.772813e-02 3.886406e-02  6.388172e-01 8.861828e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0              memory_bits  4 5.529600e+04 0.000000e+00 0.000000e+00  5.529600e+04 5.529600e+04 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0    memory_ratio_vs_exact  4 1.054688e-01 0.000000e+00 0.000000e+00  1.054688e-01 1.054688e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0          prefill_seconds  4 1.381690e-02 1.289333e-02 6.446666e-03 -6.699267e-03 3.433307e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5           decode_seconds  4 2.091175e-03 7.894738e-04 3.947369e-04  8.349460e-04 3.347404e-03 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5 hidden_cosine_similarity  4 9.888864e-01 4.129969e-04 2.064984e-04  9.882292e-01 9.895436e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5               hidden_mae  4 1.206704e-01 2.818886e-03 1.409443e-03  1.161850e-01 1.251559e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5               hidden_mse  4 2.294051e-02 9.269252e-04 4.634626e-04  2.146557e-02 2.441546e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5  logit_cosine_similarity  4 9.865526e-01 1.111068e-03 5.555342e-04  9.847847e-01 9.883206e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5                logit_mae  4 3.352737e+00 1.156045e-01 5.780226e-02  3.168784e+00 3.536690e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5                logit_mse  4 1.806536e+01 1.571144e+00 7.855720e-01  1.556532e+01 2.056540e+01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5           logit_spearman  4 9.802467e-01 3.125384e-03 1.562692e-03  9.752736e-01 9.852199e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5       logit_top5_overlap  4 8.000000e-01 5.400614e-02 2.700307e-02  7.140642e-01 8.859359e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5              memory_bits  4 5.939200e+04 0.000000e+00 0.000000e+00  5.939200e+04 5.939200e+04 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5    memory_ratio_vs_exact  4 1.132812e-01 0.000000e+00 0.000000e+00  1.132812e-01 1.132812e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5          prefill_seconds  4 5.037095e-02 8.466209e-02 4.233105e-02 -8.434533e-02 1.850872e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0           decode_seconds  4 1.212675e-03 3.853636e-04 1.926818e-04  5.994756e-04 1.825874e-03 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0 hidden_cosine_similarity  4 9.954909e-01 3.354689e-04 1.677344e-04  9.949571e-01 9.960247e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0               hidden_mae  4 7.615289e-02 2.564307e-03 1.282154e-03  7.207250e-02 8.023327e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0               hidden_mse  4 9.275340e-03 7.288967e-04 3.644484e-04  8.115502e-03 1.043518e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0  logit_cosine_similarity  4 9.948314e-01 9.808550e-04 4.904275e-04  9.932706e-01 9.963922e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0                logit_mae  4 2.077886e+00 2.141266e-01 1.070633e-01  1.737163e+00 2.418609e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0                logit_mse  4 6.925941e+00 1.490352e+00 7.451760e-01  4.554458e+00 9.297423e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0           logit_spearman  4 9.903904e-01 2.328301e-03 1.164151e-03  9.866855e-01 9.940952e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0       logit_top5_overlap  4 8.875000e-01 4.330125e-02 2.165062e-02  8.185981e-01 9.564020e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0              memory_bits  4 7.168000e+04 0.000000e+00 0.000000e+00  7.168000e+04 7.168000e+04 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0    memory_ratio_vs_exact  4 1.367188e-01 0.000000e+00 0.000000e+00  1.367188e-01 1.367188e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0          prefill_seconds  4 5.363075e-03 1.790142e-03 8.950710e-04  2.514560e-03 8.211590e-03 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0           decode_seconds  4 1.985275e-03 1.071950e-03 5.359751e-04  2.795631e-04 3.690987e-03 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0 hidden_cosine_similarity  4 9.999989e-01 1.552774e-06 7.763872e-07  9.999964e-01 1.000001e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0               hidden_mae  4 5.762692e-04 2.254708e-04 1.127354e-04  2.174948e-04 9.350436e-04 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0               hidden_mse  4 2.460368e-06 3.642159e-06 1.821080e-06 -3.335121e-06 8.255856e-06 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0  logit_cosine_similarity  4 9.475698e-01 3.498302e-03 1.749151e-03  9.420032e-01 9.531363e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0                logit_mae  4 6.987618e+00 3.039267e-01 1.519634e-01  6.504003e+00 7.471233e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0                logit_mse  4 7.767018e+01 7.972955e+00 3.986478e+00  6.498343e+01 9.035693e+01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0           logit_spearman  4 9.426843e-01 2.660934e-03 1.330467e-03  9.384502e-01 9.469185e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0       logit_top5_overlap  4 6.250000e-01 9.354142e-02 4.677071e-02  4.761547e-01 7.738453e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0              memory_bits  4 2.826240e+05 0.000000e+00 0.000000e+00  2.826240e+05 2.826240e+05 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0    memory_ratio_vs_exact  4 5.390625e-01 0.000000e+00 0.000000e+00  5.390625e-01 5.390625e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0          prefill_seconds  4 5.389325e-03 2.937842e-03 1.468921e-03  7.145621e-04 1.006409e-02 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5           decode_seconds  4 1.923750e-03 9.467297e-04 4.733649e-04  4.172917e-04 3.430208e-03 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5 hidden_cosine_similarity  4 9.999996e-01 4.802399e-07 2.401199e-07  9.999988e-01 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5               hidden_mae  4 3.971656e-04 2.545707e-04 1.272854e-04 -7.913195e-06 8.022445e-04 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5               hidden_mse  4 9.750123e-07 1.067145e-06 5.335723e-07 -7.230528e-07 2.673077e-06 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5  logit_cosine_similarity  4 9.737917e-01 1.450250e-03 7.251250e-04  9.714840e-01 9.760994e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5                logit_mae  4 4.830950e+00 2.230277e-01 1.115139e-01  4.476063e+00 5.185837e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5                logit_mse  4 3.715849e+01 2.801602e+00 1.400801e+00  3.270052e+01 4.161646e+01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5           logit_spearman  4 9.679910e-01 3.260791e-03 1.630395e-03  9.628023e-01 9.731796e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5       logit_top5_overlap  4 7.312500e-01 7.180705e-02 3.590353e-02  6.169890e-01 8.455111e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5              memory_bits  4 2.846720e+05 0.000000e+00 0.000000e+00  2.846720e+05 2.846720e+05 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5    memory_ratio_vs_exact  4 5.429688e-01 0.000000e+00 0.000000e+00  5.429688e-01 5.429688e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5          prefill_seconds  4 5.718475e-03 1.828428e-03 9.142142e-04  2.809037e-03 8.627913e-03 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0           decode_seconds  4 1.428950e-03 2.198028e-04 1.099014e-04  1.079195e-03 1.778705e-03 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0 hidden_cosine_similarity  4 9.999998e-01 1.192093e-07 5.960464e-08  9.999996e-01 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0               hidden_mae  4 2.700174e-04 9.576816e-05 4.788408e-05  1.176289e-04 4.224059e-04 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0               hidden_mse  4 2.682933e-07 1.638364e-07 8.191820e-08  7.593059e-09 5.289936e-07 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0  logit_cosine_similarity  4 9.830729e-01 2.447270e-03 1.223635e-03  9.791788e-01 9.869671e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0                logit_mae  4 3.808870e+00 2.347518e-01 1.173759e-01  3.435327e+00 4.182412e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0                logit_mse  4 2.344499e+01 3.401807e+00 1.700904e+00  1.803195e+01 2.885802e+01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0           logit_spearman  4 9.747744e-01 5.391315e-03 2.695658e-03  9.661956e-01 9.833532e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0       logit_top5_overlap  4 8.062500e-01 2.393570e-02 1.196785e-02  7.681630e-01 8.443370e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0              memory_bits  4 2.908160e+05 0.000000e+00 0.000000e+00  2.908160e+05 2.908160e+05 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0    memory_ratio_vs_exact  4 5.546875e-01 0.000000e+00 0.000000e+00  5.546875e-01 5.546875e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0          prefill_seconds  4 5.898100e-03 3.605713e-03 1.802856e-03  1.606067e-04 1.163559e-02 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5           decode_seconds  4 2.526800e-03 1.304181e-03 6.520906e-04  4.515568e-04 4.602043e-03 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5 hidden_cosine_similarity  4 9.999998e-01 1.788139e-07 8.940697e-08  9.999995e-01 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5               hidden_mae  4 2.958213e-04 5.308278e-05 2.654139e-05  2.113547e-04 3.802878e-04 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5               hidden_mse  4 3.422938e-07 1.676364e-07 8.381822e-08  7.554676e-08 6.090407e-07 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5  logit_cosine_similarity  4 9.874147e-01 1.071787e-03 5.358933e-04  9.857093e-01 9.891202e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5                logit_mae  4 3.304454e+00 1.914992e-01 9.574958e-02  2.999736e+00 3.609172e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5                logit_mse  4 1.704214e+01 1.396158e+00 6.980790e-01  1.482054e+01 1.926374e+01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5           logit_spearman  4 9.809368e-01 2.549865e-03 1.274933e-03  9.768794e-01 9.849942e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5       logit_top5_overlap  4 8.625000e-01 4.330128e-02 2.165064e-02  7.935980e-01 9.314020e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5              memory_bits  4 2.928640e+05 0.000000e+00 0.000000e+00  2.928640e+05 2.928640e+05 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5    memory_ratio_vs_exact  4 5.585938e-01 0.000000e+00 0.000000e+00  5.585938e-01 5.585938e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5          prefill_seconds  4 7.221875e-03 2.982418e-03 1.491209e-03  2.476183e-03 1.196757e-02 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0           decode_seconds  4 6.548200e-03 1.106303e-02 5.531516e-03 -1.105555e-02 2.415195e-02 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0 hidden_cosine_similarity  4 9.999999e-01 1.500017e-07 7.500087e-08  9.999997e-01 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0               hidden_mae  4 1.548026e-04 8.345554e-05 4.172777e-05  2.200618e-05 2.875989e-04 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0               hidden_mse  4 9.366999e-08 7.777750e-08 3.888875e-08 -3.009138e-08 2.174313e-07 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0  logit_cosine_similarity  4 9.944991e-01 2.262614e-04 1.131307e-04  9.941390e-01 9.948591e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0                logit_mae  4 2.150544e+00 6.881623e-02 3.440811e-02  2.041042e+00 2.260046e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0                logit_mse  4 7.510853e+00 3.203485e-01 1.601742e-01  7.001107e+00 8.020599e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0           logit_spearman  4 9.900123e-01 1.702799e-03 8.513996e-04  9.873028e-01 9.927219e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0       logit_top5_overlap  4 9.000000e-01 2.041240e-02 1.020620e-02  8.675193e-01 9.324807e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0              memory_bits  4 2.990080e+05 0.000000e+00 0.000000e+00  2.990080e+05 2.990080e+05 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0    memory_ratio_vs_exact  4 5.703125e-01 0.000000e+00 0.000000e+00  5.703125e-01 5.703125e-01 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0          prefill_seconds  4 3.473500e-03 1.168585e-03 5.842923e-04  1.614021e-03 5.332979e-03 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0           decode_seconds  4 5.268235e-02 1.027076e-01 5.135381e-02 -1.107484e-01 2.161131e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0 hidden_cosine_similarity  4 9.999995e-01 3.576279e-07 1.788139e-07  9.999989e-01 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0               hidden_mae  4 5.070624e-04 1.859485e-04 9.297423e-05  2.111769e-04 8.029479e-04 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0               hidden_mse  4 1.213255e-06 8.126604e-07 4.063302e-07 -7.986906e-08 2.506379e-06 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0  logit_cosine_similarity  4 9.520647e-01 4.126013e-03 2.063007e-03  9.454993e-01 9.586301e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0                logit_mae  4 6.765980e+00 2.831099e-01 1.415550e-01  6.315489e+00 7.216472e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0                logit_mse  4 7.254932e+01 6.405709e+00 3.202854e+00  6.235641e+01 8.274223e+01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0           logit_spearman  4 9.455705e-01 3.794897e-03 1.897449e-03  9.395320e-01 9.516090e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0       logit_top5_overlap  4 6.812500e-01 4.269562e-02 2.134781e-02  6.133118e-01 7.491883e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0              memory_bits  4 2.826240e+05 0.000000e+00 0.000000e+00  2.826240e+05 2.826240e+05 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0    memory_ratio_vs_exact  4 5.390625e-01 0.000000e+00 0.000000e+00  5.390625e-01 5.390625e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0          prefill_seconds  4 6.507460e-02 1.213532e-01 6.067662e-02 -1.280255e-01 2.581747e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5           decode_seconds  4 2.130075e-03 1.169820e-03 5.849098e-04  2.686309e-04 3.991519e-03 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5 hidden_cosine_similarity  4 9.999998e-01 2.035886e-07 1.017943e-07  9.999995e-01 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5               hidden_mae  4 3.753562e-04 1.889591e-04 9.447957e-05  7.468005e-05 6.760324e-04 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5               hidden_mse  4 5.949489e-07 5.328378e-07 2.664189e-07 -2.529150e-07 1.442813e-06 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5  logit_cosine_similarity  4 9.715194e-01 1.929699e-03 9.648496e-04  9.684488e-01 9.745900e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5                logit_mae  4 5.000299e+00 2.412283e-01 1.206142e-01  4.616451e+00 5.384147e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5                logit_mse  4 3.903775e+01 2.409356e+00 1.204678e+00  3.520393e+01 4.287157e+01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5           logit_spearman  4 9.631727e-01 2.506021e-03 1.253010e-03  9.591850e-01 9.671603e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5       logit_top5_overlap  4 7.187500e-01 6.884465e-02 3.442233e-02  6.092028e-01 8.282972e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5              memory_bits  4 2.846720e+05 0.000000e+00 0.000000e+00  2.846720e+05 2.846720e+05 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5    memory_ratio_vs_exact  4 5.429688e-01 0.000000e+00 0.000000e+00  5.429688e-01 5.429688e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5          prefill_seconds  4 6.370025e-03 1.995191e-03 9.975956e-04  3.195230e-03 9.544820e-03 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0           decode_seconds  4 1.607425e-03 7.479388e-04 3.739694e-04  4.172875e-04 2.797562e-03 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0 hidden_cosine_similarity  4 9.999996e-01 1.946680e-07 9.733398e-08  9.999993e-01 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0               hidden_mae  4 3.920274e-04 6.065398e-05 3.032699e-05  2.955134e-04 4.885414e-04 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0               hidden_mse  4 8.847929e-07 6.137360e-07 3.068680e-07 -9.179808e-08 1.861384e-06 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0  logit_cosine_similarity  4 9.827781e-01 9.279485e-04 4.639743e-04  9.813015e-01 9.842546e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0                logit_mae  4 3.862467e+00 1.039570e-01 5.197852e-02  3.697048e+00 4.027886e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0                logit_mse  4 2.359991e+01 1.059696e+00 5.298479e-01  2.191369e+01 2.528612e+01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0           logit_spearman  4 9.753414e-01 2.859373e-03 1.429686e-03  9.707915e-01 9.798913e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0       logit_top5_overlap  4 8.250000e-01 4.999997e-02 2.499999e-02  7.454389e-01 9.045611e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0              memory_bits  4 2.908160e+05 0.000000e+00 0.000000e+00  2.908160e+05 2.908160e+05 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0    memory_ratio_vs_exact  4 5.546875e-01 0.000000e+00 0.000000e+00  5.546875e-01 5.546875e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0          prefill_seconds  4 4.258225e-03 2.330332e-03 1.165166e-03  5.501468e-04 7.966303e-03 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5           decode_seconds  4 3.918175e-03 2.741068e-03 1.370534e-03 -4.434751e-04 8.279825e-03 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5 hidden_cosine_similarity  4 9.999998e-01 3.706367e-07 1.853184e-07  9.999992e-01 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5               hidden_mae  4 3.043466e-04 1.472831e-04 7.364157e-05  6.998629e-05 5.387070e-04 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5               hidden_mse  4 5.674531e-07 6.763651e-07 3.381825e-07 -5.087947e-07 1.643701e-06 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5  logit_cosine_similarity  4 9.871237e-01 8.824849e-04 4.412425e-04  9.857195e-01 9.885279e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5                logit_mae  4 3.347653e+00 1.529342e-01 7.646710e-02  3.104300e+00 3.591005e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5                logit_mse  4 1.774174e+01 1.209931e+00 6.049657e-01  1.581647e+01 1.966701e+01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5           logit_spearman  4 9.815278e-01 1.197830e-03 5.989148e-04  9.796218e-01 9.834338e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5       logit_top5_overlap  4 8.312500e-01 5.153882e-02 2.576941e-02  7.492403e-01 9.132598e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5              memory_bits  4 2.928640e+05 0.000000e+00 0.000000e+00  2.928640e+05 2.928640e+05 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5    memory_ratio_vs_exact  4 5.585938e-01 0.000000e+00 0.000000e+00  5.585938e-01 5.585938e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5          prefill_seconds  4 9.517125e-03 7.547853e-03 3.773926e-03 -2.493193e-03 2.152744e-02 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0           decode_seconds  4 1.708800e-03 6.038776e-04 3.019388e-04  7.478961e-04 2.669704e-03 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0 hidden_cosine_similarity  4 1.000000e+00 1.500017e-07 7.500087e-08  9.999998e-01 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0               hidden_mae  4 1.273578e-04 5.393078e-05 2.696539e-05  4.154190e-05 2.131737e-04 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0               hidden_mse  4 9.847738e-08 1.013840e-07 5.069201e-08 -6.284721e-08 2.598020e-07 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0  logit_cosine_similarity  4 9.948779e-01 7.171704e-04 3.585852e-04  9.937368e-01 9.960191e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0                logit_mae  4 2.084841e+00 1.871868e-01 9.359338e-02  1.786985e+00 2.382697e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0                logit_mse  4 6.910222e+00 1.147429e+00 5.737145e-01  5.084406e+00 8.736038e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0           logit_spearman  4 9.907264e-01 1.401817e-03 7.009083e-04  9.884958e-01 9.929570e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0       logit_top5_overlap  4 9.125000e-01 2.500002e-02 1.250001e-02  8.727194e-01 9.522806e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0              memory_bits  4 2.990080e+05 0.000000e+00 0.000000e+00  2.990080e+05 2.990080e+05 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0    memory_ratio_vs_exact  4 5.703125e-01 0.000000e+00 0.000000e+00  5.703125e-01 5.703125e-01 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0          prefill_seconds  4 4.401250e-03 1.462858e-03 7.314290e-04  2.073516e-03 6.728984e-03 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0           decode_seconds  4 3.133282e-02 5.831620e-02 2.915810e-02 -6.146126e-02 1.241269e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0 hidden_cosine_similarity  4 9.999993e-01 6.124601e-07 3.062300e-07  9.999983e-01 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0               hidden_mae  4 5.876542e-04 2.653480e-04 1.326740e-04  1.654264e-04 1.009882e-03 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0               hidden_mse  4 1.888114e-06 1.422744e-06 7.113718e-07 -3.757885e-07 4.152016e-06 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0  logit_cosine_similarity  4 9.476702e-01 3.149325e-03 1.574662e-03  9.426589e-01 9.526814e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0                logit_mae  4 6.891741e+00 2.054686e-01 1.027343e-01  6.564795e+00 7.218687e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0                logit_mse  4 7.550483e+01 4.639777e+00 2.319889e+00  6.812191e+01 8.288775e+01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0           logit_spearman  4 9.444034e-01 4.863393e-03 2.431696e-03  9.366647e-01 9.521422e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0       logit_top5_overlap  4 6.250000e-01 4.082479e-02 2.041240e-02  5.600387e-01 6.899614e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0              memory_bits  4 2.826240e+05 0.000000e+00 0.000000e+00  2.826240e+05 2.826240e+05 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0    memory_ratio_vs_exact  4 5.390625e-01 0.000000e+00 0.000000e+00  5.390625e-01 5.390625e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0          prefill_seconds  4 9.048677e-02 1.723896e-01 8.619479e-02 -1.838235e-01 3.647971e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5           decode_seconds  4 2.248175e-03 8.998867e-04 4.499433e-04  8.162545e-04 3.680096e-03 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5 hidden_cosine_similarity  4 9.999995e-01 7.412735e-07 3.706367e-07  9.999983e-01 1.000001e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5               hidden_mae  4 4.494121e-04 2.809057e-04 1.404528e-04  2.428476e-06 8.963957e-04 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5               hidden_mse  4 1.319156e-06 1.652239e-06 8.261196e-07 -1.309925e-06 3.948238e-06 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5  logit_cosine_similarity  4 9.704896e-01 4.456521e-03 2.228261e-03  9.633983e-01 9.775809e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5                logit_mae  4 5.063751e+00 4.049561e-01 2.024780e-01  4.419375e+00 5.708126e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5                logit_mse  4 4.014226e+01 6.043810e+00 3.021905e+00  3.052521e+01 4.975931e+01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5           logit_spearman  4 9.615615e-01 6.281442e-03 3.140721e-03  9.515664e-01 9.715567e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5       logit_top5_overlap  4 7.000000e-01 1.099242e-01 5.496212e-02  5.250860e-01 8.749140e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5              memory_bits  4 2.846720e+05 0.000000e+00 0.000000e+00  2.846720e+05 2.846720e+05 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5    memory_ratio_vs_exact  4 5.429688e-01 0.000000e+00 0.000000e+00  5.429688e-01 5.429688e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5          prefill_seconds  4 6.110225e-03 1.416018e-03 7.080092e-04  3.857024e-03 8.363426e-03 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0           decode_seconds  4 2.812225e-03 2.885278e-03 1.442639e-03 -1.778896e-03 7.403346e-03 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0 hidden_cosine_similarity  4 9.999999e-01 1.141342e-07 5.706710e-08  9.999997e-01 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0               hidden_mae  4 2.169736e-04 5.090730e-05 2.545365e-05  1.359687e-04 2.979784e-04 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0               hidden_mse  4 1.526920e-07 8.718085e-08 4.359043e-08  1.396777e-08 2.914161e-07 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0  logit_cosine_similarity  4 9.814544e-01 1.876528e-03 9.382640e-04  9.784684e-01 9.844404e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0                logit_mae  4 4.023046e+00 1.886382e-01 9.431912e-02  3.722881e+00 4.323212e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0                logit_mse  4 2.549393e+01 2.544385e+00 1.272193e+00  2.144524e+01 2.954261e+01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0           logit_spearman  4 9.742644e-01 4.804404e-03 2.402202e-03  9.666195e-01 9.819092e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0       logit_top5_overlap  4 7.625000e-01 7.772813e-02 3.886406e-02  6.388172e-01 8.861828e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0              memory_bits  4 2.908160e+05 0.000000e+00 0.000000e+00  2.908160e+05 2.908160e+05 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0    memory_ratio_vs_exact  4 5.546875e-01 0.000000e+00 0.000000e+00  5.546875e-01 5.546875e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0          prefill_seconds  4 3.785950e-03 8.644437e-04 4.322219e-04  2.410427e-03 5.161473e-03 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5           decode_seconds  4 2.012350e-03 6.321765e-04 3.160882e-04  1.006416e-03 3.018284e-03 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5 hidden_cosine_similarity  4 9.999998e-01 3.424026e-07 1.712013e-07  9.999992e-01 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5               hidden_mae  4 3.015189e-04 1.069368e-04 5.346839e-05  1.313586e-04 4.716792e-04 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5               hidden_mse  4 5.388360e-07 4.955503e-07 2.477752e-07 -2.496951e-07 1.327367e-06 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5  logit_cosine_similarity  4 9.865526e-01 1.111068e-03 5.555342e-04  9.847847e-01 9.883206e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5                logit_mae  4 3.352737e+00 1.156045e-01 5.780226e-02  3.168784e+00 3.536690e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5                logit_mse  4 1.806536e+01 1.571144e+00 7.855720e-01  1.556532e+01 2.056540e+01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5           logit_spearman  4 9.802467e-01 3.125384e-03 1.562692e-03  9.752736e-01 9.852199e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5       logit_top5_overlap  4 8.000000e-01 5.400614e-02 2.700307e-02  7.140642e-01 8.859359e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5              memory_bits  4 2.928640e+05 0.000000e+00 0.000000e+00  2.928640e+05 2.928640e+05 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5    memory_ratio_vs_exact  4 5.585938e-01 0.000000e+00 0.000000e+00  5.585938e-01 5.585938e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5          prefill_seconds  4 7.402900e-03 1.223072e-03 6.115360e-04  5.456720e-03 9.349080e-03 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0           decode_seconds  4 1.989875e-03 1.097368e-03 5.486838e-04  2.437183e-04 3.736032e-03 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0 hidden_cosine_similarity  4 9.999999e-01 2.064765e-07 1.032383e-07  9.999996e-01 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0               hidden_mae  4 1.933075e-04 8.261191e-05 4.130596e-05  6.185351e-05 3.247615e-04 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0               hidden_mse  4 2.696188e-07 3.595475e-07 1.797737e-07 -3.025015e-07 8.417391e-07 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0  logit_cosine_similarity  4 9.948314e-01 9.808550e-04 4.904275e-04  9.932706e-01 9.963922e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0                logit_mae  4 2.077886e+00 2.141266e-01 1.070633e-01  1.737163e+00 2.418609e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0                logit_mse  4 6.925941e+00 1.490352e+00 7.451760e-01  4.554458e+00 9.297423e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0           logit_spearman  4 9.903904e-01 2.328301e-03 1.164151e-03  9.866855e-01 9.940952e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0       logit_top5_overlap  4 8.875000e-01 4.330125e-02 2.165062e-02  8.185981e-01 9.564020e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0              memory_bits  4 2.990080e+05 0.000000e+00 0.000000e+00  2.990080e+05 2.990080e+05 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0    memory_ratio_vs_exact  4 5.703125e-01 0.000000e+00 0.000000e+00  5.703125e-01 5.703125e-01 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0          prefill_seconds  4 4.172075e-03 1.154254e-03 5.771268e-04  2.335400e-03 6.008750e-03 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0           decode_seconds  4 1.983200e-03 8.449156e-04 4.224578e-04  6.387507e-04 3.327649e-03 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0 hidden_cosine_similarity  4 9.556855e-01 2.592696e-03 1.296348e-03  9.515600e-01 9.598111e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0               hidden_mae  4 2.249736e-01 6.097152e-03 3.048576e-03  2.152717e-01 2.346755e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0               hidden_mse  4 8.984431e-02 5.017488e-03 2.508744e-03  8.186037e-02 9.782825e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0  logit_cosine_similarity  4 9.475698e-01 3.498302e-03 1.749151e-03  9.420032e-01 9.531363e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0                logit_mae  4 6.987618e+00 3.039267e-01 1.519634e-01  6.504003e+00 7.471233e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0                logit_mse  4 7.767018e+01 7.972955e+00 3.986478e+00  6.498343e+01 9.035693e+01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0           logit_spearman  4 9.426843e-01 2.660934e-03 1.330467e-03  9.384502e-01 9.469185e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0       logit_top5_overlap  4 6.250000e-01 9.354142e-02 4.677071e-02  4.761547e-01 7.738453e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0              memory_bits  4 7.104000e+04 0.000000e+00 0.000000e+00  7.104000e+04 7.104000e+04 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0    memory_ratio_vs_exact  4 1.354980e-01 0.000000e+00 0.000000e+00  1.354980e-01 1.354980e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0          prefill_seconds  4 9.238839e-01 1.825741e+00 9.128707e-01 -1.981278e+00 3.829046e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5           decode_seconds  4 4.782400e-03 4.574156e-03 2.287078e-03 -2.496102e-03 1.206090e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5 hidden_cosine_similarity  4 9.556810e-01 2.591929e-03 1.295964e-03  9.515567e-01 9.598054e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5               hidden_mae  4 2.249548e-01 6.115112e-03 3.057556e-03  2.152242e-01 2.346853e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5               hidden_mse  4 8.985602e-02 5.018472e-03 2.509236e-03  8.187051e-02 9.784153e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5  logit_cosine_similarity  4 9.737917e-01 1.450250e-03 7.251250e-04  9.714840e-01 9.760994e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5                logit_mae  4 4.830950e+00 2.230277e-01 1.115139e-01  4.476063e+00 5.185837e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5                logit_mse  4 3.715849e+01 2.801602e+00 1.400801e+00  3.270052e+01 4.161646e+01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5           logit_spearman  4 9.679910e-01 3.260791e-03 1.630395e-03  9.628023e-01 9.731796e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5       logit_top5_overlap  4 7.312500e-01 7.180705e-02 3.590353e-02  6.169890e-01 8.455111e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5              memory_bits  4 7.308800e+04 0.000000e+00 0.000000e+00  7.308800e+04 7.308800e+04 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5    memory_ratio_vs_exact  4 1.394043e-01 0.000000e+00 0.000000e+00  1.394043e-01 1.394043e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5          prefill_seconds  4 2.046167e-02 1.935950e-02 9.679748e-03 -1.034360e-02 5.126695e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0           decode_seconds  4 1.327500e-03 4.712953e-04 2.356477e-04  5.775640e-04 2.077436e-03 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0 hidden_cosine_similarity  4 9.872561e-01 4.479650e-04 2.239825e-04  9.865432e-01 9.879689e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0               hidden_mae  4 1.205478e-01 2.420899e-03 1.210449e-03  1.166956e-01 1.244000e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0               hidden_mse  4 2.609728e-02 8.816241e-04 4.408120e-04  2.469442e-02 2.750014e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0  logit_cosine_similarity  4 9.830729e-01 2.447270e-03 1.223635e-03  9.791788e-01 9.869671e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0                logit_mae  4 3.808870e+00 2.347518e-01 1.173759e-01  3.435327e+00 4.182412e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0                logit_mse  4 2.344499e+01 3.401807e+00 1.700904e+00  1.803195e+01 2.885802e+01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0           logit_spearman  4 9.747744e-01 5.391315e-03 2.695658e-03  9.661956e-01 9.833532e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0       logit_top5_overlap  4 8.062500e-01 2.393570e-02 1.196785e-02  7.681630e-01 8.443370e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0              memory_bits  4 8.659200e+04 0.000000e+00 0.000000e+00  8.659200e+04 8.659200e+04 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0    memory_ratio_vs_exact  4 1.651611e-01 0.000000e+00 0.000000e+00  1.651611e-01 1.651611e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0          prefill_seconds  4 7.034525e-03 2.391875e-03 1.195937e-03  3.228519e-03 1.084053e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5           decode_seconds  4 2.094525e-03 9.197429e-04 4.598714e-04  6.310088e-04 3.558041e-03 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5 hidden_cosine_similarity  4 9.872558e-01 4.475463e-04 2.237731e-04  9.865437e-01 9.879680e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5               hidden_mae  4 1.205635e-01 2.420240e-03 1.210120e-03  1.167124e-01 1.244146e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5               hidden_mse  4 2.609809e-02 8.820601e-04 4.410300e-04  2.469454e-02 2.750165e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5  logit_cosine_similarity  4 9.874147e-01 1.071787e-03 5.358933e-04  9.857093e-01 9.891202e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5                logit_mae  4 3.304454e+00 1.914992e-01 9.574958e-02  2.999736e+00 3.609172e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5                logit_mse  4 1.704214e+01 1.396158e+00 6.980790e-01  1.482054e+01 1.926374e+01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5           logit_spearman  4 9.809368e-01 2.549865e-03 1.274933e-03  9.768794e-01 9.849942e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5       logit_top5_overlap  4 8.625000e-01 4.330128e-02 2.165064e-02  7.935980e-01 9.314020e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5              memory_bits  4 8.864000e+04 0.000000e+00 0.000000e+00  8.864000e+04 8.864000e+04 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5    memory_ratio_vs_exact  4 1.690674e-01 0.000000e+00 0.000000e+00  1.690674e-01 1.690674e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5          prefill_seconds  4 1.122252e-02 6.502889e-03 3.251444e-03  8.749778e-04 2.157007e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0           decode_seconds  4 1.564600e-03 4.998126e-04 2.499063e-04  7.692866e-04 2.359913e-03 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0 hidden_cosine_similarity  4 9.964459e-01 2.164747e-04 1.082374e-04  9.961015e-01 9.967904e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0               hidden_mae  4 6.383729e-02 1.382725e-03 6.913624e-04  6.163706e-02 6.603751e-02 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0               hidden_mse  4 7.307117e-03 3.978703e-04 1.989351e-04  6.674017e-03 7.940218e-03 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0  logit_cosine_similarity  4 9.944991e-01 2.262614e-04 1.131307e-04  9.941390e-01 9.948591e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0                logit_mae  4 2.150544e+00 6.881623e-02 3.440811e-02  2.041042e+00 2.260046e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0                logit_mse  4 7.510853e+00 3.203485e-01 1.601742e-01  7.001107e+00 8.020599e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0           logit_spearman  4 9.900123e-01 1.702799e-03 8.513996e-04  9.873028e-01 9.927219e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0       logit_top5_overlap  4 9.000000e-01 2.041240e-02 1.020620e-02  8.675193e-01 9.324807e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0              memory_bits  4 1.021440e+05 0.000000e+00 0.000000e+00  1.021440e+05 1.021440e+05 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0    memory_ratio_vs_exact  4 1.948242e-01 0.000000e+00 0.000000e+00  1.948242e-01 1.948242e-01 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0          prefill_seconds  4 6.825150e-03 1.630723e-03 8.153613e-04  4.230307e-03 9.419993e-03 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0           decode_seconds  4 1.907575e-03 6.648495e-04 3.324248e-04  8.496510e-04 2.965499e-03 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0 hidden_cosine_similarity  4 9.624704e-01 2.045371e-03 1.022686e-03  9.592157e-01 9.657250e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0               hidden_mae  4 2.119964e-01 6.566563e-03 3.283281e-03  2.015475e-01 2.224452e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0               hidden_mse  4 7.624669e-02 3.510824e-03 1.755412e-03  7.066019e-02 8.183320e-02 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0  logit_cosine_similarity  4 9.475698e-01 3.498302e-03 1.749151e-03  9.420032e-01 9.531363e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0                logit_mae  4 6.987618e+00 3.039267e-01 1.519634e-01  6.504003e+00 7.471233e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0                logit_mse  4 7.767018e+01 7.972955e+00 3.986478e+00  6.498343e+01 9.035693e+01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0           logit_spearman  4 9.426843e-01 2.660934e-03 1.330467e-03  9.384502e-01 9.469185e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0       logit_top5_overlap  4 6.250000e-01 9.354142e-02 4.677071e-02  4.761547e-01 7.738453e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0              memory_bits  4 7.923200e+04 0.000000e+00 0.000000e+00  7.923200e+04 7.923200e+04 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0    memory_ratio_vs_exact  4 1.511230e-01 0.000000e+00 0.000000e+00  1.511230e-01 1.511230e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0          prefill_seconds  4 1.228460e-02 5.040591e-03 2.520296e-03  4.263895e-03 2.030531e-02 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5           decode_seconds  4 2.305475e-03 1.055723e-03 5.278616e-04  6.255838e-04 3.985366e-03 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5 hidden_cosine_similarity  4 9.624688e-01 2.045139e-03 1.022570e-03  9.592145e-01 9.657231e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5               hidden_mae  4 2.120052e-01 6.582814e-03 3.291407e-03  2.015305e-01 2.224799e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5               hidden_mse  4 7.625137e-02 3.511482e-03 1.755741e-03  7.066381e-02 8.183892e-02 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5  logit_cosine_similarity  4 9.737917e-01 1.450250e-03 7.251250e-04  9.714840e-01 9.760994e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5                logit_mae  4 4.830950e+00 2.230277e-01 1.115139e-01  4.476063e+00 5.185837e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5                logit_mse  4 3.715849e+01 2.801602e+00 1.400801e+00  3.270052e+01 4.161646e+01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5           logit_spearman  4 9.679910e-01 3.260791e-03 1.630395e-03  9.628023e-01 9.731796e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5       logit_top5_overlap  4 7.312500e-01 7.180705e-02 3.590353e-02  6.169890e-01 8.455111e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5              memory_bits  4 8.128000e+04 0.000000e+00 0.000000e+00  8.128000e+04 8.128000e+04 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5    memory_ratio_vs_exact  4 1.550293e-01 0.000000e+00 0.000000e+00  1.550293e-01 1.550293e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5          prefill_seconds  4 1.505165e-02 9.499859e-03 4.749929e-03 -6.474542e-05 3.016805e-02 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0           decode_seconds  4 2.334300e-03 1.259683e-03 6.298415e-04  3.298633e-04 4.338737e-03 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0 hidden_cosine_similarity  4 9.890233e-01 4.809704e-04 2.404852e-04  9.882580e-01 9.897886e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0               hidden_mae  4 1.138655e-01 3.438416e-03 1.719208e-03  1.083943e-01 1.193368e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0               hidden_mse  4 2.248513e-02 1.117404e-03 5.587020e-04  2.070709e-02 2.426317e-02 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0  logit_cosine_similarity  4 9.830729e-01 2.447270e-03 1.223635e-03  9.791788e-01 9.869671e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0                logit_mae  4 3.808870e+00 2.347518e-01 1.173759e-01  3.435327e+00 4.182412e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0                logit_mse  4 2.344499e+01 3.401807e+00 1.700904e+00  1.803195e+01 2.885802e+01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0           logit_spearman  4 9.747744e-01 5.391315e-03 2.695658e-03  9.661956e-01 9.833532e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0       logit_top5_overlap  4 8.062500e-01 2.393570e-02 1.196785e-02  7.681630e-01 8.443370e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0              memory_bits  4 9.478400e+04 0.000000e+00 0.000000e+00  9.478400e+04 9.478400e+04 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0    memory_ratio_vs_exact  4 1.807861e-01 0.000000e+00 0.000000e+00  1.807861e-01 1.807861e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0          prefill_seconds  4 2.648705e-02 2.905074e-02 1.452537e-02 -1.973916e-02 7.271326e-02 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5           decode_seconds  4 1.737975e-03 2.684936e-04 1.342468e-04  1.310742e-03 2.165208e-03 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5 hidden_cosine_similarity  4 9.890235e-01 4.813546e-04 2.406773e-04  9.882575e-01 9.897894e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5               hidden_mae  4 1.138661e-01 3.433044e-03 1.716522e-03  1.084034e-01 1.193289e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5               hidden_mse  4 2.248520e-02 1.117531e-03 5.587655e-04  2.070696e-02 2.426345e-02 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5  logit_cosine_similarity  4 9.874147e-01 1.071787e-03 5.358933e-04  9.857093e-01 9.891202e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5                logit_mae  4 3.304454e+00 1.914992e-01 9.574958e-02  2.999736e+00 3.609172e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5                logit_mse  4 1.704214e+01 1.396158e+00 6.980790e-01  1.482054e+01 1.926374e+01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5           logit_spearman  4 9.809368e-01 2.549865e-03 1.274933e-03  9.768794e-01 9.849942e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5       logit_top5_overlap  4 8.625000e-01 4.330128e-02 2.165064e-02  7.935980e-01 9.314020e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5              memory_bits  4 9.683200e+04 0.000000e+00 0.000000e+00  9.683200e+04 9.683200e+04 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5    memory_ratio_vs_exact  4 1.846924e-01 0.000000e+00 0.000000e+00  1.846924e-01 1.846924e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5          prefill_seconds  4 1.151260e-02 1.331478e-03 6.657391e-04  9.393921e-03 1.363128e-02 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0           decode_seconds  4 1.807875e-03 9.831986e-04 4.915993e-04  2.433867e-04 3.372363e-03 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0 hidden_cosine_similarity  4 9.969856e-01 1.808270e-04 9.041349e-05  9.966979e-01 9.972734e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0               hidden_mae  4 6.010443e-02 1.518961e-03 7.594803e-04  5.768743e-02 6.252144e-02 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0               hidden_mse  4 6.201442e-03 3.118335e-04 1.559168e-04  5.705245e-03 6.697638e-03 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0  logit_cosine_similarity  4 9.944991e-01 2.262614e-04 1.131307e-04  9.941390e-01 9.948591e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0                logit_mae  4 2.150544e+00 6.881623e-02 3.440811e-02  2.041042e+00 2.260046e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0                logit_mse  4 7.510853e+00 3.203485e-01 1.601742e-01  7.001107e+00 8.020599e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0           logit_spearman  4 9.900123e-01 1.702799e-03 8.513996e-04  9.873028e-01 9.927219e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0         logit_top1_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0         logit_top5_match  4 1.000000e+00 0.000000e+00 0.000000e+00  1.000000e+00 1.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0       logit_top5_overlap  4 9.000000e-01 2.041240e-02 1.020620e-02  8.675193e-01 9.324807e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0              memory_bits  4 1.103360e+05 0.000000e+00 0.000000e+00  1.103360e+05 1.103360e+05 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0    memory_ratio_vs_exact  4 2.104492e-01 0.000000e+00 0.000000e+00  2.104492e-01 2.104492e-01 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0             peak_vram_mb  4 0.000000e+00 0.000000e+00 0.000000e+00  0.000000e+00 0.000000e+00 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0          prefill_seconds  4 1.430565e-02 1.157438e-02 5.787192e-03 -4.111779e-03 3.272308e-02 Qwen/Qwen3.5-9B    synthetic
```

## First-Layer Thresholds

```
  dataset                       mode bit_setting  bits                   metric  threshold  n  mean  std  sem  ci95_low  ci95_high        model_id query_source
synthetic                      exact       exact   NaN hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0 hidden_cosine_similarity       0.99  2   0.0  0.0  0.0       0.0        0.0 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5 hidden_cosine_similarity       0.99  2   0.0  0.0  0.0       0.0        0.0 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0 hidden_cosine_similarity       0.99  2   0.0  0.0  0.0       0.0        0.0 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5 hidden_cosine_similarity       0.99  2   0.0  0.0  0.0       0.0        0.0 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0 hidden_cosine_similarity       0.99  2   0.0  0.0  0.0       0.0        0.0 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5 hidden_cosine_similarity       0.99  2   0.0  0.0  0.0       0.0        0.0 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0 hidden_cosine_similarity       0.99  2   0.0  0.0  0.0       0.0        0.0 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5 hidden_cosine_similarity       0.99  2   0.0  0.0  0.0       0.0        0.0 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0 hidden_cosine_similarity       0.99  2   0.0  0.0  0.0       0.0        0.0 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5 hidden_cosine_similarity       0.99  2   0.0  0.0  0.0       0.0        0.0 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0 hidden_cosine_similarity       0.99  2   0.0  0.0  0.0       0.0        0.0 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5 hidden_cosine_similarity       0.99  2   0.0  0.0  0.0       0.0        0.0 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0 hidden_cosine_similarity       0.99  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic                      exact       exact   NaN hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           2   2.0 hidden_cosine_similarity       0.95  2   0.0  0.0  0.0       0.0        0.0 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         2.5   2.5 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           3   3.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv         3.5   3.5 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic                    full_kv           4   4.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           2   2.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         2.5   2.5 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           3   3.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned         3.5   3.5 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic key_only_block_so8_learned           4   4.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           2   2.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         2.5   2.5 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           3   3.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static         3.5   3.5 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic  key_only_block_so8_static           4   4.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           2   2.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         2.5   2.5 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           3   3.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random         3.5   3.5 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic            key_only_random           4   4.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           2   2.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         2.5   2.5 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           3   3.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v         3.5   3.5 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic                protected_v           4   4.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           2   2.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         2.5   2.5 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           3   3.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank         3.5   3.5 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
synthetic        protected_v_lowrank           4   4.0 hidden_cosine_similarity       0.95  2  -1.0  0.0  0.0      -1.0       -1.0 Qwen/Qwen3.5-9B    synthetic
```
