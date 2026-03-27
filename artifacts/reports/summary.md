# TurboQuant Report

## Synthetic Replay

Current mathematical bottleneck: value quantization drives most of the downstream hidden-state drift. At 2 bits, learned-SO(8) key-only exceeds full-KV by 0.0578 hidden cosine. Protected-V improves over full-KV by up to 0.0135 hidden cosine.

### Synthetic Core Summary

```
experiment         mode  bits   metric  n      mean          std          sem  ci95_low  ci95_high
       mse       stage1   2.0      mse  2  0.000905 6.608906e-07 4.673202e-07  0.000899   0.000911
       mse       stage1   3.0      mse  2  0.000264 5.283996e-07 3.736350e-07  0.000259   0.000268
       mse       stage1   4.0      mse  2  0.000073 1.652844e-07 1.168737e-07  0.000071   0.000074
      prod       stage2   2.0     bias  2  0.001261 4.180168e-04 2.955825e-04 -0.002495   0.005016
      prod       stage2   2.0      mae  2  0.052679 3.028541e-04 2.141502e-04  0.049958   0.055400
      prod       stage2   2.0 variance  2  0.004336 8.562692e-05 6.054738e-05  0.003567   0.005105
      prod       stage2   3.0     bias  2  0.000979 1.334755e-03 9.438146e-04 -0.011014   0.012971
      prod       stage2   3.0      mae  2  0.030167 2.672110e-05 1.889467e-05  0.029927   0.030407
      prod       stage2   3.0 variance  2  0.001426 1.795069e-05 1.269305e-05  0.001264   0.001587
      prod       stage2   4.0     bias  2  0.000278 3.758706e-04 2.657806e-04 -0.003099   0.003655
      prod       stage2   4.0      mae  2  0.016056 5.253025e-04 3.714450e-04  0.011337   0.020776
      prod       stage2   4.0 variance  2  0.000410 2.177514e-05 1.539735e-05  0.000214   0.000606
prod_mixed stage2_mixed   2.5     bias  2  0.000580 4.189058e-04 2.962111e-04 -0.003183   0.004344
prod_mixed stage2_mixed   2.5      mae  2  0.038559 8.509897e-04 6.017406e-04  0.030913   0.046205
prod_mixed stage2_mixed   2.5 variance  2  0.002334 4.483766e-05 3.170501e-05  0.001931   0.002736
prod_mixed stage2_mixed   3.5     bias  2 -0.000146 5.574519e-04 3.941780e-04 -0.005155   0.004862
prod_mixed stage2_mixed   3.5      mae  2  0.025512 5.091431e-04 3.600186e-04  0.020938   0.030087
prod_mixed stage2_mixed   3.5 variance  2  0.001020 3.822933e-05 2.703222e-05  0.000676   0.001363
```

### Synthetic Primary Pareto Table

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

### Synthetic Secondary Runtime Table

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

### Synthetic First-Layer Thresholds

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

## Captured Replay

Current mathematical bottleneck: value quantization drives most of the downstream hidden-state drift. At 2 bits, learned-SO(8) key-only exceeds full-KV by 0.0615 hidden cosine. Protected-V improves over full-KV by up to 0.0166 hidden cosine.
Runtime recommendation: protected-V is promising but not ready.

### Captured Primary Pareto Table

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
           key_only_random           2 hidden_cosine_similarity 0.998047 0.003906 0.001953  0.991831   1.004263
           key_only_random         2.5 hidden_cosine_similarity 1.000000 0.006379 0.003189  0.989850   1.010150
           key_only_random         3.5 hidden_cosine_similarity 0.999023 0.001953 0.000977  0.995916   1.002131
           key_only_random           4 hidden_cosine_similarity 0.999023 0.001953 0.000977  0.995916   1.002131
 key_only_block_so8_static           2 hidden_cosine_similarity 1.000000 0.000000 0.000000  1.000000   1.000000
 key_only_block_so8_static         2.5 hidden_cosine_similarity 0.998047 0.009831 0.004915  0.982404   1.013689
 key_only_block_so8_static         3.5 hidden_cosine_similarity 0.999023 0.001953 0.000977  0.995916   1.002131
 key_only_block_so8_static           4 hidden_cosine_similarity 1.000000 0.000000 0.000000  1.000000   1.000000
key_only_block_so8_learned           2 hidden_cosine_similarity 1.001953 0.003906 0.001953  0.995737   1.008169
key_only_block_so8_learned         2.5 hidden_cosine_similarity 1.000977 0.004915 0.002458  0.993155   1.008798
key_only_block_so8_learned         3.5 hidden_cosine_similarity 0.998047 0.003906 0.001953  0.991831   1.004263
key_only_block_so8_learned           4 hidden_cosine_similarity 0.999023 0.001953 0.000977  0.995916   1.002131
               protected_v           2 hidden_cosine_similarity 0.957031 0.003189 0.001595  0.951956   0.962106
               protected_v         2.5 hidden_cosine_similarity 0.956055 0.005859 0.002930  0.946731   0.965378
               protected_v         3.5 hidden_cosine_similarity 0.985352 0.003740 0.001870  0.979400   0.991303
               protected_v           4 hidden_cosine_similarity 0.995117 0.001953 0.000977  0.992009   0.998225
       protected_v_lowrank           2 hidden_cosine_similarity 0.960938 0.003189 0.001595  0.955862   0.966013
       protected_v_lowrank         2.5 hidden_cosine_similarity 0.963867 0.004915 0.002458  0.956046   0.971688
       protected_v_lowrank         3.5 hidden_cosine_similarity 0.984375 0.003189 0.001595  0.979300   0.989450
       protected_v_lowrank           4 hidden_cosine_similarity 0.996094 0.003189 0.001595  0.991019   1.001169
                   full_kv           2 hidden_cosine_similarity 0.940430 0.006671 0.003336  0.929814   0.951045
                   full_kv         2.5 hidden_cosine_similarity 0.958984 0.003906 0.001953  0.952769   0.965200
                   full_kv         3.5 hidden_cosine_similarity 0.989258 0.001953 0.000977  0.986150   0.992366
                   full_kv           4 hidden_cosine_similarity 0.994141 0.002255 0.001128  0.990552   0.997729
                     exact       exact               hidden_mse 0.000000 0.000000 0.000000  0.000000   0.000000
           key_only_random           2               hidden_mse 0.000437 0.000311 0.000156 -0.000058   0.000932
           key_only_random         2.5               hidden_mse 0.000326 0.000144 0.000072  0.000097   0.000554
           key_only_random         3.5               hidden_mse 0.000093 0.000041 0.000021  0.000027   0.000159
           key_only_random           4               hidden_mse 0.000055 0.000065 0.000033 -0.000048   0.000159
 key_only_block_so8_static           2               hidden_mse 0.001389 0.000849 0.000424  0.000039   0.002740
 key_only_block_so8_static         2.5               hidden_mse 0.002375 0.002827 0.001414 -0.002124   0.006874
 key_only_block_so8_static         3.5               hidden_mse 0.000135 0.000102 0.000051 -0.000028   0.000298
 key_only_block_so8_static           4               hidden_mse 0.000121 0.000124 0.000062 -0.000077   0.000318
key_only_block_so8_learned           2               hidden_mse 0.001312 0.001480 0.000740 -0.001044   0.003668
key_only_block_so8_learned         2.5               hidden_mse 0.000796 0.000522 0.000261 -0.000036   0.001627
key_only_block_so8_learned         3.5               hidden_mse 0.000210 0.000205 0.000103 -0.000117   0.000536
key_only_block_so8_learned           4               hidden_mse 0.000089 0.000053 0.000027  0.000005   0.000174
               protected_v           2               hidden_mse 0.053467 0.012628 0.006314  0.033373   0.073561
               protected_v         2.5               hidden_mse 0.052856 0.011854 0.005927  0.033993   0.071720
               protected_v         3.5               hidden_mse 0.016296 0.003717 0.001858  0.010382   0.022211
               protected_v           4               hidden_mse 0.004536 0.000867 0.000434  0.003156   0.005916
       protected_v_lowrank           2               hidden_mse 0.045837 0.008234 0.004117  0.032735   0.058940
       protected_v_lowrank         2.5               hidden_mse 0.045410 0.007509 0.003755  0.033462   0.057359
       protected_v_lowrank         3.5               hidden_mse 0.014038 0.002231 0.001116  0.010487   0.017589
       protected_v_lowrank           4               hidden_mse 0.003925 0.000589 0.000294  0.002988   0.004862
                   full_kv           2               hidden_mse 0.070190 0.017103 0.008552  0.042975   0.097405
                   full_kv         2.5               hidden_mse 0.049255 0.011944 0.005972  0.030249   0.068262
                   full_kv         3.5               hidden_mse 0.013870 0.003163 0.001582  0.008837   0.018904
                   full_kv           4               hidden_mse 0.005646 0.001327 0.000663  0.003535   0.007757
                     exact       exact  logit_cosine_similarity 1.000000 0.000000 0.000000  1.000000   1.000000
           key_only_random           2  logit_cosine_similarity 0.997070 0.001953 0.000977  0.993962   1.000178
           key_only_random         2.5  logit_cosine_similarity 0.998047 0.003906 0.001953  0.991831   1.004263
           key_only_random         3.5  logit_cosine_similarity 0.997070 0.003740 0.001870  0.991119   1.003021
           key_only_random           4  logit_cosine_similarity 0.998047 0.003906 0.001953  0.991831   1.004263
 key_only_block_so8_static           2  logit_cosine_similarity 0.998047 0.003906 0.001953  0.991831   1.004263
 key_only_block_so8_static         2.5  logit_cosine_similarity 0.999023 0.001953 0.000977  0.995916   1.002131
 key_only_block_so8_static         3.5  logit_cosine_similarity 1.000000 0.000000 0.000000  1.000000   1.000000
 key_only_block_so8_static           4  logit_cosine_similarity 0.999023 0.001953 0.000977  0.995916   1.002131
key_only_block_so8_learned           2  logit_cosine_similarity 0.997070 0.003740 0.001870  0.991119   1.003021
key_only_block_so8_learned         2.5  logit_cosine_similarity 0.999023 0.001953 0.000977  0.995916   1.002131
key_only_block_so8_learned         3.5  logit_cosine_similarity 1.000000 0.000000 0.000000  1.000000   1.000000
key_only_block_so8_learned           4  logit_cosine_similarity 0.997070 0.003740 0.001870  0.991119   1.003021
               protected_v           2  logit_cosine_similarity 0.997070 0.003740 0.001870  0.991119   1.003021
               protected_v         2.5  logit_cosine_similarity 0.999023 0.001953 0.000977  0.995916   1.002131
               protected_v         3.5  logit_cosine_similarity 1.000000 0.000000 0.000000  1.000000   1.000000
               protected_v           4  logit_cosine_similarity 0.997070 0.003740 0.001870  0.991119   1.003021
       protected_v_lowrank           2  logit_cosine_similarity 0.997070 0.003740 0.001870  0.991119   1.003021
       protected_v_lowrank         2.5  logit_cosine_similarity 0.999023 0.001953 0.000977  0.995916   1.002131
       protected_v_lowrank         3.5  logit_cosine_similarity 1.000000 0.000000 0.000000  1.000000   1.000000
       protected_v_lowrank           4  logit_cosine_similarity 0.997070 0.003740 0.001870  0.991119   1.003021
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
           key_only_random           2       logit_top5_overlap 0.831250 0.064952 0.032476  0.727897   0.934603
           key_only_random         2.5       logit_top5_overlap 0.840625 0.051412 0.025706  0.758817   0.922434
           key_only_random         3.5       logit_top5_overlap 0.890625 0.048278 0.024139  0.813805   0.967446
           key_only_random           4       logit_top5_overlap 0.931250 0.021651 0.010825  0.896799   0.965701
 key_only_block_so8_static           2       logit_top5_overlap 0.843750 0.083229 0.041615  0.711314   0.976186
 key_only_block_so8_static         2.5       logit_top5_overlap 0.831250 0.062500 0.031250  0.731799   0.930701
 key_only_block_so8_static         3.5       logit_top5_overlap 0.881250 0.054486 0.027243  0.794550   0.967950
 key_only_block_so8_static           4       logit_top5_overlap 0.906250 0.037500 0.018750  0.846579   0.965921
key_only_block_so8_learned           2       logit_top5_overlap 0.846875 0.057168 0.028584  0.755907   0.937843
key_only_block_so8_learned         2.5       logit_top5_overlap 0.868750 0.048412 0.024206  0.791715   0.945785
key_only_block_so8_learned         3.5       logit_top5_overlap 0.893750 0.055434 0.027717  0.805542   0.981958
key_only_block_so8_learned           4       logit_top5_overlap 0.921875 0.046069 0.023035  0.848568   0.995182
               protected_v           2       logit_top5_overlap 0.846875 0.057168 0.028584  0.755907   0.937843
               protected_v         2.5       logit_top5_overlap 0.868750 0.048412 0.024206  0.791715   0.945785
               protected_v         3.5       logit_top5_overlap 0.893750 0.055434 0.027717  0.805542   0.981958
               protected_v           4       logit_top5_overlap 0.921875 0.046069 0.023035  0.848568   0.995182
       protected_v_lowrank           2       logit_top5_overlap 0.846875 0.057168 0.028584  0.755907   0.937843
       protected_v_lowrank         2.5       logit_top5_overlap 0.868750 0.048412 0.024206  0.791715   0.945785
       protected_v_lowrank         3.5       logit_top5_overlap 0.893750 0.055434 0.027717  0.805542   0.981958
       protected_v_lowrank           4       logit_top5_overlap 0.921875 0.046069 0.023035  0.848568   0.995182
                   full_kv           2       logit_top5_overlap 0.831250 0.064952 0.032476  0.727897   0.934603
                   full_kv         2.5       logit_top5_overlap 0.840625 0.051412 0.025706  0.758817   0.922434
                   full_kv         3.5       logit_top5_overlap 0.890625 0.048278 0.024139  0.813805   0.967446
                   full_kv           4       logit_top5_overlap 0.931250 0.021651 0.010825  0.896799   0.965701
```

### Captured Secondary Runtime Table

```
                      mode bit_setting          metric     mean      std      sem  ci95_low  ci95_high
                     exact       exact prefill_seconds 0.000000 0.000000 0.000000  0.000000   0.000000
           key_only_random           2 prefill_seconds 0.034647 0.009683 0.004842  0.019238   0.050055
           key_only_random         2.5 prefill_seconds 0.039676 0.014640 0.007320  0.016380   0.062972
           key_only_random         3.5 prefill_seconds 0.042973 0.009926 0.004963  0.027180   0.058767
           key_only_random           4 prefill_seconds 0.036120 0.003280 0.001640  0.030901   0.041340
 key_only_block_so8_static           2 prefill_seconds 0.031094 0.009066 0.004533  0.016668   0.045520
 key_only_block_so8_static         2.5 prefill_seconds 0.036048 0.009409 0.004704  0.021077   0.051019
 key_only_block_so8_static         3.5 prefill_seconds 0.038475 0.004789 0.002394  0.030855   0.046095
 key_only_block_so8_static           4 prefill_seconds 0.036284 0.009399 0.004700  0.021328   0.051241
key_only_block_so8_learned           2 prefill_seconds 0.030461 0.007483 0.003742  0.018554   0.042369
key_only_block_so8_learned         2.5 prefill_seconds 0.049113 0.024097 0.012048  0.010770   0.087456
key_only_block_so8_learned         3.5 prefill_seconds 0.036037 0.004241 0.002121  0.029288   0.042786
key_only_block_so8_learned           4 prefill_seconds 0.032119 0.002089 0.001044  0.028796   0.035443
               protected_v           2 prefill_seconds 0.912780 1.706295 0.853147 -1.802316   3.627876
               protected_v         2.5 prefill_seconds 0.065246 0.010284 0.005142  0.048882   0.081611
               protected_v         3.5 prefill_seconds 0.073350 0.009660 0.004830  0.057978   0.088722
               protected_v           4 prefill_seconds 0.060991 0.010854 0.005427  0.043719   0.078263
       protected_v_lowrank           2 prefill_seconds 0.069836 0.019224 0.009612  0.039246   0.100426
       protected_v_lowrank         2.5 prefill_seconds 0.074441 0.011751 0.005876  0.055742   0.093139
       protected_v_lowrank         3.5 prefill_seconds 0.077075 0.004292 0.002146  0.070245   0.083905
       protected_v_lowrank           4 prefill_seconds 0.074642 0.017898 0.008949  0.046162   0.103121
                   full_kv           2 prefill_seconds 0.058801 0.015495 0.007747  0.034145   0.083457
                   full_kv         2.5 prefill_seconds 0.078370 0.015540 0.007770  0.053643   0.103098
                   full_kv         3.5 prefill_seconds 0.115939 0.093330 0.046665 -0.032570   0.264448
                   full_kv           4 prefill_seconds 0.056746 0.009955 0.004977  0.040905   0.072586
                     exact       exact  decode_seconds 0.000000 0.000000 0.000000  0.000000   0.000000
           key_only_random           2  decode_seconds 0.004757 0.001642 0.000821  0.002144   0.007370
           key_only_random         2.5  decode_seconds 0.005382 0.001505 0.000753  0.002987   0.007777
           key_only_random         3.5  decode_seconds 0.012273 0.012500 0.006250 -0.007618   0.032164
           key_only_random           4  decode_seconds 0.006416 0.003488 0.001744  0.000866   0.011965
 key_only_block_so8_static           2  decode_seconds 0.013364 0.003284 0.001642  0.008139   0.018590
 key_only_block_so8_static         2.5  decode_seconds 0.015015 0.003175 0.001587  0.009964   0.020067
 key_only_block_so8_static         3.5  decode_seconds 0.020170 0.011482 0.005741  0.001900   0.038440
 key_only_block_so8_static           4  decode_seconds 0.019059 0.006472 0.003236  0.008761   0.029358
key_only_block_so8_learned           2  decode_seconds 0.013404 0.004203 0.002101  0.006716   0.020091
key_only_block_so8_learned         2.5  decode_seconds 0.018331 0.005580 0.002790  0.009452   0.027210
key_only_block_so8_learned         3.5  decode_seconds 0.015877 0.002125 0.001062  0.012496   0.019258
key_only_block_so8_learned           4  decode_seconds 0.013432 0.001939 0.000969  0.010347   0.016516
               protected_v           2  decode_seconds 0.013170 0.003934 0.001967  0.006911   0.019429
               protected_v         2.5  decode_seconds 0.016475 0.004774 0.002387  0.008880   0.024071
               protected_v         3.5  decode_seconds 0.017952 0.003084 0.001542  0.013044   0.022860
               protected_v           4  decode_seconds 0.013004 0.003290 0.001645  0.007769   0.018238
       protected_v_lowrank           2  decode_seconds 0.014611 0.001435 0.000717  0.012328   0.016893
       protected_v_lowrank         2.5  decode_seconds 0.015879 0.003219 0.001609  0.010757   0.021001
       protected_v_lowrank         3.5  decode_seconds 0.017126 0.002754 0.001377  0.012744   0.021508
       protected_v_lowrank           4  decode_seconds 0.019266 0.009157 0.004579  0.004695   0.033837
                   full_kv           2  decode_seconds 0.005164 0.002663 0.001331  0.000926   0.009401
                   full_kv         2.5  decode_seconds 0.006679 0.003548 0.001774  0.001034   0.012324
                   full_kv         3.5  decode_seconds 0.007385 0.002094 0.001047  0.004053   0.010717
                   full_kv           4  decode_seconds 0.004069 0.000956 0.000478  0.002549   0.005590
```

### Captured First-Layer Thresholds

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
captured                    full_kv         3.5   3.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
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
captured                protected_v           4   4.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.99  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
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
captured                protected_v           2   2.0   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0 reasoning-30ae1199    reasoning 30ae119902afb94a1eec462127d49b5709480c490c445bbdfd034ca15a224192 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v           2   2.0   summary-42f56222      summary 42f562224f59c159e64b383049a08972709be617bc7dadf6a89ebec35a0a2060 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5    coding-ca766984       coding ca7669847a89be275954e36db2667fc58ca706b565d2790513a6931dfd2606f1 hidden_cosine_similarity       0.95  1  -1.0  0.0  0.0      -1.0       -1.0 H:\Qwen3.5-9B-official-hf     captured
captured                protected_v         2.5   2.5   explain-359133ba      explain 359133bab8404030a172f8efff97876f8565184962bf2f2677de73be054779c6 hidden_cosine_similarity       0.95  1   0.0  0.0  0.0       0.0        0.0 H:\Qwen3.5-9B-official-hf     captured
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
