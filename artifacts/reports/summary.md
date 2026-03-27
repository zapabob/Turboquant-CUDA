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

Current mathematical bottleneck: value quantization drives most of the downstream hidden-state drift. At 2 bits, learned-SO(8) key-only exceeds full-KV by 0.0605 hidden cosine. Protected-V improves over full-KV by up to 0.0146 hidden cosine.
Runtime recommendation: protected-V is promising but not ready.

### Captured Headline

- Runtime default remains `key-only` on real captured KV.
- `full_kv` is still the memory floor, but hidden-state drift remains materially larger than the key-only baseline.
- `protected_v_lowrank` is a real middle Pareto point: better hidden retention than `full_kv`, but not yet close enough to replace `key_only_block_so8_learned`.
- `peak_vram_mb` below is replay-side additional CUDA usage for saved layer tensors, not end-to-end model inference VRAM.

### Captured Representative Comparison

| Mode | Bits | Memory / Exact | Hidden Cosine | Logit Cosine | Peak VRAM (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Key-Only (SO8 Learned) | 2.0 | 0.5664 | 1.0010 | 0.9961 | 17.64 |
| Protected-V + LR | 2.0 | 0.2122 | 0.9648 | 0.9961 | 21.06 |
| Full-KV | 2.0 | 0.1309 | 0.9404 | 0.9971 | 17.70 |
| Key-Only (SO8 Learned) | 4.0 | 0.6289 | 0.9980 | 0.9990 | 17.64 |
| Protected-V + LR | 4.0 | 0.3308 | 0.9980 | 0.9990 | 21.06 |
| Full-KV | 4.0 | 0.2559 | 0.9951 | 0.9980 | 17.70 |

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

### Captured Secondary Runtime Table

```
                      mode bit_setting          metric      mean      std      sem  ci95_low  ci95_high
                     exact       exact prefill_seconds  0.000713 0.000277 0.000139  0.000271   0.001154
           key_only_random           2 prefill_seconds  0.205184 0.393818 0.196909 -0.421468   0.831836
           key_only_random         2.5 prefill_seconds  0.013424 0.006135 0.003067  0.003662   0.023186
           key_only_random         3.5 prefill_seconds  0.011497 0.002325 0.001163  0.007797   0.015198
           key_only_random           4 prefill_seconds  0.010457 0.007652 0.003826 -0.001719   0.022632
 key_only_block_so8_static           2 prefill_seconds  0.008789 0.002149 0.001074  0.005370   0.012208
 key_only_block_so8_static         2.5 prefill_seconds  0.012957 0.004963 0.002482  0.005059   0.020855
 key_only_block_so8_static         3.5 prefill_seconds  0.011790 0.000956 0.000478  0.010269   0.013312
 key_only_block_so8_static           4 prefill_seconds  0.008793 0.002087 0.001044  0.005471   0.012114
key_only_block_so8_learned           2 prefill_seconds  0.006338 0.001076 0.000538  0.004627   0.008050
key_only_block_so8_learned         2.5 prefill_seconds  0.008937 0.000860 0.000430  0.007568   0.010306
key_only_block_so8_learned         3.5 prefill_seconds  0.008763 0.000909 0.000455  0.007316   0.010209
key_only_block_so8_learned           4 prefill_seconds  0.007540 0.002716 0.001358  0.003219   0.011861
               protected_v           2 prefill_seconds  0.504019 0.966568 0.483284 -1.034006   2.042045
               protected_v         2.5 prefill_seconds  0.019143 0.004751 0.002376  0.011582   0.026703
               protected_v         3.5 prefill_seconds  0.018293 0.003756 0.001878  0.012316   0.024270
               protected_v           4 prefill_seconds  0.017932 0.004233 0.002117  0.011196   0.024668
       protected_v_lowrank           2 prefill_seconds  0.078423 0.059139 0.029570 -0.015681   0.172526
       protected_v_lowrank         2.5 prefill_seconds  0.030303 0.006935 0.003468  0.019268   0.041339
       protected_v_lowrank         3.5 prefill_seconds  0.039057 0.020326 0.010163  0.006713   0.071400
       protected_v_lowrank           4 prefill_seconds  0.028483 0.004839 0.002419  0.020783   0.036182
                   full_kv           2 prefill_seconds  0.012549 0.002544 0.001272  0.008501   0.016597
                   full_kv         2.5 prefill_seconds  0.026739 0.011259 0.005630  0.008824   0.044655
                   full_kv         3.5 prefill_seconds  0.059155 0.062645 0.031323 -0.040527   0.158838
                   full_kv           4 prefill_seconds  0.012888 0.002798 0.001399  0.008437   0.017340
                     exact       exact  decode_seconds  0.028657 0.054202 0.027101 -0.057591   0.114905
           key_only_random           2  decode_seconds  0.003437 0.000830 0.000415  0.002116   0.004758
           key_only_random         2.5  decode_seconds  0.004230 0.001204 0.000602  0.002313   0.006146
           key_only_random         3.5  decode_seconds  0.004309 0.001684 0.000842  0.001629   0.006989
           key_only_random           4  decode_seconds  0.003475 0.001882 0.000941  0.000481   0.006469
 key_only_block_so8_static           2  decode_seconds  0.003009 0.000957 0.000478  0.001487   0.004532
 key_only_block_so8_static         2.5  decode_seconds  0.004638 0.001664 0.000832  0.001990   0.007285
 key_only_block_so8_static         3.5  decode_seconds  0.005029 0.002121 0.001061  0.001654   0.008404
 key_only_block_so8_static           4  decode_seconds  0.003432 0.000963 0.000482  0.001899   0.004964
key_only_block_so8_learned           2  decode_seconds  0.002797 0.001006 0.000503  0.001195   0.004398
key_only_block_so8_learned         2.5  decode_seconds  0.003515 0.000828 0.000414  0.002197   0.004833
key_only_block_so8_learned         3.5  decode_seconds  0.002952 0.000292 0.000146  0.002487   0.003417
key_only_block_so8_learned           4  decode_seconds  0.002809 0.001022 0.000511  0.001183   0.004436
               protected_v           2  decode_seconds  0.003815 0.001540 0.000770  0.001364   0.006266
               protected_v         2.5  decode_seconds  0.003438 0.001036 0.000518  0.001790   0.005086
               protected_v         3.5  decode_seconds  0.003215 0.001165 0.000582  0.001362   0.005068
               protected_v           4  decode_seconds  0.003676 0.001852 0.000926  0.000729   0.006622
       protected_v_lowrank           2  decode_seconds  0.004701 0.004445 0.002222 -0.002372   0.011774
       protected_v_lowrank         2.5  decode_seconds  0.003546 0.001211 0.000605  0.001619   0.005472
       protected_v_lowrank         3.5  decode_seconds  0.004763 0.002382 0.001191  0.000973   0.008554
       protected_v_lowrank           4  decode_seconds  0.002202 0.000503 0.000252  0.001401   0.003002
                   full_kv           2  decode_seconds  0.003582 0.001387 0.000694  0.001375   0.005790
                   full_kv         2.5  decode_seconds  0.004123 0.000936 0.000468  0.002634   0.005613
                   full_kv         3.5  decode_seconds  0.005006 0.002907 0.001454  0.000380   0.009632
                   full_kv           4  decode_seconds  0.002779 0.000392 0.000196  0.002155   0.003403
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
