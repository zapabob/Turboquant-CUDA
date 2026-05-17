# Triality SO(8) Rotation Audit

- Status: `pass`
- Rows audited: `4608`
- Outliers: `0`
- Orthogonality threshold: `0.01`
- Determinant threshold: `0.01`

| Bits | View | Layers | Blocks | Dtype | max orth err | mean det | max det err | Status |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 2.0 | `spinor_minus_proxy` | 8 | 256 | `bfloat16` | 5.153e-03 | 1.000116302591 | 8.535e-03 | `pass` |
| 2.0 | `spinor_plus_proxy` | 8 | 256 | `bfloat16` | 5.255e-03 | 0.999967506264 | 7.223e-03 | `pass` |
| 2.0 | `vector` | 8 | 256 | `bfloat16` | 5.186e-03 | 0.999966186743 | 8.356e-03 | `pass` |
| 2.5 | `spinor_minus_proxy` | 8 | 256 | `bfloat16` | 5.153e-03 | 1.000116302591 | 8.535e-03 | `pass` |
| 2.5 | `spinor_plus_proxy` | 8 | 256 | `bfloat16` | 5.255e-03 | 0.999967506264 | 7.223e-03 | `pass` |
| 2.5 | `vector` | 8 | 256 | `bfloat16` | 5.186e-03 | 0.999966186743 | 8.356e-03 | `pass` |
| 3.0 | `spinor_minus_proxy` | 8 | 256 | `bfloat16` | 4.911e-03 | 0.999715416940 | 8.244e-03 | `pass` |
| 3.0 | `spinor_plus_proxy` | 8 | 256 | `bfloat16` | 5.175e-03 | 0.999910841135 | 7.729e-03 | `pass` |
| 3.0 | `vector` | 8 | 256 | `bfloat16` | 4.870e-03 | 1.000145130151 | 7.628e-03 | `pass` |
| 3.5 | `spinor_minus_proxy` | 8 | 256 | `bfloat16` | 4.911e-03 | 0.999715416940 | 8.244e-03 | `pass` |
| 3.5 | `spinor_plus_proxy` | 8 | 256 | `bfloat16` | 5.175e-03 | 0.999910841135 | 7.729e-03 | `pass` |
| 3.5 | `vector` | 8 | 256 | `bfloat16` | 4.870e-03 | 1.000145130151 | 7.628e-03 | `pass` |
| 4.0 | `spinor_minus_proxy` | 8 | 256 | `bfloat16` | 5.437e-03 | 1.000064910131 | 7.682e-03 | `pass` |
| 4.0 | `spinor_plus_proxy` | 8 | 256 | `bfloat16` | 5.206e-03 | 0.999788823545 | 7.216e-03 | `pass` |
| 4.0 | `vector` | 8 | 256 | `bfloat16` | 4.754e-03 | 1.000145893930 | 6.427e-03 | `pass` |
| 8.0 | `spinor_minus_proxy` | 8 | 256 | `bfloat16` | 5.175e-03 | 0.999757523413 | 8.675e-03 | `pass` |
| 8.0 | `spinor_plus_proxy` | 8 | 256 | `bfloat16` | 5.369e-03 | 0.999977425518 | 7.266e-03 | `pass` |
| 8.0 | `vector` | 8 | 256 | `bfloat16` | 6.269e-03 | 1.000028410943 | 7.111e-03 | `pass` |
