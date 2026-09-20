# Third-party notices

The RouteLLM MF comparator follows the inference equations in lm-sys/RouteLLM, commit 0b64fdafe049e596a3f5657c219329f24af24198, routellm/routers/matrix_factorization/model.py. JevRoute changes the implementation to NumPy and a restricted F32 safetensors reader; it does not use the upstream package at runtime.

RouteLLM source is Apache-2.0 licensed. The upstream license is reproduced in licenses/RouteLLM-APACHE-2.0.txt. No upstream NOTICE file was included in the downloaded source set. Model weights and benchmark data are downloaded separately and are not bundled in wheels.
