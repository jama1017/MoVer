# Vendored Concepts DSL

- Fork: https://github.com/jama1017/Concepts
- Upstream: https://github.com/concepts-ai/Concepts
- Commit: `e75163e7e79a742c776f252df9da5759b587a85c`
- License: MIT; see `LICENSE`.
- Copied scope: the complete `concepts/dsl` Python subtree (26 files).
- Modifications: upstream absolute DSL references were rewritten into the internal `mover._vendor.concepts.dsl` namespace; `tensor_value.py` uses the directly vendored `mover._vendor.torch_index` package instead of Jactorch for `batch` and `bvindex`; and `get_simple_bool_predicate()` calls itself rather than the undefined Crow-specific helper.
- Fork changes already present at this commit suppress one logger message and remove an `ipdb` breakpoint.
- Jacinle is not vendored; Concepts imports it from its published PyPI distribution.
