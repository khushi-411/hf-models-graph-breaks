# gpt_oss_mxfp4 — graph breaks

- **HF id:** `openai/gpt-oss-20b`
- **Class:** `AutoModelForCausalLM`
- **Config overrides:** `{'num_hidden_layers': 2, 'layer_types': ['sliding_attention', 'full_attention']}`
- **torch:** `2.16.0a0+git57f02f8`  |  **transformers:** `5.24.0.dev0` (git `046bdb8cc2`)
- **Notes:** Top-100 #60 as shipped: MXFP4 experts via the kernels-community gpt-oss Triton kernels (gpt_oss runs it unquantized).

| Scenario | Status | Graphs | Graph breaks | Unique breaks |
|---|---|---|---|---|
| eval_forward | ok | 20 | 19 | 16 |
| eval_with_cache | ok | 21 | 20 | 12 |
| train_forward | ok | 21 | 20 | 12 |

## eval_forward

### 1. triton kernel unsupported feature

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py:41`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

triton kernel unsupported feature
  Explanation: Encountered triton kernel unsupported feature: Unexpected argument type for a Triton kernel: UserDefinedObjectVariable(Tensor).


  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(_GeneratorContextManager)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py:41`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

triton kernel unsupported feature
  Explanation: Encountered triton kernel unsupported feature: Unexpected argument type for a Triton kernel: UserDefinedObjectVariable(Tensor).


  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(_GeneratorContextManager)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/gpt_oss/modeling_gpt_oss.py line 431)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/gpt_oss/modeling_gpt_oss.py", line 475
            for i, decoder_layer in enumerate(self.layers):
                                    ~~~~~~~~~^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 3. triton kernel unsupported feature

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py:41`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

triton kernel unsupported feature
  Explanation: Encountered triton kernel unsupported feature: Unexpected argument type for a Triton kernel: UserDefinedObjectVariable(Tensor).


  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 4. Reconstruction failure

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py:41`  (hit ×1)

```
Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable TritonKernelVariable().
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: TritonKernelVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

Stack variable source attribution:
  TritonKernelVariable() originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 41
        _topk_forward[(pids, )](
        ~~~~~~~~~~~~~^^^^^^^^^^
  UserDefinedObjectVariable(Tensor) originated from:
  File "/admin/home/khushi_agrawal/Documents/pytorch/torch/_dynamo/polyfills/__init__.py", line 471
        obj = cls.__new__(cls, *args, **kwargs)
              ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  ConstantVariable(int: 32) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/tensor.py", line 136
            return self.storage.data.stride() if i is None else self.storage.data.stride(i)
                                                                ~~~~~~~~~~~~~~~~~~~~~~~~^^^
  TensorVariable() originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 26
        y_vals = torch.empty((n_rows_max, k), dtype=x.dtype, device=dev)
                 ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  TensorVariable() originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 30
            y_indx = torch.empty((n_rows_max, k), dtype=torch.int16, device=dev)
                     ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  ConstantVariable(int: 4) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 43
            y_vals, y_indx, y_vals.stride(0), use_provided_indx,  # output [topk]
                            ~~~~~~~~~~~~~^^^
  ConstantVariable(bool: False) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 31
            use_provided_indx = False
                                ^^^^^
  TensorVariable() originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 36
        bitmatrix = torch.transpose(bitmatrix, 0, 1)[:n_rows_max]
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  ConstantVariable(int: 1) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 44
            bitmatrix, bitmatrix.stride(0), bitmatrix.stride(1),  # output [bitmatrix]
                       ~~~~~~~~~~~~~~~~^^^
  ConstantVariable(int: 64) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 44
            bitmatrix, bitmatrix.stride(0), bitmatrix.stride(1),  # output [bitmatrix]
                                            ~~~~~~~~~~~~~~~~^^^
  ConstantVariable(int: 64) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 10
            x_shape = [x.shape[0] if n_rows is None else n_rows, x.shape[1]]
                       ~~~~~~~^^^
  ConstantVariable(int: 32) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 10
            x_shape = [x.shape[0] if n_rows is None else n_rows, x.shape[1]]
                                                                 ~~~~~~~^^^
  TensorVariable() originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 39
        scratchpad = torch.empty((s_cols, ), dtype=torch.int32, device=dev)
                     ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  ConstantVariable(int: 128) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 16
        BLOCK_S = 128
                  ^^^
  ConstantVariable(int: 1) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 13
        cdiv = lambda a, b: (a + b - 1) // b
                            ~~~~~~~~~~~~^^~~
  ConstantVariable(int: 32) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 14
        BLOCK_M = 32
                  ^^
  ConstantVariable(int: 32) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 15
        BLOCK_N = 32
                  ^^
  LazyVariableTracker(unrealized: <class 'bool'>) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 48
            APPLY_SOFTMAX=apply_softmax, N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,  # constants
                          ^^^^^^^^^^^^^
  ConstantVariable(int: 32) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 33
        n_cols_pad = cdiv(n_cols, BLOCK_N) * BLOCK_N
                     ~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~
  ConstantVariable(int: 4) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 26
        y_vals = torch.empty((n_rows_max, k), dtype=x.dtype, device=dev)
                              ^^^^^^^^^^
  TupleVariable(length=5) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 41
        _topk_forward[(pids, )](
        ~~~~~~~~~~~~~~~~~~~~~~~^
            x, x.stride(0),  # inputs
            ^^^^^^^^^^^^^^^^^^^^^^^^^
        ...<5 lines>...
            APPLY_SOFTMAX=apply_softmax, N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,  # constants
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        )
        ^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 5. triton kernel unsupported feature

- **Location:** `python/site-packages/triton/runtime/jit.py:375`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

triton kernel unsupported feature
  Explanation: Encountered triton kernel unsupported feature: Unexpected argument type for a Triton kernel: UserDefinedObjectVariable(Tensor).


  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 6. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `python/site-packages/triton/runtime/jit.py:734`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Unsupported python_type() call
  Explanation: This VariableTracker does not implement python_type(), which is required for object protocol operations.
  Hint: This is likely to be a Dynamo bug. Please report an issue to PyTorch.

  Developer debug context: GetAttrVariable(TritonKernelVariable(), debug) does not implement python_type()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html

Stack variable source attribution:
  GetAttrVariable(TritonKernelVariable(), debug) originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/runtime/jit.py", line 734
            kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
                                                  ^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 7. Attempted to call function marked as skipped

- **Location:** `<string>:4`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `triton._C.libtriton.native_specialize_impl.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: triton._C.libtriton, qualname: native_specialize_impl, skip reason: cannot determine source file for triton._C.libtriton (likely a C extension or builtin)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 8. Attempted to inline function marked as skipped

- **Location:** `python/site-packages/triton/runtime/jit.py:711`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Attempted to inline function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `CUDABackend.parse_options` should not be traced.
  Hint: Avoid calling the function `CUDABackend.parse_options`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `CUDABackend.parse_options` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: qualname: CUDABackend.parse_options, name: parse_options, filename: `/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/compiler.py`, skip reason: file is under skip directory (/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/compiler.py)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 9. Unsupported python_type() call

- **Location:** `python/site-packages/triton/runtime/jit.py:713`  (hit ×1)

```
Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Unsupported python_type() call
  Explanation: This VariableTracker does not implement python_type(), which is required for object protocol operations.
  Hint: This is likely to be a Dynamo bug. Please report an issue to PyTorch.

  Developer debug context: GetAttrVariable(TritonKernelVariable(), params) does not implement python_type()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html

Stack variable source attribution:
  NullVariable originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/runtime/jit.py", line 713
            sigkeys = [x.name for x in self.params]
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 10. Attempted to call function marked as skipped

- **Location:** `python/site-packages/triton/runtime/jit.py:729`  (hit ×1)

```
Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Attempted to call function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `BaseBackend.parse_attr` in file `/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/compiler.py` should not be traced.
  Hint: Avoid calling the function `BaseBackend.parse_attr`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `BaseBackend.parse_attr` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: module: triton.backends.compiler, qualname: BaseBackend.parse_attr, skip reason: file is under skip directory (/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/compiler.py)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

Stack variable source attribution:
  ConstantVariable(str: 'fpsan_homomorphic_casts') originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/runtime/jit.py", line 720
            for k in kwargs:
                     ^^^^^^
  ConstDictVariable() originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/runtime/jit.py", line 729
            attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  ListIteratorVariable(length=19, index=1) originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/runtime/jit.py", line 729
            attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}
                                                                                    ^^^^^
  TupleVariable(length=1) originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/_utils.py", line 50
                    _impl((*path, idx), item)
                          ^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 11. Attempted to inline function marked as skipped

- **Location:** `torch/_dynamo/polyfills/__init__.py:479`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Attempted to inline function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `CudaLauncher.__init__` should not be traced.
  Hint: Avoid calling the function `CudaLauncher.__init__`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `CudaLauncher.__init__` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: qualname: CudaLauncher.__init__, name: __init__, filename: `/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py`, skip reason: file is under skip directory (/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 12. Unsupported python_type() call

- **Location:** `python/site-packages/triton/compiler/compiler.py:505`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported python_type() call
  Explanation: This VariableTracker does not implement python_type(), which is required for object protocol operations.
  Hint: This is likely to be a Dynamo bug. Please report an issue to PyTorch.

  Developer debug context: GetAttrVariable(TritonKernelVariable(), arg_names) does not implement python_type()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 13. Attempted to inline function marked as skipped

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/target_info.py:13`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Attempted to inline function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `CudaDriver.get_current_target` should not be traced.
  Hint: Avoid calling the function `CudaDriver.get_current_target`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `CudaDriver.get_current_target` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: qualname: CudaDriver.get_current_target, name: get_current_target, filename: `/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py`, skip reason: file is under skip directory (/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 14. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/target_info.py:13`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Attempted to inline function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `CudaDriver.get_current_target` should not be traced.
  Hint: Avoid calling the function `CudaDriver.get_current_target`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `CudaDriver.get_current_target` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: qualname: CudaDriver.get_current_target, name: get_current_target, filename: `/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py`, skip reason: file is under skip directory (/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(_GeneratorContextManager)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

Stack variable source attribution:
  WithExitFunctionVariable() originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/integrations/mxfp4.py", line 363
            with on_device(hidden_states.device):
                 ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:443 in forward
transformers/integrations/mxfp4.py:366 in forward_routed
```
</details>

### 15. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/matmul_ogs.py:438`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable UserMethodVariable(<function Storage.is_tma_compliant at 0x77fcd6d5ddd0>, UserDefinedObjectVariable(Storage)).
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: UserMethodVariable(<function Storage.is_tma_compliant at 0x77fcd6d5ddd0>, UserDefinedObjectVariable(Storage))

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

Stack variable source attribution:
  UserMethodVariable(<function Storage.is_tma_compliant at 0x77fcd6d5ddd0>, UserDefinedObjectVariable(Storage)) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/matmul_ogs.py", line 438
        can_use_tma = x.storage.is_tma_compliant() and \
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:443 in forward
transformers/integrations/mxfp4.py:366 in forward_routed
```
</details>

### 16. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/target_info.py:13`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Attempted to inline function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `CudaDriver.get_current_target` should not be traced.
  Hint: Avoid calling the function `CudaDriver.get_current_target`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `CudaDriver.get_current_target` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: qualname: CudaDriver.get_current_target, name: get_current_target, filename: `/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py`, skip reason: file is under skip directory (/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: torch_dynamo_resume_in_make_default_opt_flags_nvidia_at_259 (/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/matmul_ogs_details/opt_flags.py line 259)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  ListIteratorVariable(length=3, index=1) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/matmul_ogs_details/opt_flags.py", line 287
        for ep in subtiles_to_check:
                  ^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:443 in forward
transformers/integrations/mxfp4.py:366 in forward_routed
```
</details>


## eval_with_cache

### 1. triton kernel unsupported feature

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py:41`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

triton kernel unsupported feature
  Explanation: Encountered triton kernel unsupported feature: Unexpected argument type for a Triton kernel: UserDefinedObjectVariable(Tensor).


  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(_GeneratorContextManager)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py:41`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

triton kernel unsupported feature
  Explanation: Encountered triton kernel unsupported feature: Unexpected argument type for a Triton kernel: UserDefinedObjectVariable(Tensor).


  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(_GeneratorContextManager)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/gpt_oss/modeling_gpt_oss.py line 431)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/gpt_oss/modeling_gpt_oss.py", line 475
            for i, decoder_layer in enumerate(self.layers):
                                    ~~~~~~~~~^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 3. triton kernel unsupported feature

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py:41`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

triton kernel unsupported feature
  Explanation: Encountered triton kernel unsupported feature: Unexpected argument type for a Triton kernel: UserDefinedObjectVariable(Tensor).


  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 4. Reconstruction failure

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py:41`  (hit ×1)

```
Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable TritonKernelVariable().
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: TritonKernelVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

Stack variable source attribution:
  TritonKernelVariable() originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 41
        _topk_forward[(pids, )](
        ~~~~~~~~~~~~~^^^^^^^^^^
  UserDefinedObjectVariable(Tensor) originated from:
  File "/admin/home/khushi_agrawal/Documents/pytorch/torch/_dynamo/polyfills/__init__.py", line 471
        obj = cls.__new__(cls, *args, **kwargs)
              ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  ConstantVariable(int: 32) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/tensor.py", line 136
            return self.storage.data.stride() if i is None else self.storage.data.stride(i)
                                                                ~~~~~~~~~~~~~~~~~~~~~~~~^^^
  TensorVariable() originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 26
        y_vals = torch.empty((n_rows_max, k), dtype=x.dtype, device=dev)
                 ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  TensorVariable() originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 30
            y_indx = torch.empty((n_rows_max, k), dtype=torch.int16, device=dev)
                     ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  ConstantVariable(int: 4) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 43
            y_vals, y_indx, y_vals.stride(0), use_provided_indx,  # output [topk]
                            ~~~~~~~~~~~~~^^^
  ConstantVariable(bool: False) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 31
            use_provided_indx = False
                                ^^^^^
  TensorVariable() originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 36
        bitmatrix = torch.transpose(bitmatrix, 0, 1)[:n_rows_max]
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  ConstantVariable(int: 1) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 44
            bitmatrix, bitmatrix.stride(0), bitmatrix.stride(1),  # output [bitmatrix]
                       ~~~~~~~~~~~~~~~~^^^
  ConstantVariable(int: 64) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 44
            bitmatrix, bitmatrix.stride(0), bitmatrix.stride(1),  # output [bitmatrix]
                                            ~~~~~~~~~~~~~~~~^^^
  ConstantVariable(int: 64) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 10
            x_shape = [x.shape[0] if n_rows is None else n_rows, x.shape[1]]
                       ~~~~~~~^^^
  ConstantVariable(int: 32) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 10
            x_shape = [x.shape[0] if n_rows is None else n_rows, x.shape[1]]
                                                                 ~~~~~~~^^^
  TensorVariable() originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 39
        scratchpad = torch.empty((s_cols, ), dtype=torch.int32, device=dev)
                     ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  ConstantVariable(int: 128) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 16
        BLOCK_S = 128
                  ^^^
  ConstantVariable(int: 1) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 13
        cdiv = lambda a, b: (a + b - 1) // b
                            ~~~~~~~~~~~~^^~~
  ConstantVariable(int: 32) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 14
        BLOCK_M = 32
                  ^^
  ConstantVariable(int: 32) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 15
        BLOCK_N = 32
                  ^^
  LazyVariableTracker(unrealized: <class 'bool'>) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 48
            APPLY_SOFTMAX=apply_softmax, N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,  # constants
                          ^^^^^^^^^^^^^
  ConstantVariable(int: 32) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 33
        n_cols_pad = cdiv(n_cols, BLOCK_N) * BLOCK_N
                     ~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~
  ConstantVariable(int: 4) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 26
        y_vals = torch.empty((n_rows_max, k), dtype=x.dtype, device=dev)
                              ^^^^^^^^^^
  TupleVariable(length=5) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 41
        _topk_forward[(pids, )](
        ~~~~~~~~~~~~~~~~~~~~~~~^
            x, x.stride(0),  # inputs
            ^^^^^^^^^^^^^^^^^^^^^^^^^
        ...<5 lines>...
            APPLY_SOFTMAX=apply_softmax, N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,  # constants
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        )
        ^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 5. triton kernel unsupported feature

- **Location:** `python/site-packages/triton/runtime/jit.py:375`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

triton kernel unsupported feature
  Explanation: Encountered triton kernel unsupported feature: Unexpected argument type for a Triton kernel: UserDefinedObjectVariable(Tensor).


  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 6. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `python/site-packages/triton/runtime/jit.py:734`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Unsupported python_type() call
  Explanation: This VariableTracker does not implement python_type(), which is required for object protocol operations.
  Hint: This is likely to be a Dynamo bug. Please report an issue to PyTorch.

  Developer debug context: GetAttrVariable(TritonKernelVariable(), debug) does not implement python_type()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html

Stack variable source attribution:
  GetAttrVariable(TritonKernelVariable(), debug) originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/runtime/jit.py", line 734
            kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
                                                  ^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 7. Attempted to call function marked as skipped

- **Location:** `<string>:4`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `triton._C.libtriton.native_specialize_impl.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: triton._C.libtriton, qualname: native_specialize_impl, skip reason: cannot determine source file for triton._C.libtriton (likely a C extension or builtin)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 8. Unsupported python_type() call

- **Location:** `python/site-packages/triton/compiler/compiler.py:505`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported python_type() call
  Explanation: This VariableTracker does not implement python_type(), which is required for object protocol operations.
  Hint: This is likely to be a Dynamo bug. Please report an issue to PyTorch.

  Developer debug context: GetAttrVariable(TritonKernelVariable(), arg_names) does not implement python_type()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 9. Attempted to inline function marked as skipped

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/target_info.py:13`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Attempted to inline function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `CudaDriver.get_current_target` should not be traced.
  Hint: Avoid calling the function `CudaDriver.get_current_target`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `CudaDriver.get_current_target` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: qualname: CudaDriver.get_current_target, name: get_current_target, filename: `/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py`, skip reason: file is under skip directory (/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 10. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/target_info.py:13`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Attempted to inline function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `CudaDriver.get_current_target` should not be traced.
  Hint: Avoid calling the function `CudaDriver.get_current_target`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `CudaDriver.get_current_target` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: qualname: CudaDriver.get_current_target, name: get_current_target, filename: `/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py`, skip reason: file is under skip directory (/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(_GeneratorContextManager)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

Stack variable source attribution:
  WithExitFunctionVariable() originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/integrations/mxfp4.py", line 363
            with on_device(hidden_states.device):
                 ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:443 in forward
transformers/integrations/mxfp4.py:366 in forward_routed
```
</details>

### 11. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/matmul_ogs.py:438`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable UserMethodVariable(<function Storage.is_tma_compliant at 0x77fcd6d5ddd0>, UserDefinedObjectVariable(Storage)).
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: UserMethodVariable(<function Storage.is_tma_compliant at 0x77fcd6d5ddd0>, UserDefinedObjectVariable(Storage))

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

Stack variable source attribution:
  UserMethodVariable(<function Storage.is_tma_compliant at 0x77fcd6d5ddd0>, UserDefinedObjectVariable(Storage)) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/matmul_ogs.py", line 438
        can_use_tma = x.storage.is_tma_compliant() and \
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:443 in forward
transformers/integrations/mxfp4.py:366 in forward_routed
```
</details>

### 12. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/target_info.py:13`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Attempted to inline function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `CudaDriver.get_current_target` should not be traced.
  Hint: Avoid calling the function `CudaDriver.get_current_target`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `CudaDriver.get_current_target` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: qualname: CudaDriver.get_current_target, name: get_current_target, filename: `/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py`, skip reason: file is under skip directory (/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: torch_dynamo_resume_in_make_default_opt_flags_nvidia_at_259 (/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/matmul_ogs_details/opt_flags.py line 259)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  ListIteratorVariable(length=3, index=1) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/matmul_ogs_details/opt_flags.py", line 287
        for ep in subtiles_to_check:
                  ^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:443 in forward
transformers/integrations/mxfp4.py:366 in forward_routed
```
</details>


## train_forward

### 1. triton kernel unsupported feature

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py:41`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

triton kernel unsupported feature
  Explanation: Encountered triton kernel unsupported feature: Unexpected argument type for a Triton kernel: UserDefinedObjectVariable(Tensor).


  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(_GeneratorContextManager)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 2. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py:41`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

triton kernel unsupported feature
  Explanation: Encountered triton kernel unsupported feature: Unexpected argument type for a Triton kernel: UserDefinedObjectVariable(Tensor).


  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(_GeneratorContextManager)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: forward (/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/gpt_oss/modeling_gpt_oss.py line 431)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  LocalGeneratorObjectVariable(enumerate) originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/models/gpt_oss/modeling_gpt_oss.py", line 475
            for i, decoder_layer in enumerate(self.layers):
                                    ~~~~~~~~~^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 3. triton kernel unsupported feature

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py:41`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_KW: a function call with keyword arguments, e.g. f(x=True):

triton kernel unsupported feature
  Higher Order Operator: autograd.Function
  Explanation: Encountered triton kernel unsupported feature: Unexpected argument type for a Triton kernel: UserDefinedObjectVariable(Tensor).


  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 4. Reconstruction failure

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py:41`  (hit ×1)

```
Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable TritonKernelVariable().
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: TritonKernelVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

Stack variable source attribution:
  TritonKernelVariable() originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 41
        _topk_forward[(pids, )](
        ~~~~~~~~~~~~~^^^^^^^^^^
  UserDefinedObjectVariable(Tensor) originated from:
  File "/admin/home/khushi_agrawal/Documents/pytorch/torch/_dynamo/polyfills/__init__.py", line 471
        obj = cls.__new__(cls, *args, **kwargs)
              ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  ConstantVariable(int: 32) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/tensor.py", line 136
            return self.storage.data.stride() if i is None else self.storage.data.stride(i)
                                                                ~~~~~~~~~~~~~~~~~~~~~~~~^^^
  TensorVariable() originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 26
        y_vals = torch.empty((n_rows_max, k), dtype=x.dtype, device=dev)
                 ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  TensorVariable() originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 30
            y_indx = torch.empty((n_rows_max, k), dtype=torch.int16, device=dev)
                     ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  ConstantVariable(int: 4) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 43
            y_vals, y_indx, y_vals.stride(0), use_provided_indx,  # output [topk]
                            ~~~~~~~~~~~~~^^^
  ConstantVariable(bool: False) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 31
            use_provided_indx = False
                                ^^^^^
  TensorVariable() originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 36
        bitmatrix = torch.transpose(bitmatrix, 0, 1)[:n_rows_max]
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  ConstantVariable(int: 1) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 44
            bitmatrix, bitmatrix.stride(0), bitmatrix.stride(1),  # output [bitmatrix]
                       ~~~~~~~~~~~~~~~~^^^
  ConstantVariable(int: 64) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 44
            bitmatrix, bitmatrix.stride(0), bitmatrix.stride(1),  # output [bitmatrix]
                                            ~~~~~~~~~~~~~~~~^^^
  ConstantVariable(int: 64) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 10
            x_shape = [x.shape[0] if n_rows is None else n_rows, x.shape[1]]
                       ~~~~~~~^^^
  ConstantVariable(int: 32) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 10
            x_shape = [x.shape[0] if n_rows is None else n_rows, x.shape[1]]
                                                                 ~~~~~~~^^^
  TensorVariable() originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 39
        scratchpad = torch.empty((s_cols, ), dtype=torch.int32, device=dev)
                     ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  ConstantVariable(int: 128) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 16
        BLOCK_S = 128
                  ^^^
  ConstantVariable(int: 1) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 13
        cdiv = lambda a, b: (a + b - 1) // b
                            ~~~~~~~~~~~~^^~~
  ConstantVariable(int: 32) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 14
        BLOCK_M = 32
                  ^^
  ConstantVariable(int: 32) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 15
        BLOCK_N = 32
                  ^^
  LazyVariableTracker(unrealized: <class 'bool'>) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 48
            APPLY_SOFTMAX=apply_softmax, N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,  # constants
                          ^^^^^^^^^^^^^
  ConstantVariable(int: 32) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 33
        n_cols_pad = cdiv(n_cols, BLOCK_N) * BLOCK_N
                     ~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~
  ConstantVariable(int: 4) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 26
        y_vals = torch.empty((n_rows_max, k), dtype=x.dtype, device=dev)
                              ^^^^^^^^^^
  TupleVariable(length=5) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/topk.py", line 41
        _topk_forward[(pids, )](
        ~~~~~~~~~~~~~~~~~~~~~~~^
            x, x.stride(0),  # inputs
            ^^^^^^^^^^^^^^^^^^^^^^^^^
        ...<5 lines>...
            APPLY_SOFTMAX=apply_softmax, N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,  # constants
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        )
        ^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 5. triton kernel unsupported feature

- **Location:** `python/site-packages/triton/runtime/jit.py:375`  (hit ×1)

```
Encountered graph break when attempting to trace CALL_FUNCTION_EX: a variadic function call, e.g. f(*args):

triton kernel unsupported feature
  Explanation: Encountered triton kernel unsupported feature: Unexpected argument type for a Triton kernel: UserDefinedObjectVariable(Tensor).


  Developer debug context: 

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0361.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 6. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `python/site-packages/triton/runtime/jit.py:734`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Unsupported python_type() call
  Explanation: This VariableTracker does not implement python_type(), which is required for object protocol operations.
  Hint: This is likely to be a Dynamo bug. Please report an issue to PyTorch.

  Developer debug context: GetAttrVariable(TritonKernelVariable(), debug) does not implement python_type()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html

Stack variable source attribution:
  GetAttrVariable(TritonKernelVariable(), debug) originated from:
  File "/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/runtime/jit.py", line 734
            kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug
                                                  ^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 7. Attempted to call function marked as skipped

- **Location:** `<string>:4`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `triton._C.libtriton.native_specialize_impl.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: triton._C.libtriton, qualname: native_specialize_impl, skip reason: cannot determine source file for triton._C.libtriton (likely a C extension or builtin)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 8. Unsupported python_type() call

- **Location:** `python/site-packages/triton/compiler/compiler.py:505`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Unsupported python_type() call
  Explanation: This VariableTracker does not implement python_type(), which is required for object protocol operations.
  Hint: This is likely to be a Dynamo bug. Please report an issue to PyTorch.

  Developer debug context: GetAttrVariable(TritonKernelVariable(), arg_names) does not implement python_type()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4841.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 9. Attempted to inline function marked as skipped

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/target_info.py:13`  (hit ×1)

```
Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Attempted to inline function marked as skipped
  Higher Order Operator: autograd.Function
  Explanation: Dynamo developers have intentionally marked that the function `CudaDriver.get_current_target` should not be traced.
  Hint: Avoid calling the function `CudaDriver.get_current_target`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `CudaDriver.get_current_target` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: qualname: CudaDriver.get_current_target, name: get_current_target, filename: `/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py`, skip reason: file is under skip directory (/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

<details><summary>User stack</summary>

```
transformers/utils/generic.py:1036 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:440 in forward
```
</details>

### 10. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/target_info.py:13`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Attempted to inline function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `CudaDriver.get_current_target` should not be traced.
  Hint: Avoid calling the function `CudaDriver.get_current_target`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `CudaDriver.get_current_target` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: qualname: CudaDriver.get_current_target, name: get_current_target, filename: `/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py`, skip reason: file is under skip directory (/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(_GeneratorContextManager)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

Stack variable source attribution:
  WithExitFunctionVariable() originated from:
  File "/admin/home/khushi_agrawal/Documents/transformers/src/transformers/integrations/mxfp4.py", line 363
            with on_device(hidden_states.device):
                 ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:443 in forward
transformers/integrations/mxfp4.py:366 in forward_routed
```
</details>

### 11. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/matmul_ogs.py:438`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable UserMethodVariable(<function Storage.is_tma_compliant at 0x77fcd6d5ddd0>, UserDefinedObjectVariable(Storage)).
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: UserMethodVariable(<function Storage.is_tma_compliant at 0x77fcd6d5ddd0>, UserDefinedObjectVariable(Storage))

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

Stack variable source attribution:
  UserMethodVariable(<function Storage.is_tma_compliant at 0x77fcd6d5ddd0>, UserDefinedObjectVariable(Storage)) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/matmul_ogs.py", line 438
        can_use_tma = x.storage.is_tma_compliant() and \
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:443 in forward
transformers/integrations/mxfp4.py:366 in forward_routed
```
</details>

### 12. Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

- **Location:** `/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/target_info.py:13`  (hit ×1)

```
Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Attempted to inline function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `CudaDriver.get_current_target` should not be traced.
  Hint: Avoid calling the function `CudaDriver.get_current_target`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `CudaDriver.get_current_target` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: qualname: CudaDriver.get_current_target, name: get_current_target, filename: `/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py`, skip reason: file is under skip directory (/admin/home/khushi_agrawal/miniforge3/envs/torch-dev-quant/lib/python3.15/site-packages/triton/backends/nvidia/driver.py)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: torch_dynamo_resume_in_make_default_opt_flags_nvidia_at_259 (/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/matmul_ogs_details/opt_flags.py line 259)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  ListIteratorVariable(length=3, index=1) originated from:
  File "/fsx/khushi_agrawal/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/9d361800edf5e0ceac59b613bac2cf9c50795f03/build/torch-cuda/matmul_ogs_details/opt_flags.py", line 287
        for ep in subtiles_to_check:
                  ^^^^^^^^^^^^^^^^^
```

Docs: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

<details><summary>User stack</summary>

```
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:622 in forward
transformers/utils/output_capturing.py:361 in wrapper
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:476 in forward
transformers/modeling_layers.py:124 in forward_with_checkpointing
transformers/utils/deprecation.py:173 in wrapped_func
transformers/models/gpt_oss/modeling_gpt_oss.py:368 in forward
transformers/integrations/mxfp4.py:443 in forward
transformers/integrations/mxfp4.py:366 in forward_routed
```
</details>
