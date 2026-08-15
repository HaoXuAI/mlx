.. _custom_cuda_kernels:

Custom CUDA Kernels
===================

MLX supports writing custom CUDA kernels through :func:`fast.cuda_kernel`, which
compiles a kernel body at run time, and :func:`fast.precompiled_cuda_kernel`,
which takes PTX or cubin you have already compiled yourself.

.. currentmodule:: mlx.core

Simple Example
--------------

Let's write a kernel that computes ``exp`` elementwise:

.. code-block:: python

  source = """
      auto elem = cg::this_grid().thread_rank();
      out[elem] = exp(inp[elem]);
  """

  kernel = mx.fast.cuda_kernel(
      name="myexp",
      input_names=["inp"],
      output_names=["out"],
      source=source,
  )

  def exp_elementwise(a: mx.array):
      outputs = kernel(
          inputs=[a],
          grid=(a.size, 1, 1),
          threadgroup=(256, 1, 1),
          output_shapes=[a.shape],
          output_dtypes=[a.dtype],
      )
      return outputs[0]

  a = mx.random.normal(shape=(4, 16))
  b = exp_elementwise(a)
  assert mx.allclose(b, mx.exp(a))

Every kernel you create is compiled the first time it runs, so build it once
with :func:`fast.cuda_kernel` and call it many times.

.. note::
   Only pass the body of the kernel in ``source``. MLX generates the function
   signature from ``input_names`` and ``output_names``.

The generated signature for ``myexp`` above is:

.. code-block:: cpp

  namespace mlx::core::cu {

  namespace cg = cooperative_groups;

  __global__ void myexp(
      const float* inp,
      float* out) {
    auto elem = cg::this_grid().thread_rank();
    out[elem] = exp(inp[elem]);
  }

  }

``cooperative_groups`` is aliased to ``cg`` for you. Pass ``verbose=True`` when
calling the kernel to print the full generated source.

Grid and Threadgroup
--------------------

``grid`` is given in **threads, not thread blocks**, for consistency with
:func:`fast.metal_kernel`. MLX launches ``ceil(grid / threadgroup)`` blocks of
``threadgroup`` threads each, so the example above launches
``ceil(64 / 256) = 1`` block of 64 threads.

This is worth reading twice if you are used to writing CUDA launch
configurations directly, where the first argument is a block count.

Using Shape and Strides
-----------------------

Referring to ``inp_shape``, ``inp_strides`` or ``inp_ndim`` in the source adds
the corresponding parameter to the signature for that input, so a kernel can
handle non-contiguous inputs without a copy:

.. code-block:: python

  source = """
      auto elem = cg::this_grid().thread_rank();
      auto loc = elem_to_loc(elem, inp_shape.data(), inp_strides.data(), inp_ndim);
      out[elem] = inp[loc];
  """

Compile-time Constants
----------------------

Values passed through ``template`` become ``constexpr`` in the generated source
rather than runtime arguments:

.. code-block:: python

  outputs = kernel(
      inputs=[a],
      template=[("SIZE", 128)],
      ...
  )

Precompiled Kernels
-------------------

:func:`fast.precompiled_cuda_kernel` loads PTX or cubin instead of compiling a
source string. Unlike :func:`fast.cuda_kernel`, MLX does not generate a
signature here, so the kernel's parameters must match what MLX passes, in this
order:

1. one pointer per entry in ``inputs``
2. one pointer per entry in ``output_shapes``
3. the values in ``scalars``, which are runtime parameters on this path rather
   than compile-time constants

For a kernel built with ``nvcc -ptx``:

.. code-block:: cpp

  // double.cu, compiled with: nvcc -ptx -arch=sm_89 -o double.ptx double.cu
  extern "C" __global__ void my_double(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
      y[i] = x[i] * 2.0f;
    }
  }

.. code-block:: python

  n = 1024
  x = mx.random.uniform(shape=(n,))

  (y,) = mx.fast.precompiled_cuda_kernel(
      name="my_double",
      compiled_source=open("double.ptx", "rb").read(),
      inputs=[x],
      output_shapes=[(n,)],
      output_dtypes=[mx.float32],
      scalars=[n],
      grid=(n, 1, 1),
      threadgroup=(128, 1, 1),
  )
  assert mx.allclose(y, x * 2)

.. warning::
   Supplying fewer parameters than the kernel declares is undefined behaviour:
   the driver reads past the end of the argument list. Depending on what happens
   to be on the stack this shows up as ``cuGraphAddKernelNode ... invalid
   argument`` or as a segmentation fault, so check the signature of the compiled
   kernel rather than the source you wrote.

Using Triton
------------

Triton can emit PTX, so a Triton kernel can be run through
:func:`fast.precompiled_cuda_kernel`. Declare the parameters in MLX's order --
inputs, then outputs, then scalars:

.. code-block:: python

  import re

  import torch  # only to give Triton a tensor to specialise on
  import triton
  import triton.language as tl

  @triton.jit
  def _double(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
      pid = tl.program_id(axis=0)
      offs = pid * BLOCK + tl.arange(0, BLOCK)
      mask = offs < n_elements
      tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2.0, mask=mask)

  n = 1024
  spec = torch.empty(n, device="cuda", dtype=torch.float32)
  compiled = _double.warmup(spec, spec, n, BLOCK=128, grid=(1,))
  ptx = compiled.asm["ptx"]

  # Take the entry symbol and the required block size from the PTX itself.
  name = re.search(r"\.visible\s+\.entry\s+([\w$]+)", ptx).group(1)
  block = int(re.search(r"\.reqntid\s+(\d+)", ptx).group(1))

  x = mx.random.uniform(shape=(n,))
  (y,) = mx.fast.precompiled_cuda_kernel(
      name=name,
      compiled_source=ptx.encode(),
      inputs=[x],
      output_shapes=[(n,)],
      output_dtypes=[mx.float32],
      # n_elements, then padding for Triton's implicit parameters (see below)
      scalars=[n, 0, 0, 0, 0],
      grid=(n, 1, 1),
      threadgroup=(block, 1, 1),
  )
  assert mx.allclose(y, x * 2)

Two things about Triton's output do not follow from the Python source:

**The block size is fixed at compile time.** Triton derives it from
``num_warps`` (four by default, so 128 threads) and records it as ``.reqntid``
in the PTX. It is a requirement rather than an upper bound, and it is unrelated
to a ``BLOCK`` constant in your kernel, which describes how many elements each
program handles. Read ``.reqntid`` and pass exactly that as ``threadgroup``.

**Triton appends implicit parameters.** A three-argument kernel compiles to five
``.param`` entries in Triton 3.7, the last two being pointers Triton uses for
its own scratch space. MLX has no way to pass a pointer, since ``scalars`` holds
``bool``, ``int`` or ``float``, so the workaround above pads the argument list
with four ``int`` zeros to cover the sixteen bytes. Kernels that do not touch
that scratch space work; a cleaner fix would be for MLX to query the compiled
kernel's parameter list and fill the remainder itself.

Inspect the generated ``.entry`` block to see what a given Triton version
actually expects:

.. code-block:: python

  entry = ptx[ptx.index(f".visible .entry {name}"):]
  print(entry[:entry.index("{")])
