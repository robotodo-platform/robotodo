# SPDX-License-Identifier: Apache-2.0

"""
USDRT utilities.
"""


import warp
import torch
from typing import Any


@warp.kernel(enable_backward=False)
def _fabric_ragged_compute_lengths_kernel(
    a: warp.fabricarrayarray(), 
    indices: Any,
    lengths: warp.array(),
):
    i = warp.tid()
    lengths[i] = lengths.dtype(len(a[indices[i]]))


@warp.kernel(enable_backward=False)
def _fabric_pack_kernel(
    a: warp.fabricarrayarray(), 
    indices: Any,
    lengths: warp.array(),
    offsets: warp.array(),
    a_packed: warp.array(),
):
    i, j = warp.tid()
    if j < lengths[i]:
        a_packed[offsets[i] + offsets[i].dtype(j)] = a[indices[i]][j]


def fabric_read_torch_jagged(
    a: warp.fabricarrayarray(),
    indices: warp.array(),
):
    lengths = warp.empty(len(a), dtype=warp.int64)
    warp.launch(
        _fabric_ragged_compute_lengths_kernel, 
        dim=len(a), 
        inputs=[a, indices], 
        outputs=[lengths],
        stream=warp.stream_from_torch(),
    )
    lengths_pt = warp.to_torch(lengths)
    offsets_pt = torch.nn.functional.pad(torch.cumsum(lengths_pt, dim=0), (1, 0))
    
    a_packed = warp.empty(torch.sum(lengths_pt).reshape(1), dtype=a.dtype)
    warp.launch(
        _fabric_pack_kernel,
        dim=[len(a), torch.max(lengths_pt)],
        inputs=[a, indices, lengths, warp.from_torch(offsets_pt)],
        outputs=[a_packed],
        stream=warp.stream_from_torch(),
    )

    return torch.nested.nested_tensor_from_jagged(
        values=warp.to_torch(a_packed),
        # lengths=lengths_pt,
        offsets=offsets_pt,
    )


@warp.kernel(enable_backward=False)
def _fabric_copy_kernel(
    a: warp.fabricarray(), 
    indices: Any,
    a_out: warp.array(),
):
    i = warp.tid()
    a_out[i] = a[indices[i]]


def fabric_read_torch(
    a: warp.fabricarray(),
    indices: warp.array(),
):
    return warp.to_torch(a.contiguous())[warp.to_torch(indices.contiguous()).to(torch.long)]
    # TODO why is warp slower????
    # a_out = warp.empty_like(a)
    # warp.launch(
    #     _fabric_copy_kernel,
    #     dim=[len(a)],
    #     inputs=[a, indices],
    #     outputs=[a_out],
    # )
    # return warp.to_torch(a_out)


def fabric_write_torch(
    a: warp.fabricarray(),
    indices: warp.array(),
    a_in: torch.Tensor,
):
    return a.assign(
        warp.from_torch(
            a_in[warp.to_torch(indices.contiguous()).to(torch.long)],
            dtype=a.dtype,
        )
    )
