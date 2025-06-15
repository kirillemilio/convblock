"""Mask utils contain functions useful in attention modules."""

from __future__ import annotations

import torch

from ....utils import ArrayLike, transform_to_int_tuple


def generate_shift_mask_1d(size: int, window_size: int) -> torch.Tensor:
    """
    Generate a 1D attention mask for shifted windows.

    Mask the last window in a 1D sequence to block attention between patches
    introduced by a cyclic shift.

    Parameters
    ----------
    size : int
        Number of windows along the sequence axis.
    window_size : int
        Size of each window (patch sequence length).

    Returns
    -------
    torch.Tensor
        Mask tensor of shape (size, window_size, window_size) with
        `-inf` at disallowed attention positions.
    """
    shift = window_size // 2
    mask = torch.zeros(*(size, window_size, window_size), dtype=torch.float32)
    mask[-1, :-shift, -shift:] = -torch.inf
    mask[-1, -shift:, :-shift] = -torch.inf
    return mask


def generate_shift_mask_2d(
    size: tuple[int, int], window_size: tuple[int, int]
) -> torch.Tensor:
    """
    Generate a 2D attention mask for shifted window attention.

    Apply masking to prevent attention between shifted patches in the last row
    and column of the window grid.

    Parameters
    ----------
    size : tuple of int
        Grid size as (height_windows, width_windows).
    window_size : tuple of int
        Window size as (window_height, window_width).

    Returns
    -------
    torch.Tensor
        Mask tensor of shape (H, W, wh, ww, wh, ww) with `-inf` at masked
        attention positions.
    """
    shift = (window_size[0] // 2, window_size[1] // 2)
    mask = torch.zeros(
        *(
            size[0],
            size[1],
            window_size[0],
            window_size[1],
            window_size[0],
            window_size[1],
        ),
        dtype=torch.float32,
    )

    ######################################
    # Dealing with shift along h(y) axis #
    ######################################
    mask[
        -1,  # last grid row
        :,  # all grid columns
        : -shift[0],  # within grid all rows before shift
        :,  # all columns
        -shift[0] :,  # within grid all rows alien because of shift
        :,  # all columns
    ] = -torch.inf

    mask[
        -1,  # last grid row
        :,  # all grid columns
        -shift[0] :,  # within grid all rows alien because of shift
        :,  # all columns
        : -shift[0],  # within grid all rows before shift
        :,  # all columns
    ] = -torch.inf

    ######################################
    # Dealing with shift along w(x) axis #
    ######################################
    mask[
        :,  # all grid rows
        -1,  # last grid column
        :,  # within grid all rows
        : -shift[1],  # within grid columns before shift
        :,  # within grid all rows
        -shift[1] :,  # within grid columns alien because of shift
    ] = -torch.inf

    mask[
        :,  # all grid rows
        -1,  # last grid column
        :,  # within grid all rows
        -shift[1] :,  # within grid columns alien because of shift
        :,  # within grid all rows
        : -shift[1],  # within grid columns before shift
    ] = -torch.inf

    return mask


def generate_shift_mask_3d(
    size: tuple[int, int, int], window_size: tuple[int, int, int]
) -> torch.Tensor:
    """
    Generate a 3D attention mask for shifted spatiotemporal windows.

    Mask invalid attention in the last time slice, height, and width blocks
    caused by cyclic shifts in 3D Swin-like attention.

    Parameters
    ----------
    size : tuple of int
        Grid size as (temporal_windows, height_windows, width_windows).
    window_size : tuple of int
        Window size as (window_t, window_h, window_w).

    Returns
    -------
    torch.Tensor
        Mask tensor of shape (T, H, W, wt, wh, ww, wt, wh, ww) with `-inf`
        at disallowed attention locations.
    """
    shift = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
    mask = torch.zeros(
        *(
            size[0],
            size[1],
            size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
        ),
        dtype=torch.float32,
    )

    ######################################
    # Dealing with shift along t(t) axis #
    ######################################
    mask[
        -1,  # last grid temporal row
        :,  # all grid rows
        :,  # all grid columns
        : -shift[0],  # within grid all temporal before shift
        :,  # all rows
        :,  # all columns
        -shift[0] :,  # within grid all temporal alien because of shift
        :,  # all rows
        :,  # all columns
    ] = -torch.inf

    mask[
        -1,  # last grid temporal row
        :,  # all grid rows
        :,  # all grid columns
        -shift[0] :,  # within grid all temporal alien because of shift
        :,  # all rows
        :,  # all columns
        : -shift[0],  # within grid all temporal before shift
        :,  # all rows
        :,  # all columns
    ] = -torch.inf

    ######################################
    # Dealing with shift along h(y) axis #
    ######################################
    mask[
        :,  # all grid temporal
        -1,  # last grid row
        :,  # all grid columns
        :,  # within grid all temporal
        : -shift[1],  # within grid all rows before shift
        :,  # within grid all columns
        :,  # within grid all temporal
        -shift[1] :,  # within grid rows alien because of shift
        :,  # within grid all columns
    ] = -torch.inf

    mask[
        :,  # all grid temporal
        -1,  # last grid row
        :,  # all grid columns
        :,  # within grid all temporal
        -shift[1] :,  # within grid rows alien because of shift
        :,  # within grid all columns
        :,  # within grid all temporal alien because of shift
        : -shift[1],  # within grid all rows before shift
        :,  # within grid all columns
    ] = -torch.inf

    ######################################
    # Dealing with shift along w(x) axis #
    ######################################
    mask[
        :,  # all grid temporal
        :,  # all grid rows
        :,  # last grid column
        :,  # within grid all temporal
        :,  # within grid all rows
        : -shift[2],  # within grid all columns before shift
        :,  # within grid all temporal
        :,  # within grid all rows
        -shift[2] :,  # within grid column alien because of shift
    ] = -torch.inf

    mask[
        :,  # all grid temporal
        :,  # all grid rows
        :,  # last grid column
        :,  # within grid all temporal
        :,  # within grid all rows
        -shift[2] :,  # within grid column alien because of shift
        :,  # within grid all temporal
        :,  # within grid all rows
        : -shift[2],  # within grid all columns before shift
    ] = -torch.inf
    return mask


def generate_shift_mask(
    input_shape: ArrayLike[int], window_size: ArrayLike[int] | int
) -> torch.Tensor:
    """
    Generate a shift attention mask for 1D, 2D, or 3D inputs.

    Automatically infer spatial dimensionality from input shape and apply
    appropriate shifted window masking for attention mechanisms.

    Parameters
    ----------
    input_shape : array-like of int
        Input tensor shape as (B, ...) where ... is 1D, 2D, or 3D spatial dims.
    window_size : int or array-like of int
        Size of the attention window along each spatial axis.

    Returns
    -------
    torch.Tensor
        Attention mask with `-inf` values blocking invalid positions due
        to cyclic shifts. Shape depends on dimensionality.

    Raises
    ------
    ValueError
        If number of spatial dimensions is not supported.
    """
    ndims = len(input_shape) - 1
    window_size_ = transform_to_int_tuple(window_size, "window_size", ndims)
    if ndims == 1:
        return generate_shift_mask_1d(size=input_shape[1], window_size=window_size_[0])
    elif ndims == 2:
        return generate_shift_mask_2d(size=input_shape[1:], window_size=window_size_)
    elif ndims == 3:
        return generate_shift_mask_3d(size=input_shape[1:], window_size=window_size_)
    raise ValueError(f"Unsupported number of dimensions for shift mask: {ndims}")
