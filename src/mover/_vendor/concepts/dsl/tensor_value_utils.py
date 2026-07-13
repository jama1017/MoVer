#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : tensor_value_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 06/18/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import torch
from typing import Union, Sequence, List

from mover._vendor.concepts.dsl.tensor_value import TensorValue
from mover._vendor.concepts.dsl.tensor_state import StateObjectReference, StateObjectList
from mover._vendor.concepts.dsl.value import ListValue

__all__= ['expand_argument_values']


def expand_argument_values(
    argument_values: Sequence[Union[TensorValue, int, str, slice, StateObjectReference]],
    handle_wildcard: bool = False
) -> List[TensorValue]:
    """Expand a list of argument values to the same batch size.

    Args:
        argument_values: a list of argument values.
        handle_wildcard: whether to handle the wildcard variable '??'. If set to True, the function will return the
            original argument values without expanding them if any of the argument values contains the wildcard variable.

    Returns:
        the result list of argument values. All return values will have the same batch size.
    """

    if handle_wildcard:
        has_slot_var = False
        for arg in argument_values:
            if isinstance(arg, TensorValue):
                for var in arg.batch_variables:
                    if var == '??':
                        has_slot_var = True
                        break
        if has_slot_var:
            return list(argument_values)

    if len(argument_values) < 2:
        return list(argument_values)

    argument_values = list(argument_values)
    batch_variables = list()
    batch_sizes = list()
    for arg in argument_values:
        if isinstance(arg, TensorValue):
            for var in arg.batch_variables:
                if var not in batch_variables:
                    batch_variables.append(var)
                    batch_sizes.append(arg.get_variable_size(var))
        else:
            assert isinstance(arg, (int, str, slice, StateObjectReference, StateObjectList, ListValue)), arg

    masks = list()
    for i, arg in enumerate(argument_values):
        if isinstance(arg, TensorValue):
            argument_values[i] = arg.expand(batch_variables, batch_sizes)
            if argument_values[i].tensor_mask is not None:
                masks.append(argument_values[i].tensor_mask)

    if len(masks) > 0:
        final_mask = torch.stack(masks, dim=-1).amin(dim=-1)
        for arg in argument_values:
            if isinstance(arg, TensorValue):
                arg.tensor_mask = final_mask
                arg._mask_certified_flag = True  # now we have corrected the mask.
    return argument_values
