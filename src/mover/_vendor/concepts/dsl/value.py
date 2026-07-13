#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : value.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/18/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""The baseclass for all value representations of domain-specific languages."""

from typing import Any, Union, Sequence

from mover._vendor.concepts.dsl.dsl_types import TypeBase, ObjectType, ValueType, ListType
from mover._vendor.concepts.dsl.dsl_types import get_format_context

__all__ = ['ValueBase', 'Value', 'ListValue']


class ValueBase(object):
    """The baseclass for all value representations of domain-specific languages."""

    dtype: TypeBase
    """The type of the value."""

    def __init__(self, dtype: TypeBase):
        """Initialize the Value object.

        Args:
            dtype: the type of the value.
        """
        self.dtype = dtype


class Value(ValueBase):
    """A simple value object holds a pair of dtype and a value."""

    dtype: Union[ObjectType, ValueType]
    """The type of the value."""

    value: Any
    """The value."""

    def __init__(self, dtype: Union[ObjectType, ValueType], value: Any):
        """Initialize the Value object.

        Args:
            dtype: the type of the value.
            value: the value.
        """
        super().__init__(dtype)
        self.value = value
        self._check_type()

    def _check_type(self):
        """Check if the value is of the correct type. Default to no-op."""
        pass

    def __str__(self):
        if get_format_context().object_format_type:
            return f'V({self.value}, dtype={self.dtype})'
        else:
            return self.value

    def __repr__(self):
        return f'V({self.value}, dtype={self.dtype})'


class ListValue(ValueBase):
    """A list of values."""

    dtype: ListType
    """The type of the value."""

    @property
    def element_type(self):
        return self.dtype.element_type

    values: tuple
    """The values."""

    def __init__(self, dtype: ListType, values: Sequence[Any]):
        """Initialize the Value object.

        Args:
            dtype: the type of the value.
            values: the values.
        """
        super().__init__(dtype)
        self.values = tuple(values)

    def __len__(self):
        return len(self.values)

    def __str__(self):
        elements_str = ', '.join(str(v) for v in self.values)
        if get_format_context().object_format_type:
            return f'LV({{{elements_str}}}, dtype={self.dtype})'
        else:
            return f'{{{elements_str}}}'

    def __repr__(self):
        elements_str = ', '.join(str(v) for v in self.values)
        return f'LV({{{elements_str}}}, dtype={self.dtype})'

