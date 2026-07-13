#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : parser_base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/25/2022
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""The baseclass for all parsers of domain-specific languages."""

from mover._vendor.concepts.dsl.dsl_domain import DSLDomainBase
from mover._vendor.concepts.dsl.expression import Expression

__all__ = ['ParserBase']


class ParserBase(object):
    """The baseclass for all parsers of domain-specific languages."""

    def parse_domain_file(self, path: str) -> DSLDomainBase:
        """Parse a domain from a file.

        Args:
            path: the path to the file.

        Returns:
            the parsed domain.
        """
        with open(path) as f:
            return self.parse_domain_string(f.read())

    def parse_domain_string(self, string: str) -> DSLDomainBase:
        """Parse a domain from a string.

        Args:
            string: the string to parse.

        Returns:
            the parsed domain.
        """
        raise NotImplementedError()

    def parse_expression(self, string: str, **kwargs) -> Expression:
        """Parse an expression from a string.

        Args:
            string: the string to parse.

        Returns:
            the parsed expression.
        """
        raise NotImplementedError()

