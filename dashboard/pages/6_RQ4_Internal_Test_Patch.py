#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from components import render_alignment_page
from data_loader import load_rq4_table


render_alignment_page(
    load_rq4_table(),
    title="RQ4 Internal Test-Patch",
    page_key="rq4_internal_test_patch",
    allow_representations=True,
    allow_patch_views=True,
)
