#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from components import render_alignment_page
from data_loader import load_rq2_table


render_alignment_page(
    load_rq2_table(),
    title="RQ2 Report-Test",
    page_key="rq2_report_test",
    allow_representations=True,
    allow_patch_views=False,
)
