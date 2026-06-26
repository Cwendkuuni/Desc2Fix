#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from components import render_alignment_page
from data_loader import load_rq3_table


render_alignment_page(
    load_rq3_table(),
    title="RQ3 Report-Patch",
    page_key="rq3_report_patch",
    allow_representations=True,
    allow_patch_views=True,
)
