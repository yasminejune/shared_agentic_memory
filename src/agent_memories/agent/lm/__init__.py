"""Gemma 2 2B IT plumbing and prompt templates shared by both DP mechanisms.

Nothing in here is privacy specific: it is the logit access and the prompt
wrapping that agent/amin_et_al/ and agent/invisible_ink/ both build on, and
that generalisation/ reuses for its own prompts.

Submodules are imported directly (``from ...agent.lm import token_generation
as tg``) so that pulling in the prompts does not load torch.
"""
