"""A language-model agent that turns a plain-English mission into drone actions.

    sim.py       a small, deterministic mission simulator the agent acts on
    tools.py     the six tools, as JSON schemas a model can call
    models.py    model back-ends: Cohere, a rule-based baseline, a scripted one
    agent.py     the loop: model -> tool call -> result -> model ...
    evaluate.py  20 written missions, scored
"""
