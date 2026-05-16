"""Backend dispatcher.

Reads the BACKEND env var (default: anthropic) and re-exports the Agent /
Annotator / Digest / ProfileBuilder / ProactiveClassifier classes from the
matching backend module. bot.py imports from here, so swapping is a single
env var change at startup — no per-module conditionals.

Valid values:
    BACKEND=anthropic   (default — uses agent.py, annotator.py, etc.)
    BACKEND=openai      (uses agent_openai.py, annotator_openai.py, etc.)

Mem0's internal LLM is also routed by this env (see memory.py).
"""

import os

BACKEND = os.environ.get("BACKEND", "anthropic").lower().strip()

if BACKEND == "openai":
    from agent_openai import Agent
    from annotator_openai import Annotator
    from digest_openai import Digest
    from proactive_openai import ProactiveClassifier
    from user_profile_openai import ProfileBuilder
elif BACKEND == "anthropic":
    from agent import Agent
    from annotator import Annotator
    from digest import Digest
    from proactive import ProactiveClassifier
    from user_profile import ProfileBuilder
else:
    raise SystemExit(
        f"Invalid BACKEND={BACKEND!r}; must be 'anthropic' or 'openai'"
    )

__all__ = [
    "BACKEND",
    "Agent",
    "Annotator",
    "Digest",
    "ProactiveClassifier",
    "ProfileBuilder",
]
