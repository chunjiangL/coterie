"""Backend dispatcher.

Reads the BACKEND env var (default: anthropic) and re-exports the Agent /
Annotator / Digest / ProfileBuilder / ProactiveClassifier classes from the
matching subpackage. Platform adapters import from here, so swapping is a
single env var change at startup — no per-module conditionals.

Valid values:
    BACKEND=anthropic   (default — uses coterie.claude.*)
    BACKEND=openai      (uses coterie.gpt.*)

Mem0's internal LLM is also routed by this env (see memory.py).
"""

import os

BACKEND = os.environ.get("BACKEND", "anthropic").lower().strip()

if BACKEND == "openai":
    from coterie.gpt.agent import Agent
    from coterie.gpt.annotator import Annotator
    from coterie.gpt.digest import Digest
    from coterie.gpt.proactive import ProactiveClassifier
    from coterie.gpt.user_profile import ProfileBuilder
elif BACKEND == "anthropic":
    from coterie.claude.agent import Agent
    from coterie.claude.annotator import Annotator
    from coterie.claude.digest import Digest
    from coterie.claude.proactive import ProactiveClassifier
    from coterie.claude.user_profile import ProfileBuilder
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
