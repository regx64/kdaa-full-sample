"""KDAA pipeline orchestrator: sound-autodaq -> meligation -> classism, one command.

See PLAN.md for the design. Stages (kdaa.stages):
export -> mel -> pack -> check -> train -> evaluate -> export_int8 -> report.
"""

__version__ = "0.1.0"
