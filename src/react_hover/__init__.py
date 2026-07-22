"""Official DSPy multi-hop ReAct agent eval + optimization (HoVer).

Based on: https://dspy.ai/tutorials/agents/
"""

from react_hover.agent import build_react_agent, safe_predict
from react_hover.metric import top5_recall

__all__ = ["build_react_agent", "safe_predict", "top5_recall"]
