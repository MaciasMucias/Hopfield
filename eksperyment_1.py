"""
**Skuteczność reguł uczenia**:
  - Skuteczność każdej z reguł uczenia i jej wpływ na liczbę stabilnych wzorców uczących.
"""

from hopfield.base import Hopfield
from hopfield.rules import OjiRule, HebbianRule
from hopfield.helpers import pattern_completion_error
