import logging

from . import common as common
from . import demo as demo
from . import null as null
from . import sa_score as sa_score
from . import tdc as tdc
from ._registry import OracleProtocol as OracleProtocol
from ._registry import get_oracle as get_oracle
from .cached import CachedOracle as CachedOracle
from .custom import CustomOracle as CustomOracle

try:
    from . import autodock as autodock
except ImportError as e:
    logging.warning(f"{e}. AutoDock oracles will not be available.")

try:
    from . import guacamol as guacamol
except ImportError as e:
    logging.warning(f"{e}. Guacamol oracles will not be available.")

try:
    from . import seh_proxy as seh_proxy
except ImportError as e:
    logging.warning(f"{e}. sEH proxy oracle will not be available.")
