"""ClueGround-VFM protocol, guards, metrics, and staged pipeline snapshot.

The bundled entrypoint pins the paper method to task-isolated MS-CXR. Generic
helpers retain the separately defined anatomy/device contracts for provenance,
but the strict runner cannot select or transfer those supervision sources.
"""

from .contracts import PROTOCOLS, ProtocolSpec, get_protocol

__all__ = ["PROTOCOLS", "ProtocolSpec", "get_protocol"]
