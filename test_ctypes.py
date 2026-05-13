import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a

point = _k4a.k4a_float2_t()
point.xy.x = 100.0
point.xy.y = 100.0
print("Created successfully:", point.xy.x, point.xy.y)
