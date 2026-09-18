from matplotlib import pyplot as plot
from caenParse import *

cp = caenParser("./wavecatcher_02092026_run9_Ascii.dat_0001")

cp.info()

event = cp.extract(999)

event[0].float2FixEvent(m=2, n=14, overflow="clamp")

axes = event[0].plotOverlapped(xLabelName="Sample", yLabelName="Amplitude", xLim=(0, 1023), show=False)

for ax in axes.flat:
    ax.grid(True)

plot.show()