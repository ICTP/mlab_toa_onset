from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np

from caenParse import Channel, ErrorCode, caenParser, extract, info

matplotlib.use("Agg")


DATA = """=== DATA SAMPLES [3] in Volts == NB OF CHANNELS ACQUIRED: 2 ===
=== EVENT 7 ===
=== UnixTime = 1.25 date = 2026.2.9 time = 1h.2m.3s.4ms == TDC From FPGA = 99 == TDC Corrected = 1h2m3s,4ns ===
=== CH: 0 EVENTID: 4 FCR: 5 Baseline: 0.1 V Amplitude: 0.2 V Charge: 0.3 pC LeadingEdgeTime: 1.2 ns TrailingEdgeTime: 2.3 ns TrigCount: 6 TimeCount 7 ===
0.100000 -0.200000 0.300000
=== CH: 2 EVENTID: 8 FCR: 9 Baseline: 0.0 V Amplitude: 0.4 V Charge: 1.0 pC LeadingEdgeTime: 3.0 ns TrailingEdgeTime: 4.0 ns TrigCount: 10 TimeCount: 11 ===
0.400000 0.500000 0.600000
=== EVENT 9 ===
=== UnixTime = 2 date = 2026.2.9 time = noon == TDC From FPGA = 100 == TDC Corrected = noon ===
=== CH: 0 EVENTID: 1 FCR: 2 Baseline: 0 V Amplitude: 0 V Charge: 0 pC LeadingEdgeTime: 0 ns TrailingEdgeTime: 0 ns TrigCount: 0 TimeCount 0 ===
1 2 3
"""


def testInfoAndExtract(tmp_path: Path):
    path = tmp_path / "example.dat"
    path.write_text(DATA, encoding="ascii")
    summary = info(path)
    assert (summary.totalEvents, summary.firstEvent, summary.lastEvent) == (2, 7, 9)
    assert summary.pulseTraces == 3
    assert summary.sampleDtype == np.dtype(np.float32)

    event, code = extract(path, 7)
    assert code == ErrorCode.success
    assert event is not None
    assert event.metadata.fromFpga == 99
    assert event.channels[1].metadata.timeCount == 11
    assert event.getChannel(0).dtype == np.float32
    np.testing.assert_allclose(event.getChannel(2), [0.4, 0.5, 0.6])
    assert len(event.getEvent()) == 2


def testParserClass(tmp_path: Path):
    path = tmp_path / "example.dat"
    path.write_text(DATA, encoding="ascii")
    parser = caenParser(path)
    assert parser.filePath == path
    assert parser.info().totalEvents == 2
    event, code = parser.extract(7)
    assert code == ErrorCode.success
    assert event is not None and event.number == 7
    assert parser.extractOrRaise(9).number == 9


def testErrorsAndPlot(tmp_path: Path):
    path = tmp_path / "example.dat"
    path.write_text(DATA, encoding="ascii")
    event, _ = extract(path, 7)
    assert event is not None
    with patch("matplotlib.pyplot.show") as show:
        assert event.plot(0).get_xlabel() == "Sample"
        show.assert_called_once_with()
    with patch("matplotlib.pyplot.show") as show:
        axes = event.eventPlot(1e9, channelRange=[0, 2])
        show.assert_called_once_with()
    assert axes.shape == (1, 2)
    missing, code = extract(path, 8)
    assert missing is None and code == ErrorCode.eventNotFound
    missing, code = extract(tmp_path / "missing.dat", 7)
    assert missing is None and code == ErrorCode.fileNotFound



def testPlotOverlapped(tmp_path: Path):
    path = tmp_path / "example.dat"
    path.write_text(DATA, encoding="ascii")
    event, _ = extract(path, 7)
    assert event is not None
    template = event.channels[0]
    event.channels = [
        Channel(number, template.metadata, template.samples) for number in range(8)
    ]
    with patch("matplotlib.pyplot.show") as show:
        axes = event.plotOverlapped()
        show.assert_called_once_with()
    assert axes.shape == (2, 2)
    assert all(len(axis.lines) == 2 for axis in axes.flat)
    assert all(axis.lines[1].get_color() == "red" for axis in axes.flat)
