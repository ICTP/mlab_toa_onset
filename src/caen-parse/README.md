# caen-parse

`caen-parse` reads ASCII `.dat` files produced by a CAEN WaveCatcher/x743
digitizer. It scans files without loading the complete acquisition into memory and
represents each event and channel with typed Python objects and NumPy arrays.

## Installation

From this directory:

```bash
python -m pip install -e .
```

This installs NumPy, Matplotlib, and `fixedpoint`.

## Parser object

Create one `caenParser` for each acquisition file. The parser stores the path but
does not keep the file open.

```python
from caenParse import caenParser

parser = caenParser("run.dat")
summary = parser.info()
print(summary.totalEvents)
print(summary.firstEvent, summary.lastEvent)
print(summary.fileSize, summary.pulseTraces, summary.sampleDtype)
```

`info` also prints the event count/range, byte size, pulse-trace count, and the
detected NumPy sample representation. Decimal/scientific data with up to seven
significant digits is represented as `float32`; higher precision becomes
`float64`. Integer data uses the smallest signed/unsigned NumPy integer type that
can contain it.

## Extract and inspect an event

```python
from caenParse import ErrorCode, caenParser

parser = caenParser("run.dat")
event, errorCode = parser.extract(501)
if errorCode == ErrorCode.success:
    print(event.number, event.metadata)
    print(event.channels[0].metadata)

    channel0 = event.getChannel(0)  # NumPy array
    allChannels = event.getEvent()  # list ordered by hardware channel number
```

Error codes are `success` (0), `fileNotFound` (1), `eventNotFound` (2),
`parseError` (3), and `invalidArgument` (4). `extract` prints a concise status.
For exception-oriented code, use `parser.extractOrRaise(eventNumber)`.

The standalone `info(path)`, `extract(path, eventNumber)`, and
`extractOrRaise(path, eventNumber)` functions remain available for backward
compatibility, but `caenParser` is the recommended interface.

Metadata units are implicit in the attributes: baseline/amplitude are volts,
charge is pC, and leading/trailing-edge times are ns. Original date/time and TDC
corrected strings are preserved because their format is device-specific.

## Fixed-point conversion

Conversion produces a NumPy object array whose elements are `FixedPoint`
instances. The default is signed Q1.15 with clamping and nearest rounding:

```python
fixedChannel = event.float2FixChannel(0)
fixedEvent = event.float2FixEvent(m=2, n=14, overflow="clamp")
```

Choose `m` and `n` for the range and precision required by downstream hardware.

## Plotting

```python
event.plot(
    0,
    samplingFrequency=3.2e9,
    xLabelName="Time (s)",
    color="navy",
    lineWidth=1.0,
    lineType="--",
    xLim=(0, 100e-9),
)

event.eventPlot(
    samplingFrequency=3.2e9,
    channelRange=range(4),
    columns=2,
    color="crimson",
)

# Four subplots containing channel pairs 0/1, 2/3, 4/5, and 6/7.
# Odd-numbered channels are drawn in red.
event.plotOverlapped(samplingFrequency=3.2e9, lineWidth=1.0)
```

Omit `samplingFrequency` to use sample indices on the x-axis. Passing one integer
as `channelRange` selects that channel and delegates to `plot`; passing an
iterable creates subplots. Plotting methods display the completed figure by
default. Pass `show=False` when further customization or saving should happen
before display. Plotting methods return Matplotlib axes:

```python
axes = event.eventPlot(channelRange=[0, 1, 2, 3], show=False)
axes[0, 0].figure.savefig("event501.png", dpi=150)
axes[0, 0].figure.show()
```

## Useful object attributes

- `Event`: `number`, `metadata`, `channels`
- `EventMetadata`: `unixTime`, `date`, `time`, `fromFpga`, `tdcCorrected`
- `Channel`: `number`, `metadata`, `samples`
- `ChannelMetadata`: `eventId`, `fcr`, `baseline`, `amplitude`, `charge`,
  `leadingEdgeTime`, `trailingEdgeTime`, `trigCount`, `timeCount`

## Development

```bash
python -m pytest
```
