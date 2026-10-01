"""Streaming parser and data objects for CAEN WaveCatcher ASCII files."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray


EVENT_RE = re.compile(r"^===\s+EVENT\s+(\d+)\s+===$")
CHANNEL_RE = re.compile(r"^===\s+CH:\s*(\d+)\s+(.*?)\s+===$")
EVENT_META_RE = re.compile(
    r"UnixTime\s*=\s*(\S+)\s+date\s*=\s*(\S+)\s+time\s*=\s*(.*?)\s*"
    r"==\s*TDC\s+From\s+FPGA\s*=\s*(\S+)\s*==\s*TDC\s+Corrected\s*=\s*(.*?)\s*==="
)
CHANNEL_META_RE = re.compile(
    r"EVENTID:\s*(\d+)\s+FCR:\s*(\d+)\s+Baseline:\s*(\S+)\s+V\s+"
    r"Amplitude:\s*(\S+)\s+V\s+Charge:\s*(\S+)\s+pC\s+"
    r"LeadingEdgeTime:\s*(\S+)\s+ns\s+TrailingEdgeTime:\s*(\S+)\s+ns\s+"
    r"TrigCount:\s*(\d+)\s+TimeCount\s*:?\s*(\d+)"
)


class CaenParseError(ValueError):
    """Raised when a data file does not match the expected ASCII format."""


class ErrorCode(IntEnum):
    """Status returned by :func:`extract`."""

    success = 0
    fileNotFound = 1
    eventNotFound = 2
    parseError = 3
    invalidArgument = 4


@dataclass(frozen=True)
class FileInfo:
    totalEvents: int
    firstEvent: Optional[int]
    lastEvent: Optional[int]
    fileSize: int
    pulseTraces: int
    sampleDtype: np.dtype


@dataclass(frozen=True)
class EventMetadata:
    unixTime: float
    date: str
    time: str
    fromFpga: int
    tdcCorrected: str


@dataclass(frozen=True)
class ChannelMetadata:
    eventId: int
    fcr: int
    baseline: float
    amplitude: float
    charge: float
    leadingEdgeTime: float
    trailingEdgeTime: float
    trigCount: int
    timeCount: int


@dataclass(frozen=True)
class Channel:
    number: int
    metadata: ChannelMetadata
    samples: NDArray[Any]


@dataclass
class Event:
    number: int
    metadata: EventMetadata
    channels: List[Channel]

    def getChannel(self, channelNumber: int) -> NDArray[Any]:
        """Return the samples for a hardware channel number."""
        for channel in self.channels:
            if channel.number == channelNumber:
                return channel.samples
        raise KeyError(f"Channel {channelNumber} is not present in event {self.number}")

    def getEvent(self) -> List[NDArray[Any]]:
        """Return channel arrays ordered by channel number."""
        return [channel.samples for channel in sorted(self.channels, key=lambda item: item.number)]

    def float2FixChannel(
        self,
        channelNumber: int,
        *,
        signed: bool = True,
        m: int = 1,
        n: int = 15,
        overflow: str = "clamp",
        rounding: str = "nearest",
    ) -> NDArray[Any]:
        """Convert one trace to an object array of ``fixedpoint.FixedPoint`` values."""
        try:
            from fixedpoint import FixedPoint
        except ImportError as error:
            raise ImportError("float2FixChannel requires: pip install fixedpoint") from error

        values = self.getChannel(channelNumber)
        return np.asarray(
            [
                FixedPoint(
                    float(value), signed=signed, m=m, n=n,
                    overflow=overflow, rounding=rounding,
                )
                for value in values
            ],
            dtype=object,
        )

    def float2FixEvent(self, **fixedPointOptions: Any) -> List[NDArray[Any]]:
        """Convert every trace using :meth:`float2FixChannel`."""
        return [
            self.float2FixChannel(channel.number, **fixedPointOptions)
            for channel in sorted(self.channels, key=lambda item: item.number)
        ]

    def plot(
        self,
        channelNumber: int,
        samplingFrequency: Optional[float] = None,
        *,
        xLabelName: Optional[str] = None,
        yLabelName: str = "Amplitude (V)",
        xLim: Optional[Tuple[float, float]] = None,
        yLim: Optional[Tuple[float, float]] = None,
        color: Optional[str] = None,
        lineWidth: float = 1.5,
        lineType: str = "-",
        ax: Any = None,
        show: bool = True,
        **plotOptions: Any,
    ) -> Any:
        """Plot one trace, display it by default, and return its axes."""
        import matplotlib.pyplot as plt

        samples = self.getChannel(channelNumber)
        if samplingFrequency is not None and samplingFrequency <= 0:
            raise ValueError("samplingFrequency must be greater than zero")
        xValues = (
            np.arange(samples.size) / samplingFrequency
            if samplingFrequency is not None else np.arange(samples.size)
        )
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(
            xValues, samples, color=color, linewidth=lineWidth,
            linestyle=lineType, **plotOptions,
        )
        ax.set_xlabel(xLabelName or ("Time (s)" if samplingFrequency is not None else "Sample"))
        ax.set_ylabel(yLabelName)
        ax.set_title(f"Event {self.number}, channel {channelNumber}")
        if xLim is not None:
            ax.set_xlim(xLim)
        if yLim is not None:
            ax.set_ylim(yLim)
        if show:
            plt.show()
        return ax

    def eventPlot(
        self,
        samplingFrequency: Optional[float] = None,
        *,
        channelRange: Optional[Union[int, Iterable[int]]] = None,
        columns: int = 2,
        show: bool = True,
        **plotOptions: Any,
    ) -> Any:
        """Plot and display selected channels, returning the resulting axes."""
        import matplotlib.pyplot as plt

        if isinstance(channelRange, int):
            return self.plot(
                channelRange, samplingFrequency, show=show, **plotOptions
            )
        available = [item.number for item in sorted(self.channels, key=lambda item: item.number)]
        selected = available if channelRange is None else list(channelRange)
        missing = sorted(set(selected) - set(available))
        if missing:
            raise KeyError(f"Channels not present in event {self.number}: {missing}")
        if not selected:
            raise ValueError("channelRange cannot be empty")
        if columns < 1:
            raise ValueError("columns must be at least 1")
        rows = (len(selected) + columns - 1) // columns
        _, axes = plt.subplots(rows, columns, squeeze=False)
        flatAxes = axes.ravel()
        for channelNumber, axis in zip(selected, flatAxes):
            self.plot(
                channelNumber, samplingFrequency, ax=axis, show=False,
                **plotOptions,
            )
        for axis in flatAxes[len(selected):]:
            axis.set_visible(False)
        axes[0, 0].figure.tight_layout()
        if show:
            plt.show()
        return axes

    def plotOverlapped(
        self,
        samplingFrequency: Optional[float] = None,
        *,
        columns: int = 2,
        show: bool = True,
        **plotOptions: Any,
    ) -> Any:
        """Plot channel pairs 0/1, 2/3, 4/5, and 6/7 overlapped."""
        import matplotlib.pyplot as plt

        requiredChannels = set(range(8))
        availableChannels = {channel.number for channel in self.channels}
        missingChannels = sorted(requiredChannels - availableChannels)
        if missingChannels:
            raise KeyError(
                f"Channels required by plotOverlapped are missing from event "
                f"{self.number}: {missingChannels}"
            )
        if columns < 1:
            raise ValueError("columns must be at least 1")

        channelPairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        rows = (len(channelPairs) + columns - 1) // columns
        _, axes = plt.subplots(rows, columns, squeeze=False)
        flatAxes = axes.ravel()
        evenColor = plotOptions.pop("color", None)

        for (evenChannel, oddChannel), axis in zip(channelPairs, flatAxes):
            self.plot(
                evenChannel, samplingFrequency, ax=axis, show=False,
                color=evenColor, label=f"Channel {evenChannel}",
                **plotOptions,
            )
            self.plot(
                oddChannel, samplingFrequency, ax=axis, show=False,
                color="red", label=f"Channel {oddChannel}",
                **plotOptions,
            )
            axis.set_title(
                f"Event {self.number}, channels {evenChannel} and {oddChannel}"
            )
            axis.legend()

        for axis in flatAxes[len(channelPairs):]:
            axis.set_visible(False)
        axes[0, 0].figure.tight_layout()
        if show:
            plt.show()
        return axes


class caenParser:
    """Stateful entry point for inspecting and extracting events from one file.

    The path is stored, but the file is opened only when ``info``, ``extract``,
    or ``extractOrRaise`` is called.
    """

    def __init__(self, filePath: Union[str, os.PathLike[str]]) -> None:
        self.filePath = Path(filePath)

    def info(self) -> FileInfo:
        """Print and return information about the configured data file."""
        return info(self.filePath)

    def extract(self, eventNumber: int) -> Tuple[Optional[Event], ErrorCode]:
        """Extract an event and return ``(event, errorCode)``."""
        return extract(self.filePath, eventNumber)

    def extractOrRaise(self, eventNumber: int) -> Event:
        """Extract an event, raising an exception if extraction fails."""
        return extractOrRaise(self.filePath, eventNumber)


def _detectDtype(tokens: Sequence[str]) -> np.dtype:
    if not tokens:
        return np.dtype(np.float32)
    isFloat = any("." in token or "e" in token.lower() for token in tokens)
    if isFloat:
        # Seven significant decimal digits fit float32; retain more precise sources as float64.
        significant = max(
            len(token.lower().split("e", 1)[0].replace(".", "").lstrip("+-0"))
            for token in tokens
        )
        return np.dtype(np.float32 if significant <= 7 else np.float64)
    values = [int(token) for token in tokens]
    minimum, maximum = min(values), max(values)
    candidates = (
        (np.uint8, np.uint16, np.uint32, np.uint64)
        if minimum >= 0 else (np.int8, np.int16, np.int32, np.int64)
    )
    for candidate in candidates:
        limits = np.iinfo(candidate)
        if limits.min <= minimum and maximum <= limits.max:
            return np.dtype(candidate)
    raise CaenParseError("Sample integer lies outside the NumPy 64-bit range")


def _parseEventMetadata(line: str) -> EventMetadata:
    match = EVENT_META_RE.search(line)
    if not match:
        raise CaenParseError(f"Malformed event metadata: {line.strip()}")
    return EventMetadata(float(match[1]), match[2], match[3].strip(), int(match[4]), match[5].strip())


def _parseChannelMetadata(text: str) -> ChannelMetadata:
    match = CHANNEL_META_RE.search(text)
    if not match:
        raise CaenParseError(f"Malformed channel metadata: {text.strip()}")
    return ChannelMetadata(
        int(match[1]), int(match[2]), float(match[3]), float(match[4]),
        float(match[5]), float(match[6]), float(match[7]), int(match[8]), int(match[9]),
    )


def _eventBlocks(filePath: Path):
    currentNumber = None
    currentLines: List[str] = []
    with filePath.open("r", encoding="ascii", newline=None) as stream:
        for line in stream:
            eventMatch = EVENT_RE.match(line.strip())
            if eventMatch:
                if currentNumber is not None:
                    yield currentNumber, currentLines
                currentNumber = int(eventMatch[1])
                currentLines = []
            elif currentNumber is not None:
                currentLines.append(line.strip())
    if currentNumber is not None:
        yield currentNumber, currentLines


def _parseEvent(number: int, lines: Sequence[str]) -> Event:
    nonempty = [line for line in lines if line]
    if not nonempty:
        raise CaenParseError(f"Event {number} has no metadata")
    metadata = _parseEventMetadata(nonempty[0])
    rawChannels = []
    index = 1
    while index < len(nonempty):
        channelMatch = CHANNEL_RE.match(nonempty[index])
        if not channelMatch:
            raise CaenParseError(f"Expected channel header in event {number}: {nonempty[index]}")
        channelNumber = int(channelMatch[1])
        channelMetadata = _parseChannelMetadata(channelMatch[2])
        index += 1
        sampleTokens: List[str] = []
        while index < len(nonempty) and not CHANNEL_RE.match(nonempty[index]):
            sampleTokens.extend(nonempty[index].split())
            index += 1
        if not sampleTokens:
            raise CaenParseError(f"Channel {channelNumber} in event {number} has no samples")
        rawChannels.append((channelNumber, channelMetadata, sampleTokens))
    dtype = _detectDtype([token for _, _, tokens in rawChannels for token in tokens])
    channels = [
        Channel(channelNumber, channelMetadata, np.asarray(sampleTokens, dtype=dtype))
        for channelNumber, channelMetadata, sampleTokens in rawChannels
    ]
    return Event(number, metadata, channels)


def info(filePath: Union[str, os.PathLike[str]]) -> FileInfo:
    """Scan a file, print its summary, and return the same information."""
    path = Path(filePath)
    eventNumbers: List[int] = []
    pulseTraces = 0
    sampleTokens: List[str] = []
    expectingSamples = False
    with path.open("r", encoding="ascii", newline=None) as stream:
        for line in stream:
            stripped = line.strip()
            eventMatch = EVENT_RE.match(stripped)
            if eventMatch:
                eventNumbers.append(int(eventMatch[1]))
                expectingSamples = False
            elif CHANNEL_RE.match(stripped):
                pulseTraces += 1
                expectingSamples = True
            elif expectingSamples and stripped and not stripped.startswith("==="):
                sampleTokens.extend(stripped.split())
                expectingSamples = False
    dtype = _detectDtype(sampleTokens)
    result = FileInfo(
        len(eventNumbers), min(eventNumbers) if eventNumbers else None,
        max(eventNumbers) if eventNumbers else None, path.stat().st_size,
        pulseTraces, dtype,
    )
    eventRange = (
        f"{result.firstEvent} to {result.lastEvent}" if eventNumbers else "none"
    )
    print(f"Events: {result.totalEvents} (range: {eventRange})")
    print(f"File size: {result.fileSize} bytes")
    print(f"Pulse traces: {result.pulseTraces}")
    print(f"Sample representation: {result.sampleDtype.name}")
    return result


def extract(
    filePath: Union[str, os.PathLike[str]], eventNumber: int
) -> Tuple[Optional[Event], ErrorCode]:
    """Extract an event, print feedback, and return ``(event, errorCode)``."""
    if not isinstance(eventNumber, (int, np.integer)) or isinstance(eventNumber, bool):
        print(f"Extraction failed: event number must be an integer (code {ErrorCode.invalidArgument.value})")
        return None, ErrorCode.invalidArgument
    path = Path(filePath)
    if not path.is_file():
        print(f"Extraction failed: file not found: {path} (code {ErrorCode.fileNotFound.value})")
        return None, ErrorCode.fileNotFound
    try:
        for number, lines in _eventBlocks(path):
            if number == int(eventNumber):
                event = _parseEvent(number, lines)
                print(f"Extracted event {number}: {len(event.channels)} channels (code 0)")
                return event, ErrorCode.success
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Extraction failed: {error} (code {ErrorCode.parseError.value})")
        return None, ErrorCode.parseError
    print(f"Extraction failed: event {eventNumber} not found (code {ErrorCode.eventNotFound.value})")
    return None, ErrorCode.eventNotFound


def extractOrRaise(filePath: Union[str, os.PathLike[str]], eventNumber: int) -> Event:
    """Extract an event and raise an exception on failure (convenience API)."""
    event, code = extract(filePath, eventNumber)
    if event is None:
        if code == ErrorCode.fileNotFound:
            raise FileNotFoundError(filePath)
        if code == ErrorCode.eventNotFound:
            raise KeyError(f"Event {eventNumber} was not found")
        raise CaenParseError(f"Could not extract event {eventNumber}; error code {code.value}")
    return event
