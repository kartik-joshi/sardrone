"""Shared test helpers: read an .mcap back with the independent `mcap` library.

The writer uses the Foxglove SDK (Rust underneath); the tests read with the
pure-Python `mcap` + `mcap-protobuf-support` packages. If both agree, the
file is valid for third-party viewers too.
"""

import json
from collections import defaultdict

import pytest
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from sardrone.mcap.timeutil import NS_PER_SEC

# 2026-11-14T00:00:00Z - a fixed start so tests are reproducible.
START_NS = 1_794_614_400 * NS_PER_SEC


@pytest.fixture
def start_ns() -> int:
    return START_NS


class Episode:
    """Everything in an .mcap, loaded into memory for assertions."""

    def __init__(self, path):
        self.channels = {}  # topic -> (message_encoding, schema_name, schema_encoding)
        self.messages = defaultdict(list)  # topic -> [(log_time, decoded_message)]
        self.metadata = {}  # name -> dict
        with open(path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            summary = reader.get_summary()
            self.statistics = summary.statistics
            for channel in summary.channels.values():
                schema = summary.schemas[channel.schema_id]
                self.channels[channel.topic] = (
                    channel.message_encoding,
                    schema.name,
                    schema.encoding,
                )
            for schema, channel, message in reader.iter_messages():
                if channel.message_encoding == "json":
                    decoded = json.loads(message.data)
                else:
                    decoded = DecoderFactory().decoder_for("protobuf", schema)(message.data)
                self.messages[channel.topic].append((message.log_time, decoded))
            for record in reader.iter_metadata():
                self.metadata[record.name] = dict(record.metadata)

    def decoded(self, topic):
        return [decoded for _, decoded in self.messages[topic]]

    def log_times(self, topic):
        return [log_time for log_time, _ in self.messages[topic]]


@pytest.fixture
def read_episode():
    return Episode
