import struct
import time

import pytest

from webauthn.ctaphid import (
	CID_BROADCAST,
	CTAPHID_CANCEL,
	CTAPHID_CBOR,
	CTAPHID_ERROR,
	CTAPHID_INIT,
	CTAPHID_KEEPALIVE,
	CTAPHID_MSG,
	CTAPHID_PING,
	ERR_INVALID_CMD,
	CtapHidDevice,
)


class FakeAuthenticator:
	def __init__(self, response=b"\x00", delay=0.0):
		self.response = response
		self.delay = delay
		self.cancelled = False
		self.seen = []

	def handle_cbor(self, payload):
		self.seen.append(payload)
		if self.delay:
			end = time.monotonic() + self.delay
			while time.monotonic() < end and not self.cancelled:
				time.sleep(0.01)
		return self.response

	def cancel(self):
		self.cancelled = True


def build_device(authenticator=None):
	authenticator = authenticator or FakeAuthenticator()
	sent = []
	device = CtapHidDevice(authenticator, sent.append)
	return device, sent, authenticator


def init_frame(cid, command, payload=b""):
	frame = struct.pack(">IBH", cid, 0x80 | command, len(payload)) + payload
	return frame.ljust(64, b"\x00")


def parse_frame(frame):
	cid = struct.unpack(">I", frame[:4])[0]
	command = frame[4] & 0x7F
	length = struct.unpack(">H", frame[5:7])[0]
	return cid, command, frame[7:7 + length]


def do_init(device, sent):
	device.feed_report(init_frame(CID_BROADCAST, CTAPHID_INIT, b"12345678"))
	_, command, payload = parse_frame(sent.pop())
	assert command == CTAPHID_INIT
	assert payload[:8] == b"12345678"
	return struct.unpack(">I", payload[8:12])[0]


def test_init_allocates_channel(device_sent=None):
	device, sent, _ = build_device()
	cid = do_init(device, sent)
	assert cid != CID_BROADCAST and cid != 0


def test_init_capability_flags():
	device, sent, _ = build_device()
	device.feed_report(init_frame(CID_BROADCAST, CTAPHID_INIT, b"abcdefgh"))
	_, _, payload = parse_frame(sent.pop())
	capabilities = payload[16]
	assert capabilities & 0x04  # CBOR
	assert capabilities & 0x08  # NMSG (no U2F)


def test_ping_roundtrip():
	device, sent, _ = build_device()
	cid = do_init(device, sent)
	device.feed_report(init_frame(cid, CTAPHID_PING, b"hello"))
	_, command, payload = parse_frame(sent.pop())
	assert command == CTAPHID_PING
	assert payload == b"hello"


def test_cbor_dispatch():
	auth = FakeAuthenticator(response=b"\x00\xa1\x01\x02")
	device, sent, _ = build_device(auth)
	cid = do_init(device, sent)
	device.feed_report(init_frame(cid, CTAPHID_CBOR, b"\x04"))
	# Wait for the worker thread to finish
	time.sleep(0.2)
	_, command, payload = parse_frame(sent.pop())
	assert command == CTAPHID_CBOR
	assert payload == b"\x00\xa1\x01\x02"
	assert auth.seen == [b"\x04"]


def test_u2f_msg_rejected():
	device, sent, _ = build_device()
	cid = do_init(device, sent)
	device.feed_report(init_frame(cid, CTAPHID_MSG, b"\x00"))
	_, command, payload = parse_frame(sent.pop())
	assert command == CTAPHID_ERROR
	assert payload == bytes([ERR_INVALID_CMD])


def test_unknown_channel_rejected():
	device, sent, _ = build_device()
	do_init(device, sent)
	device.feed_report(init_frame(0xDEADBEEF, CTAPHID_PING, b"x"))
	_, command, payload = parse_frame(sent.pop())
	assert command == CTAPHID_ERROR


def test_keepalive_emitted_during_slow_cbor():
	auth = FakeAuthenticator(response=b"\x00", delay=0.35)
	device, sent, _ = build_device(auth)
	cid = do_init(device, sent)
	device.feed_report(init_frame(cid, CTAPHID_CBOR, b"\x02"))
	time.sleep(0.5)
	commands = [parse_frame(f)[1] for f in sent]
	assert CTAPHID_KEEPALIVE in commands
	# The final frame is the actual CBOR response
	assert parse_frame(sent[-1])[1] == CTAPHID_CBOR


def test_cancel_aborts_verification():
	auth = FakeAuthenticator(response=b"\x00", delay=2.0)
	device, sent, _ = build_device(auth)
	cid = do_init(device, sent)
	device.feed_report(init_frame(cid, CTAPHID_CBOR, b"\x02"))
	time.sleep(0.1)
	device.feed_report(init_frame(cid, CTAPHID_CANCEL, b""))
	time.sleep(0.2)
	assert auth.cancelled is True
	# No CBOR response frame is sent after a cancel
	assert all(parse_frame(f)[1] != CTAPHID_CBOR for f in sent)


def test_fragmented_request_reassembled():
	auth = FakeAuthenticator(response=b"\x00")
	device, sent, _ = build_device(auth)
	cid = do_init(device, sent)
	# 100 byte payload spans an init frame plus one continuation frame
	payload = bytes(range(100))
	command = CTAPHID_CBOR
	init = struct.pack(">IBH", cid, 0x80 | command, len(payload)) + payload[:57]
	device.feed_report(init.ljust(64, b"\x00"))
	cont = struct.pack(">IB", cid, 0) + payload[57:]
	device.feed_report(cont.ljust(64, b"\x00"))
	time.sleep(0.2)
	assert auth.seen == [payload]


def test_long_response_fragmented():
	# 200 byte response must span multiple frames on the way out
	auth = FakeAuthenticator(response=b"\x00" + bytes(range(200)))
	device, sent, _ = build_device(auth)
	cid = do_init(device, sent)
	device.feed_report(init_frame(cid, CTAPHID_CBOR, b"\x04"))
	time.sleep(0.2)
	# First frame is the CBOR init frame, followed by continuation frames
	assert parse_frame(sent[0])[1] == CTAPHID_CBOR
	total = struct.unpack(">H", sent[0][5:7])[0]
	assert total == 201


def test_concurrent_sends_do_not_interleave():
	# Two threads each send a multi-frame message; with a slow sink to widen
	# the race window, every message's frames must still arrive contiguously
	import threading

	order = []

	def slow_sink(frame):
		cid = struct.unpack(">I", frame[:4])[0]
		order.append(cid)
		time.sleep(0.001)

	device = CtapHidDevice(FakeAuthenticator(), slow_sink)
	data = bytes(range(150))  # spans an init frame plus two continuation frames

	def send(cid):
		device._send_message(cid, CTAPHID_PING, data)

	threads = [threading.Thread(target=send, args=(cid,)) for cid in (0xAA, 0xBB)]
	for t in threads:
		t.start()
	for t in threads:
		t.join()

	# Each cid's frames form one unbroken run: at most one transition per cid
	transitions = sum(1 for a, b in zip(order, order[1:]) if a != b)
	assert transitions == 1, order


def test_reinit_cancels_active_operation():
	auth = FakeAuthenticator(response=b"\x00", delay=2.0)
	device, sent, _ = build_device(auth)
	cid = do_init(device, sent)
	device.feed_report(init_frame(cid, CTAPHID_CBOR, b"\x02"))
	time.sleep(0.1)
	# The browser re-INITs the same channel to restart (e.g. a double click)
	device.feed_report(init_frame(cid, CTAPHID_INIT, b"12345678"))
	time.sleep(0.1)
	assert auth.cancelled is True


def test_channel_busy_rejected():
	auth = FakeAuthenticator(response=b"\x00", delay=0.4)
	device, sent, _ = build_device(auth)
	cid_a = do_init(device, sent)
	cid_b = do_init(device, sent)
	device.feed_report(init_frame(cid_a, CTAPHID_CBOR, b"\x02"))
	time.sleep(0.05)
	device.feed_report(init_frame(cid_b, CTAPHID_CBOR, b"\x02"))
	# The second channel should get a busy error quickly
	busy = [parse_frame(f) for f in sent if parse_frame(f)[0] == cid_b]
	assert busy and busy[-1][1] == CTAPHID_ERROR
