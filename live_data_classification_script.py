# 🔌 OpenSignals TCP/IP Live Stream (Fixed for JSON packets)
# Requirements:
#  - OpenSignals running with TCP-IP enabled
#  - Acquisition started (green ▶ pressed in OpenSignals)
#  - Port set to 5555 in Integration settings

import socket, json, threading, select, queue, time, warnings
import matplotlib.pyplot as plt
from collections import deque
from queue import SimpleQueue
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, List

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - runtime dependency only
    torch = None
    nn = None

try:  # ensure pickle can restore the label encoder
    from sklearn.preprocessing import LabelEncoder  # noqa: F401
except Exception:  # pragma: no cover
    LabelEncoder = None  # type: ignore

warnings.filterwarnings("ignore", category=UserWarning)

MENU_INPUT = {
    0: 'devices', 1: 'config,{MAC|DEVICE_ID}', 2: 'config,{MAC|DEVICE_ID}, {PARAM}, {VALUE}',
    3: 'enable,{MAC|DEVICE_ID}', 4: 'disable,{MAC|DEVICE_ID}', 5: 'start',
    6: 'set_digital_output, {MAC|DEVICE_ID}, {CHANNEL}, {STATE}', 7: 'stop', 8: 'exit'
}

CLASSIFIER_MODEL_PATH = Path("models/cnn_a4_caden5/cnn_model_a4.pth")
CLASSIFIER_TARGET_CHANNEL = "A4"
CLASSIFICATION_INTERVAL_SEC = 1.0
ANALOG_START_INDEX = 5  # nSeq + 4 digital IOs precede the analog channels by default.

def show_menu():
    print("\nAvailable Commands:")
    for id in MENU_INPUT.keys():
        print(f"{id} | {MENU_INPUT[id]}")

def action_decode(action):
    if action == '0': return 'devices'
    elif action == '1': return f"config,{input('MAC or DEVICE_ID: ')}"
    elif action == '2':
        device = input('MAC or DEVICE_ID: ')
        param = input('Param (samplingfreq | activechannels): ')
        value = input('Value: ')
        return f'config,{device},{param},{value}'
    elif action == '3': return f"enable,{input('MAC or DEVICE_ID: ')}"
    elif action == '4': return f"disable,{input('MAC or DEVICE_ID: ')}"
    elif action == '5': return 'start'
    elif action == '6':
        device = input('MAC or DEVICE_ID: ')
        channel = input('Digital output channel number: ')
        state = input('Enable (1) or disable (0)?: ')
        return f'set_digital_output_channel,{device},{channel},{state}'
    elif action == '7': return 'stop'
    elif action == '8': return ''
    else: return ''


# ---------- Helper classes ----------
class SaveAcquisition:
    def __init__(self, output_dir="data/live_streams"):
        self.output_dir = Path(output_dir)
        self.file_handle = None
        self.file_path = None
        self.lock = threading.Lock()
        self.metadata = None
        self.device_key = "LiveStream"
        self.analog_channels = None
        self.file_header_written = False
        self.default_sampling_rate = 1000
        self.frame_width = None
        self.float_start_index = None

    def start(self):
        if self.file_handle is not None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        file_slug = self.device_key.replace(":", "_").replace(" ", "_")
        self.file_path = self.output_dir / f"opensignals_{file_slug}_{ts}.txt"
        self.file_handle = self.file_path.open("w", encoding="utf-8")
        self.file_header_written = False
        print(f"Acquisition started. Writing samples to {self.file_path}.")

    def stop(self):
        if self.file_handle:
            self.file_handle.close()
            print(f"Acquisition stopped. Samples saved to {self.file_path}.")
        else:
            print("Acquisition stopped.")
        self.file_handle = None
        self.file_path = None
        self.file_header_written = False

    def update_metadata_from_packet(self, packet):
        metadata = self._extract_metadata(packet)
        if not metadata:
            return
        self.metadata = metadata
        first_key = next(iter(metadata.keys()), None)
        if first_key:
            self.device_key = first_key
        info = next(iter(metadata.values()), {})
        channels = info.get("channels")
        if isinstance(channels, int) and channels > 0:
            self.analog_channels = channels
        sampling_rate = info.get("sampling rate")
        try:
            if sampling_rate:
                self.default_sampling_rate = float(sampling_rate)
        except (TypeError, ValueError):
            pass

    def save_frame(self, frame):
        if self.file_handle is None:
            return
        with self.lock:
            self._update_frame_structure(frame)
            self._write_header_if_needed()
            line = "\t".join(self._format_value(idx, value) for idx, value in enumerate(frame))
            self.file_handle.write(f"{line}\n")
            self.file_handle.flush()

    def _extract_metadata(self, packet):
        if not isinstance(packet, dict):
            return None
        rd = packet.get("returnData")
        if isinstance(rd, dict):
            filtered = {}
            for key, value in rd.items():
                if isinstance(value, dict) and "data" not in value:
                    filtered[key] = value
            if filtered:
                return filtered
        return None

    def _write_header_if_needed(self):
        if not self.file_handle or self.file_header_written:
            return
        header_metadata = self.metadata or self._default_metadata()
        header_line = json.dumps(header_metadata)
        self.file_handle.write(f"# {header_line}\n")
        self.file_handle.write("# EndOfHeader\n")
        self.file_header_written = True

    def _default_metadata(self):
        now = datetime.now()
        analog = self.analog_channels or (self.frame_width - 5 if self.frame_width else 0)
        analog = max(int(analog or 0), 0)
        resolution = [16] * analog if analog else []
        column_str = f"A1-A{analog}" if analog else ""
        return {
            self.device_key: {
                "sampling rate": self.default_sampling_rate,
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S.%f")[:-3],
                "resolution": resolution,
                "channels": analog,
                "column": column_str,
            }
        }

    def _update_frame_structure(self, frame):
        if self.frame_width is None:
            self.frame_width = len(frame)
        if self.float_start_index is None:
            if self.analog_channels and self.frame_width:
                seq_and_digital = self.frame_width - self.analog_channels
                self.float_start_index = max(seq_and_digital, 1)
            else:
                # assume BITalino layout: seq + 4 digital + analog channels
                self.float_start_index = 5 if self.frame_width > 5 else 1

    def _format_value(self, idx, value):
        if isinstance(value, (int, float)):
            if idx == 0:
                return str(int(round(value)))
            if (
                self.float_start_index is not None
                and idx >= self.float_start_index
            ):
                return f"{float(value):.6f}"
            if isinstance(value, float) and not value.is_integer():
                return f"{float(value):.6f}"
            return str(int(round(value)))
            return str(value)


class CNN1D(nn.Module if nn else object):
    """Lightweight 1D CNN used for inference; matches the training architecture."""

    def __init__(self, input_channels: int, num_classes: int, window_length: int):
        if nn is None:
            raise RuntimeError("PyTorch is required for LiveClassifier but is not installed.")
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        flattened_size = (window_length // 2) * 64
        self.fc1 = nn.Linear(flattened_size, 64)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout(x)
        return self.fc2(x)


class LiveClassifier:
    """Streams frames into the trained CNN and emits predictions every second."""

    def __init__(
        self,
        model_path: Path,
        target_channel: str = CLASSIFIER_TARGET_CHANNEL,
        classification_interval_sec: float = CLASSIFICATION_INTERVAL_SEC,
        on_prediction: Optional[Callable[[str, float], None]] = None,
    ):
        self.model_path = Path(model_path)
        self.target_channel = target_channel.upper()
        self.classification_interval_sec = classification_interval_sec
        self.on_prediction = on_prediction
        self.device = self._detect_device()
        self.model: Optional[CNN1D] = None
        self.label_encoder = None
        self.class_names: List[str] = []
        self.window_length: Optional[int] = None
        self.buffer: deque[float] = deque()
        self.sample_rate = 1000
        self.interval_samples = max(1, int(self.sample_rate * self.classification_interval_sec))
        self.samples_since_last = 0
        self.channel_index: Optional[int] = None
        self.column_names: Optional[List[str]] = None
        self.available = False
        self._load_model()

    def _detect_device(self):
        if torch is None:
            return None
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self):
        if torch is None:
            print("⚠️ PyTorch is not installed; live classification is disabled.")
            return
        if not self.model_path.exists():
            print(f"⚠️ Model not found at {self.model_path}; live classification disabled.")
            return
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.window_length = int(checkpoint.get("window_length", 0) or 0)
        if self.window_length <= 0:
            print("⚠️ Invalid window_length in checkpoint; classification disabled.")
            return
        num_classes = int(checkpoint.get("n_classes") or 0)
        n_channels = int(checkpoint.get("n_channels") or 1)
        label_encoder = checkpoint.get("label_encoder")
        if label_encoder is not None and hasattr(label_encoder, "classes_"):
            self.class_names = list(label_encoder.classes_)
            self.label_encoder = label_encoder
        else:
            print("⚠️ Label encoder missing in checkpoint; classification disabled.")
            return
        self.model = CNN1D(n_channels, num_classes, self.window_length)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.buffer = deque(maxlen=self.window_length)
        self.available = True
        print(f"✅ Loaded classifier from {self.model_path} ({len(self.class_names)} classes).")

    def set_prediction_callback(self, callback: Callable[[str, float], None]):
        self.on_prediction = callback

    def update_metadata_from_packet(self, packet: dict):
        """Extract sampling rate / column info from JSON packets."""
        if not isinstance(packet, dict):
            return
        rd = packet.get("returnData")
        candidates = {}
        if isinstance(rd, dict):
            for key, value in rd.items():
                if isinstance(value, dict) and "data" not in value:
                    candidates[key] = value
        if not candidates:
            return
        first = next(iter(candidates.values()))
        rate = first.get("sampling rate")
        if rate:
            try:
                rate = float(rate)
                if rate > 0:
                    self.sample_rate = int(rate)
                    self.interval_samples = max(1, int(self.sample_rate * self.classification_interval_sec))
            except (TypeError, ValueError):
                pass
        cols = first.get("column")
        if isinstance(cols, list):
            self.column_names = [str(c).strip().upper() for c in cols]
        elif isinstance(cols, str):
            text = cols.strip().upper()
            expanded: List[str] = []
            if "-" in text and text.count("A") >= 1:
                try:
                    start_token, end_token = text.split("-")
                    start_idx = int("".join(filter(str.isdigit, start_token)))
                    end_idx = int("".join(filter(str.isdigit, end_token)))
                    for idx in range(start_idx, end_idx + 1):
                        expanded.append(f"A{idx}")
                except ValueError:
                    expanded = [text]
            else:
                expanded = [text]
            self.column_names = expanded

    def add_frame(self, frame: list):
        if not self.available or self.model is None or self.window_length is None:
            return
        value = self._extract_channel_value(frame)
        if value is None:
            return
        self.buffer.append(value)
        self.samples_since_last += 1
        if len(self.buffer) < self.window_length:
            return
        if self.samples_since_last < self.interval_samples:
            return
        self.samples_since_last = 0
        self._classify_latest()

    def _extract_channel_value(self, frame: list):
        if self.channel_index is None:
            self.channel_index = self._infer_channel_index(frame)
        if self.channel_index is None or self.channel_index >= len(frame):
            return None
        try:
            return float(frame[self.channel_index])
        except (TypeError, ValueError):
            return None

    def _infer_channel_index(self, frame: list):
        if self.column_names:
            normalized = [str(col).strip().upper() for col in self.column_names]
            if self.target_channel in normalized:
                return normalized.index(self.target_channel)
        analog_start = ANALOG_START_INDEX if len(frame) > ANALOG_START_INDEX else 1
        digits = "".join(ch for ch in self.target_channel if ch.isdigit())
        channel_offset = max(0, int(digits) - 1) if digits else 0
        idx = analog_start + channel_offset
        return idx if idx < len(frame) else None

    def _classify_latest(self):
        if torch is None or self.model is None or self.device is None:
            return
        window = np.array(self.buffer, dtype=np.float32)
        if window.shape[0] != self.window_length:
            tail = np.zeros(self.window_length, dtype=np.float32)
            tail[-window.shape[0]:] = window
            window = tail
        tensor = torch.from_numpy(window).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
        label = self.class_names[pred_idx.item()]
        conf_value = confidence.item()
        if self.on_prediction:
            try:
                self.on_prediction(label, conf_value)
            except Exception as exc:  # pragma: no cover
                print("Prediction callback error:", exc)
        else:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] 🔮 Predicted {label} ({conf_value:.2f})")


class LivePlot:
    def __init__(self, window_size=1000):
        self.data = deque([0]*window_size, maxlen=window_size)
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(range(window_size), list(self.data))
        self.ax.set_ylim(-1024, 1024)
        self.ax.set_title("Live Signal (from OpenSignals)")
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Amplitude")
        self.status_text = self.ax.text(
            0.01, 0.95, "",
            transform=self.ax.transAxes,
            fontsize=10,
            color="tab:orange",
            va="top",
        )
        self.fig.show()

    def update(self, new_sample):
        self.data.append(new_sample)
        self.line.set_ydata(self.data)
        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.line)
        self.fig.canvas.flush_events()

    def set_prediction(self, label: str, confidence: float):
        if self.status_text is None:
            return
        self.status_text.set_text(f"Last: {label} ({confidence:.2f})")
        self.fig.canvas.draw_idle()


# ---------- Core client ----------
class TCPClient:
    def __init__(self, ip='127.0.0.1', port=5555, classifier: Optional[LiveClassifier] = None):
        self.tcpIp, self.tcpPort = ip, port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.buffer_size = 99999
        self.inputCheck, self.outputCheck = [], []
        self.msgQueue = queue.Queue()
        self.isChecking = False
        self.isAcquiring = False
        self.txtFile = SaveAcquisition()
        self.classifier = classifier

    def connect(self):
        print(f"Connecting to {self.tcpIp}:{self.tcpPort} ...")
        self.socket.connect((self.tcpIp, self.tcpPort))
        print("Connected!")
        self.outputCheck.append(self.socket)
        self.isChecking = True

    def start(self):
        threading.Thread(target=self.msgChecker, daemon=True).start()

    def stop(self):
        self.isChecking = False
        self.socket.close()
        print("Connection closed.")

    def msgChecker(self):
        while self.isChecking:
            readable, writable, _ = select.select(self.inputCheck, self.outputCheck, self.inputCheck)
            for s in readable:
                message = s.recv(self.buffer_size)
                if len(message) == 0:
                    continue  # nothing received
                if not self.isAcquiring:
                    try:
                        packet = json.loads(message.decode())
                        self.txtFile.update_metadata_from_packet(packet)
                        if self.classifier:
                            self.classifier.update_metadata_from_packet(packet)
                        print("Received (status):", packet)
                    except:
                        print("Status message (non-JSON):", message[:100])
                    self.inputCheck = []
                else:
                    # during acquisition, parse JSON packets
                    try:
                        msg_str = message.decode()
                        packet = json.loads(msg_str)
                        self.txtFile.update_metadata_from_packet(packet)
                        if self.classifier:
                            self.classifier.update_metadata_from_packet(packet)
                        rd = packet.get("returnData")
                        if rd is None:
                            continue

                        # Some firmware versions wrap the JSON payload in a string.
                        if isinstance(rd, str):
                            try:
                                rd = json.loads(rd)
                            except json.JSONDecodeError:
                                continue

                        # Normalise into (device_key, samples) pairs.
                        if isinstance(rd, dict):
                            iterable = rd.items()
                        elif isinstance(rd, list):
                            iterable = enumerate(rd)
                        else:
                            iterable = []

                        for _, samples in iterable:
                            # Device payload can be either a raw list of samples or
                            # another dict with a `data` list.
                            if isinstance(samples, dict):
                                samples = samples.get("data")
                            if not isinstance(samples, list) or not samples:
                                continue
                            for frame in samples:
                                if isinstance(frame, list) and frame:
                                    value = frame[-1]
                                else:
                                    value = frame
                                self.txtFile.save_frame(frame)
                                if self.classifier:
                                    self.classifier.add_frame(frame)
                                update_queue.put(value)
                    except json.JSONDecodeError:
                        # If message is split across packets, skip silently
                        pass
                    except Exception as e:
                        print("Decode error:", e)

            for s in writable:
                try:
                    next_msg = self.msgQueue.get_nowait()
                    s.send(str(next_msg).encode())
                except queue.Empty:
                    continue

    def addMsgToSend(self, data):
        if "start" in data: self.setIsAcquiring(True)
        elif "stop" in data: self.setIsAcquiring(False)
        self.msgQueue.put(data)
        if self.socket not in self.outputCheck: self.outputCheck.append(self.socket)
        if self.socket not in self.inputCheck: self.inputCheck.append(self.socket)

    def setIsAcquiring(self, isAcquiring):
        self.isAcquiring = isAcquiring
        if isAcquiring: self.txtFile.start()
        else: self.txtFile.stop()


# ---------- Main ----------
plotter = LivePlot(window_size=1000)
update_queue = SimpleQueue()
classifier = LiveClassifier(CLASSIFIER_MODEL_PATH)

def handle_prediction(label: str, confidence: float):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] 🔮 Predicted {label} ({confidence:.2f})")
    plotter.set_prediction(label, confidence)

classifier.set_prediction_callback(handle_prediction)

client = TCPClient(ip='127.0.0.1', port=5555, classifier=classifier)
client.connect()
client.start()

plt.pause(0.5)
print("✅ Ready! Press '5' to start acquisition once OpenSignals is running.")

def refresh_plot():
    """GUI-safe background thread that updates the live plot."""
    while True:
        while not update_queue.empty():
            val = update_queue.get()
            plotter.update(val)
        time.sleep(0.01)

threading.Thread(target=refresh_plot, daemon=True).start()

# Command loop
while True:
    show_menu()
    user_action = input("Select action: ")
    if user_action == '5':
        client.setIsAcquiring(True)
    elif user_action == '7':
        client.setIsAcquiring(False)
    elif user_action == '8':
        client.stop()
        break
    msg = action_decode(user_action)
    client.addMsgToSend(msg)
