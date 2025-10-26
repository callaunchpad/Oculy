# 🔌 OpenSignals TCP/IP Live Stream (Fixed for JSON packets)
# Requirements:
#  - OpenSignals running with TCP-IP enabled
#  - Acquisition started (green ▶ pressed in OpenSignals)
#  - Port set to 5555 in Integration settings

import socket, json, threading, select, queue, time, warnings
import matplotlib.pyplot as plt
from collections import deque
from queue import SimpleQueue

warnings.filterwarnings("ignore", category=UserWarning)

MENU_INPUT = {
    0: 'devices', 1: 'config,{MAC|DEVICE_ID}', 2: 'config,{MAC|DEVICE_ID}, {PARAM}, {VALUE}',
    3: 'enable,{MAC|DEVICE_ID}', 4: 'disable,{MAC|DEVICE_ID}', 5: 'start',
    6: 'set_digital_output, {MAC|DEVICE_ID}, {CHANNEL}, {STATE}', 7: 'stop', 8: 'exit'
}

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
    def start(self): print("Acquisition started (file writer would start here).")
    def stop(self): print("Acquisition stopped.")


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
        self.fig.show()

    def update(self, new_sample):
        self.data.append(new_sample)
        self.line.set_ydata(self.data)
        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.line)
        self.fig.canvas.flush_events()


# ---------- Core client ----------
class TCPClient:
    def __init__(self, ip='127.0.0.1', port=5555):
        self.tcpIp, self.tcpPort = ip, port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.buffer_size = 99999
        self.inputCheck, self.outputCheck = [], []
        self.msgQueue = queue.Queue()
        self.isChecking = False
        self.isAcquiring = False
        self.txtFile = SaveAcquisition()

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
                        print("Received (status):", packet)
                    except:
                        print("Status message (non-JSON):", message[:100])
                    self.inputCheck = []
                else:
                    # during acquisition, parse JSON packets
                    try:
                        msg_str = message.decode()
                        packet = json.loads(msg_str)
                        if "returnData" in packet:
                            rd = packet["returnData"]
                            # iterate through all devices
                            for dev_key, samples in rd.items():
                                if isinstance(samples, list) and len(samples) > 0:
                                    # take the last analog channel sample
                                    if isinstance(samples[-1], list):
                                        val = samples[-1][-1]
                                        update_queue.put(val)
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
client = TCPClient(ip='127.0.0.1', port=5555)
client.connect()
plotter = LivePlot(window_size=1000)
update_queue = SimpleQueue()
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
