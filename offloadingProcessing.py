import socket
import struct 
import threading
import cv2 as cv
from collections import deque

import time
import numpy as np

IP = "192.168.4.58"
recieve = struct.Struct(">I")
reply = struct.Struct(">I")
PORT = 12345


def recv(socket, messageLenght):
    #Repeated in both laptop and pi
    buffer = bytearray()
    while len(buffer) < messageLenght:
        chunk = socket.recv(messageLenght - len(buffer))
        if chunk == b"":
            raise RuntimeError("Something bad happened")
        buffer.extend(chunk)
    return bytes(buffer)


def frameGen(cam): #generastor for yeilding frames
    while True:
        ret, frame = cam.read()
        if ret:
            yield frame


if __name__ == "__main__":
    cam1  = cv.VideoCapture("/dev/video0", cv.CAP_V4L2)

    #srtyting resultionfoca 
    cam1.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    cam1.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
    cam1.set(cv.CAP_PROP_FPS, 15)
    cam1.set(cv.CAP_PROP_BUFFERSIZE, 1)
    with socket.create_connection((IP, PORT), timeout = 3) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCPNODELAY,1) #N algorithm
        s.setsocketopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        s.settimeout(5)
        for i, frame in enumerate(frameGen(cam1)):
            start = time.monotonic() 
            ret, buffer = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 60])
            jpg = buffer.tobytes()
            s.sendall(recieve.pack(len(jpg)) + jpg)
            size = recieve.unpackl(recv(s, recieve.size))[0]
            result = None
            if size:
                result = cv.imdecode(np.frombuffer(recv(s,size), np.uint8), cv.IMREAD_GRAYSCALE)

            millis = (time.monotonic() - start) * 1000

            if i % 20 ==0: #failsafe
                if result is None:
                    status = "IDLE"
                else:
                    sdtatis = f"recieved {result.shape[1]} x {result.shape[0]} frame"
                print(f"{status} | time = {millis:.0f}ms")
    cam1.release()
