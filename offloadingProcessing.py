import socket
import struct 
import threading
import cv2 as cv
from collections import deque

import time
import numpy as np

IP = "192.168.4.58"
structRef = struct.Struct(">II")
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


# def frameGen(cam): #generastor for yeilding frames
#     while True:
#         ret, frame = cam.read()
#         if ret:
#             yield frame


if __name__ == "__main__":
    camL  = cv.VideoCapture("/dev/video0", cv.CAP_V4L2)
    camR = cv.VideoCapture("/dev/video0", cv.CAP_V4L2)

    #srtyting resultionfoca 
    camL.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    camL.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
    camL.set(cv.CAP_PROP_BUFFERSIZE, 1)

    camR.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    camR.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
    camR.set(cv.CAP_PROP_BUFFERSIZE, 1)    
    with socket.create_connection((IP, PORT), timeout = 3) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY,1) #N algorithm
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        s.settimeout(5)

        counter = 0
        while True:
            

            for i in range(8): #skip 8 frames. Constantly updasted wihtout needign to skip frames

                camL.grab()
                camR.grab()

            ret1, frameLeft = camL.retreive()
            ret2, frameRight = camR.retrieve()


            start = time.monotonic() 
            sendPayload = []

            grayL = cv.cvtColor(frameLeft, cv.COLOR_BGR2GRAY)
            grayR = cv.cvtColor(frameRight, cv.COLOR_BGR2GTAY)




            _, bufferLeft = cv.imencode(".jpg", grayL, [cv.IMWRITE_JPEG_QUALITY, 60])
            _, bufferRight = cv.imencode(".jpg", grayR, [cv.IMWRITE_JPEG_QUALITY, 60])

            sendPayload.append(bufferLeft.tobytes())
            sendPayload.append(bufferRight.tobytes())


            s.sendall(structRef.pack(len(sendPayload[0]), len(sendPayload[1])) + sendPayload[0] + sendPayload[1])
            recv(s,1)

          
            millis = (time.monotonic() - start) * 1000

            i+= 1

    camL.release()
    camR.release()