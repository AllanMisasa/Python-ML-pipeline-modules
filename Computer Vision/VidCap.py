import os
import rtsp
import time
import datetime
import cv2 as cv

rtsp_url = f"rtsp://admin:admin123@192.168.137.123:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1"

cap = cv.VideoCapture(rtsp_url)

def suffix(): # Just called once to get the starting number for the image names so there is no overlap
    nums = []
    files = os.listdir(".")
    for file in files:
        if file.startswith("frame"):
            nums.append(int(file[5:-4]))
    max_num = max(nums)
    return max_num

def snapshots(num_images): # Takes a snapshot of the video and saves it as a jpg. Input is the number of images to take.
    start_num = suffix()
    for image in range(num_images): 
        ret, frame = cap.read()
        suf = start_num + image + 1
        name = "frame%d.jpg" % suf
        cv.imwrite(name, frame)
        print("Saved %s" % name)

def repeated_snapshot(interval, num_images): # Takes a snapshot of the video every interval seconds.
    time_current = time.time()
    start_num = suffix()
    while True:
        if time.time() >= time_current + 1:
            ret, frame = cap.read()
            suf = start_num + 1
            name = "frame%d.jpg" % suf
            resized = cv.resize(frame, (640, 480))
            cv.imwrite(name, resized)
            print("Saved %s" % name)
            start_num += 1
        if time.time() >= 1 + time_current + (num_images*interval):
            break
        time.sleep(interval)
        
#snapshots(1)


repeated_snapshot(1, 10)

'''
def rtsp_snapshot(interval, num_images, width, height): # Takes a snapshot of the video every interval seconds.
    start_num = suffix()
    num = 0
    client = rtsp.Client(rtsp_server_uri = rtsp_url)
    while num < num_images:
        suf = start_num + 1

        client.read().resize([width, height]).save("frame%d.jpg" % suf)
        print("Saved %s" % suf)
        start_num += 1
        time.sleep(interval)
    client.close()


rtsp_snapshot(1, 10, 640, 480)
'''