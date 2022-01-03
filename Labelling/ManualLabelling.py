# This is a sample Python script.
import os
import time
import glob
import cv2 as cv
from queue import Queue
import datetime

input_path = 'F:\\Term 2\\image_data3\\New Labels\\new'
output_path = 'F:\\Term 2\\image_data3\\New Labels'
mappings = {"u": "unknown",
            "s": "s",
            "j": "j",
            "i": "i",
            "o": "o",
            "c": "c",
            "e": "e"}

j = 0
# map the keys to their ordinal numbers
kMappings = {}
for key in mappings.keys():
    kMappings[ord(key)] = mappings[key]

print(f"[INFO] mappings: {mappings}")

print("[INFO] loading images...")
p = os.path.sep.join([input_path, '**', '*.png'])
print(p)

file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
print(f"[INFO] images found: {len(file_list)}")

# loop over the image paths
for filename in file_list:  # [::10]:
    img = cv.imread(filename)

    cv.imshow("image", img)
    k = cv.waitKey() & 0xFF
    # if the `q` key or ESC was pressed, break from the loop
    if k == ord("q") or k == 27:
        break
    # otherwise, check to see if a key was pressed that we are
    # interested in capturing
    elif k in kMappings.keys():
        # construct the path to the label subdirectory
        p = os.path.sep.join([output_path, kMappings[k]])
        if not os.path.exists(p):
            os.makedirs(p)

        # construct the path to the output image
        p = os.path.sep.join([p, os.path.basename(filename)])

        print(f"[INFO] saving image: {p}")
        cv.imwrite(p, img)
        print('[INFO] Deleting', filename)

        l = os.listdir(output_path + '\\' + kMappings[k])
        i = len(l) +2
        rename = output_path + '\\' + kMappings[k] + '\\' + kMappings[k] + str(i) + '.png'
        print(rename)
        os.rename(p, rename)
        print("Deleting the picture")
        os.remove(filename)
        j = j + 1
# storing the picture index
time = datetime.datetime.now()
file = open('picture count.txt', 'a+')

file.write("Current date and time = %s\n" % time)
file.write("Total pictures labelled = %d\r\n" % j)

file.close()