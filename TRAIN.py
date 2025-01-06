# Copyright (c) 2024 [RICHEASYGOAT]
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from ultralytics import YOLO
import cv2
import numpy as np
from roboflow import Roboflow

rf = Roboflow(api_key="7idEzKSCpPKLTJwlUp3j")
project = rf.workspace("yolov11-midrp").project("hoi-gmd2i")
version = project.version(5)
dataset = version.download("yolov8")

rf = Roboflow(api_key="7idEzKSCpPKLTJwlUp3j")
project = rf.workspace("test-vongh").project("awkward-posture-of-human")
version = project.version(3)
dataset = version.download("yolov8")

model = YOLO('yolov8n-pose.pt')
model = YOLO('yolov8n.pt')


results = model.train(data="/home/idc/ultralytics/HOI-3/data.yaml", epochs=200, imgsz=640)
results = model.train(data="/home/idc/ultralytics/Awkward-posture-of-human-3/data.yaml", epochs=300, imgsz=640)

# model = YOLO('/home/idc/ultralytics/runs/pose/train/weights/best.pt')
# metrics = model.val(data="/home/idc/ultralytics/Awkward-posture-of-human-3/data.yaml")


