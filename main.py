from Detector import *

detector = Detector(model_type='OD')

# detector.onImage("data/img/test/0.png")
# detector.onImage("data/img/people1.jpg")
detector.onVideo("data/videos/Les filles de Poutine sanctionnées.mp4")
