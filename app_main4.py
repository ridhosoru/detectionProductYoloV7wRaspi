#import library
from PyQt5 import QtWidgets, uic, QtCore, QtCore, QtGui, uic
from PyQt5.QtCore import QDir, QThread, pyqtSignal, pyqtSlot, Qt,QPoint,QTime,QDate,QTimer,QDate
from PyQt5.QtGui import QImage, QPixmap
import sys
import time
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QGraphicsDropShadowEffect,QMessageBox
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
from datetime import datetime
import timeit
import cv2
import torch
from numpy import random
import numpy as np
import sys
import cv2
from threading import Thread
from pathlib import Path
#import utils
#call file
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import RPi.GPIO as GPIO
import time


#define pin raspi
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(23, GPIO.OUT)
GPIO.setup(24, GPIO.OUT)
GPIO.output(23,GPIO.HIGH)
GPIO.output(24,GPIO.HIGH)
sensorp=16
GPIO.setup(sensorp,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)

# database

#opt and class
classes_to_filter =['screwringmerah', 'screwringputih', 'screwringungu', 'capungu', 'capmerahmuda', 'caphijau', 'bottlemerah', 'bottlebiru', 'bottleungu']

opt = {
    # Path to weights file default weights are for nano model
    "weights": "best.pt",    
    "source": "capture.png", #load img
    "save-txt":"",
    "yaml": "data/custom_data.yaml",
    "img-size": 640,  # default image size
    "conf-thres": 0.5,  # confidence threshold for inference.
    "iou-thres": 0.45,  # NMS IoU threshold for inference.
    "device": '',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes": classes_to_filter , # list of classes to filter or None
    "nosave": '',
    "project":'hasil',
    "name":'',
    "exist_ok":'',
    "augmentasi":''
}

timestart= timeit.default_timer()
timestop = timeit.default_timer()
waktuexe = (timestop-timestart)
print(waktuexe*10**3)

#class communicate
class Communicate(QObject):
    sig = pyqtSignal(str)

#class thread worker
class Worker(QThread):
    def __init__(self, parent=None, communicate = Communicate()):
        super(Worker,self).__init__(parent)
        self.communicate = communicate
        self.loop = Loop(communicate=self.communicate)
    def run(self):
        self.loop.methodA()

class Loop(object):
    def __init__(self,communicate = Communicate()):
        self.count = 0
        self.communicate = communicate
    def methodA(self):
        
        while True :
            if GPIO.input(sensorp):
                time.sleep(1)
                self.communicate.sig.emit(f"{self.count}")
            else :
                GPIO.output(23,1)
                GPIO.output(24,1)

                
                

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('mainwindow.ui', self) 
        self.__press_pos = QPoint()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.show()
        self._run_flag = True
        self.closebtn.clicked.connect(self.closew)
        self.minimizebtn.clicked.connect(self.minw)
        global date
        date = QDate.currentDate()
        date = date.toString("dd MMM yyyy")
        self.datelbl.setText(date)
        self.timer = QTimer()
        self.timer.timeout.connect(self.timelbl)
        self.timer.start(1000)
        self.timelbl()
        #self.opencamBtn.clicked.connect(self.sensorstr)
        #self.opencamBtn.clicked.connect(self.threadopcam)
        #self.startdetBtn.clicked.connect(self.threaddet)
        #self.capt_btn.clicked.connect(self.captimgg)
        #self.startbtn.clicked.connect(self.startsys)
        ####communicate
        self.communicate = Communicate()
        self.communicate.sig[str].connect(self.connect)
        self.startbtn.clicked.connect(self.startthr)
        self.thread=Worker(communicate = self.communicate)      
        
    def startthr(self):
        print("start")
        self.thread.start()

    def connect(self,text):
        if text == "0":
            print("detected")
            self.statusdet.setText("detected")
            self.capturing()
            self.statusdet.setText("processing")
            print("processing")
            self.processing()
    
    def processing(self):
        self.threaddet()
    
    def capturing(self):
        self.captimgg()
        #time.sleep(0.5)
        
    def threaddet(self):
        with torch.no_grad():
            self.yolov7()
        self.lbl_det.setPixmap(QtGui.QPixmap("hasil.png"))  
    #@QtCore.pyqtSlot()
    def yolov7(self):
        save_img = not opt['nosave'] 
        #load source,device,model,stride,imgsz
        save_txt= opt['save-txt']
        #save_dir = Path(increment_path(Path(opt['project']) / opt['name'], exist_ok=opt['exist_ok']))  # increment run
        #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        source = (opt["source"])
        device = select_device(opt['device'])
        model = attempt_load(opt['weights'],map_location='cpu')
        stride = int(model.stride.max())
        imgsz = check_img_size(opt['img-size'], s=stride)
        #make a hal 
        half = device.type != 'cpu'
        if half:
            model.half()  # to FP16
        classify = False
        #clasify obj
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        #get name color
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        #run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt['augmentasi'])[0]
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt['augmentasi'])[0]
        t2 = time_synchronized()
        pred = non_max_suppression(pred,0.5)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            #save_path = str(save_dir / p.name)  # img.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        #with open(txt_path + '.txt', 'a') as f:
                        #    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            if save_img:
                cv2.imwrite("hasil.png", im0)
                #print(f" The image with the result is saved in: {save_path}")
            count_of_object = int(n)
            x= self.tp_1.text()
            x2=self.tp2.text()
            a2=int(x2)
            a=int(x)
            y= a + count_of_object
            y2=a2+count_of_object
            nameprod = names[int(c)]
            nameprodstr = str(nameprod)
            napr = str(self.cb1.currentText())
            if self.cb1.currentText()==nameprod or self.cb2.currentText()==nameprod :
                self.pdnm.setText(str(nameprod))
                if self.cb1.currentText()==nameprod:
                    GPIO.output(23,0)
                    self.qty1.setText(str(count_of_object))
                    self.tp_1.setText(str(y))
                    self.statusdet.setText("done")
                if self.cb2.currentText() == nameprod:
                    GPIO.output(24,0)
                    self.qty2.setText(str(count_of_object))
                    self.tp2.setText(str(y2))
                    self.statusdet.setText("done")

            else:
                QMessageBox.about(self, "warning", "choose right product")
                self.tp_1.setText("0")
                self.tp2.setText("0")   
            print(label)         
            print(count_of_object)

    def captimgg(self):
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT,320)
        #for result in range(10):
            #result,imm = cam.read()
        #cv2.imwrite("capture.png",imm)
        #cv2.waitKey(0)
        ##
        if cam.isOpened():
            current_frame = 0
            while True:
                for ret in range(10):
                    ret, frame = cam.read()
                    if ret:
                        cv2.imwrite("capture.png", frame)
                        self.lbl_opcam.setPixmap(QtGui.QPixmap("capture.png"))  
                    current_frame += 1
                break
            cam.release()
        ##
        cv2.destroyAllWindows()

    def timelbl(self):
        time = datetime.now()
        global formate_time
        formate_time = time.strftime("%I:%M:%S %p")
        self.timelabel.setText(formate_time)

    def minw(self):
        self.showMinimized()
    
    def closew(self):
        self.close()
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.__press_pos = event.pos()  

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.__press_pos = QPoint()

    def mouseMoveEvent(self, event):
        if not self.__press_pos.isNull():  
            self.move(self.pos() + (event.pos() - self.__press_pos))
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()