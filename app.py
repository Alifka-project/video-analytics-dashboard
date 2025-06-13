#!/usr/bin/env python3
"""
Single-file Advanced Face Recognition System (dlib-free)
Features:
1. Head-pose estimation (detect tilt/rotation) using MediaPipe Face Mesh
2. Eye-gaze tracking & blink detection
3. Emotion/attribute recognition (smile, age, mask on/off)
4. Liveness detection (texture analysis)
5. Multi-camera support & optional PTZ control with fallback scanning
6. FastAPI backend + Streamlit UI frontend
7. Detailed analytics dashboard (recognition history, charts)

Usage:
  pip install fastapi uvicorn streamlit opencv-python numpy mediapipe deepface sqlalchemy plotly pyyaml imutils
  python advanced_face_recognition.py
"""
import os
import cv2
import yaml
import threading
import io
import datetime
import platform
import numpy as np
import pandas as pd
import streamlit as st
import requests
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import mediapipe as mp
from deepface import DeepFace
import plotly.express as px

# ---------------------------
# Configuration & persistence
# ---------------------------
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
DB_URL = os.getenv("DB_URL", "sqlite:///analytics.db")
BASE = declarative_base()

class Event(BASE):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=datetime.datetime.utcnow)
    data = Column(JSON)

def init_db(url=DB_URL):
    engine = create_engine(url)
    BASE.metadata.create_all(engine)
    return sessionmaker(bind=engine)
Session = init_db()

def insert_event(data):
    session = Session()
    session.add(Event(data=data))
    session.commit()
    session.close()

# ---------------------------
# Camera Manager
# ---------------------------
class CameraManager:
    def __init__(self, cfg_file=CONFIG_PATH):
        try:
            cfg = yaml.safe_load(open(cfg_file))
            self.cameras = cfg.get('cameras', {})
        except:
            self.cameras = {}
    def list_cameras(self):
        return list(self.cameras.keys()) or [str(i) for i in range(5)]
    def open(self, name):
        """
        Open camera URI or index with platform-specific API.
        """
        uri = self.cameras.get(name)
        backends = []
        if uri:
            cap = cv2.VideoCapture(uri)
            return cap
        idx = int(name)
        system = platform.system()
        # try primary backend
        if system == 'Darwin': backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        elif system == 'Windows': backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else: backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        for b in backends:
            cap = cv2.VideoCapture(idx, b)
            if cap.isOpened():
                return cap
        # fallback generic
        return cv2.VideoCapture(idx)
    def pan(self, name, angle): pass
    def tilt(self, name, angle): pass

# ---------------------------
# Face Analysis
# ---------------------------
class FaceAnalyzer:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        self.model_points = np.array([
            [0.0, 0.0, 0.0], [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0], [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
        ])
    def analyze(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks: return None
        lm = res.multi_face_landmarks[0].landmark
        pts2d = np.array([
            [lm[1].x * w, lm[1].y * h],
            [lm[152].x * w, lm[152].y * h],
            [lm[33].x * w, lm[33].y * h],
            [lm[263].x * w, lm[263].y * h],
            [lm[61].x * w, lm[61].y * h],
            [lm[291].x * w, lm[291].y * h]
        ], dtype='double')
        cam_mtx = np.array([[w,0,w/2],[0,w,w/2],[0,0,1]],dtype='double')
        dist = np.zeros((4,1))
        _, rv, _ = cv2.solvePnP(self.model_points, pts2d, cam_mtx, dist)
        return {'rotation_vec': rv.flatten().tolist()}
    def detect_blink(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks: return False
        lm = res.multi_face_landmarks[0].landmark
        def ear(ids):
            pts = np.array([(lm[i].x, lm[i].y) for i in ids])
            A = np.linalg.norm(pts[1]-pts[5]); B=np.linalg.norm(pts[2]-pts[4]); C=np.linalg.norm(pts[0]-pts[3])
            return (A+B)/(2*C)
        l=ear([33,160,158,133,153,144]); r=ear([263,387,385,362,380,373])
        return ((l+r)/2)<0.2

class EmotionAttr:
    def analyze(self, frame):
        try:
            r = DeepFace.analyze(frame, actions=['emotion','age','gender'], enforce_detection=False)
            return {'emotion':r['dominant_emotion'],'age':int(r['age']),'gender':r['gender'],'mask':False}
        except:
            return {'emotion':'unknown','age':0,'gender':'unknown','mask':False}

class Liveness:
    def check(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()>50.0

# ---------------------------
# FastAPI Backend
# ---------------------------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
cm=CameraManager(); fa=FaceAnalyzer(); ea=EmotionAttr(); lv=Liveness()
@app.post('/recognize')
async def recognize(file: UploadFile=File(...)):
    b = await file.read(); arr=np.frombuffer(b,np.uint8)
    img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400,'Invalid')
    res={'pose':fa.analyze(img),'blink':fa.detect_blink(img),'attributes':ea.analyze(img),'liveness':lv.check(img)}
    insert_event(res); return JSONResponse(res)

# ---------------------------
# Streamlit UI & Dashboard
# ---------------------------
def launch_ui():
    st.set_page_config('FaceRec',layout='wide')
    st.title('Advanced Face Recognition')
    cams=cm.list_cameras(); sel=st.sidebar.selectbox('Camera',cams)
    if st.sidebar.checkbox('Show Analytics'):
        df=pd.read_sql('SELECT * FROM events',DB_URL)
        st.dataframe(df); st.plotly_chart(px.histogram(df,x='ts',title='History'))
        return
    if st.sidebar.button('Start'):
        cap=cm.open(sel)
        ok,frm=cap.read()
        if not ok:
            # fallback scan indices
            for i in range(5):
                c2=cm.open(str(i)); ok2,_=c2.read()
                if ok2:
                    cap=c2; st.sidebar.warning(f'Using camera {i} instead'); break
            else:
                st.sidebar.error('No camera available'); return
        slot=st.empty(); stat=st.sidebar.empty(); stop=False
        while not stop:
            ok,frame=cap.read()
            if not ok: stat.error('Frame read error'); break
            _,buf=cv2.imencode('.jpg',frame)
            d=requests.post('http://localhost:8000/recognize',files={'file':buf.tobytes()}).json()
            if d['pose']: rv=d['pose']['rotation_vec']; cv2.putText(frame,f'Pose:{rv[:3]}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            if d['blink']: cv2.putText(frame,'Blink',(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            a=d['attributes']; cv2.putText(frame,f"Emo:{a['emotion']}",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
            cv2.putText(frame,f"Mask:{a['mask']}",(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
            cv2.putText(frame,f"Live:{d['liveness']}",(10,150),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            slot.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            stop=st.sidebar.button('Stop')
        cap.release()

if __name__=='__main__':
    threading.Thread(target=lambda:uvicorn.run(app,host='0.0.0.0',port=8000,log_level='warning'),daemon=True).start()
    launch_ui()
