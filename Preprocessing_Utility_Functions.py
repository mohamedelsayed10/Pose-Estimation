import numpy as np
import os,cv2,math,glob,random
import scipy.io as sio
from math import cos, sin
from pathlib import Path
import pandas as pd
import mediapipe
import warnings
warnings.filterwarnings('ignore')
import mediapipe as mp
import matplotlib.pyplot as plt
import cv2 


def draw_axis(img, pitch,yaw,roll, tdx=None, tdy=None, size = 100):

    yaw = -yaw
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


def process_image(image_path, faces):
    image = cv2.imread(image_path)
    results = faces.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return results

def extract_landmarks(results, imgname, xy, nose):
    xy[imgname] = []
    nose[imgname] = []
    
    if results.multi_face_landmarks is not None: 
        nose[imgname].append(results.multi_face_landmarks[0].landmark[4])
        
        distance = np.sqrt((results.multi_face_landmarks[0].landmark[4].x - results.multi_face_landmarks[0].landmark[152].x)**2 + 
                           (results.multi_face_landmarks[0].landmark[4].y - results.multi_face_landmarks[0].landmark[152].y)**2)

        for face in results.multi_face_landmarks:
            for landmark in face.landmark:
                x = (landmark.x - face.landmark[4].x) / distance
                y = (landmark.y - face.landmark[4].y) / distance
                xy[imgname].extend([x, y])

def process_jpg(imgname, xy, nose):
    face_module = mediapipe.solutions.face_mesh
    with face_module.FaceMesh(static_image_mode=True) as faces:
        image_path = fr'C:\Users\mohamed elsayed\Desktop\projects\Pose Estimation\AFLW2000\{imgname}'
        results = process_image(image_path, faces)
        extract_landmarks(results, imgname, xy, nose)

def process_mat(imgname, pitch, yaw, roll):
    mat = sio.loadmat(fr'C:\Users\mohamed elsayed\Desktop\projects\Pose Estimation\AFLW2000\{imgname}')
    pose_para = mat["Pose_Para"][0][:3]
    pitch.append(pose_para[0])
    yaw.append(pose_para[1])
    roll.append(pose_para[2]) 


def process_image_with_pca(image, pca):
    points = []
    nose = 0

    face_module = mp.solutions.face_mesh
    with face_module.FaceMesh(static_image_mode=True) as faces:
        results = faces.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks is not None:
            nose = results.multi_face_landmarks[0].landmark[4]
            distance = np.sqrt((results.multi_face_landmarks[0].landmark[4].x - results.multi_face_landmarks[0].landmark[152].x)**2 +
                               (results.multi_face_landmarks[0].landmark[4].y - results.multi_face_landmarks[0].landmark[152].y)**2)

            for face in results.multi_face_landmarks:
                for landmark in face.landmark:
                    x = (landmark.x - face.landmark[4].x) / distance
                    y = (landmark.y - face.landmark[4].y) / distance
                    points.append(x)
                    points.append(y)

    if len(points) == 0:
        return None, None
    dftest = pd.DataFrame(points).T
    dftest.columns = [f'x{i//2}' if i % 2 == 0 else f'y{i//2}' for i in range(dftest.shape[1])]
    dftest_pca = pca.transform(dftest)
    return dftest_pca, nose


def predict_on_image(image, Modelroll, Modelpitch, Modelyaw, pca):
    dftest_pca, nose = process_image_with_pca(image, pca)
    if dftest_pca is None:
        return None, None, None, None
    roll = Modelroll.predict(dftest_pca)
    pitch = Modelpitch.predict(dftest_pca)
    yaw = Modelyaw.predict(dftest_pca)
    return roll, pitch, yaw, nose


from moviepy.editor import VideoFileClip, AudioFileClip

def predict_on_video(video_path, output_path, votingModelroll, votingModelpitch, votingModelyaw, pca):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'AVC1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        roll, pitch, yaw, nose = predict_on_image(frame, votingModelroll, votingModelpitch, votingModelyaw, pca)
        if roll is not None and pitch is not None and yaw is not None and nose is not None:
            frame = draw_axis(frame, pitch[0], yaw[0], roll[0], tdx=int(nose.x * width), tdy=int(nose.y * height))

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile("saved_audio.wav", codec='pcm_s16le', fps=44100)

    processed_video = VideoFileClip(output_path)
    final_video = processed_video.set_audio(AudioFileClip("saved_audio.wav"))
    final_video.write_videofile(r"C:\Users\mohamed elsayed\Desktop\output_with_audio.mp4", codec='libx264', audio_codec='aac')
    

    os.remove("saved_audio.wav")
    os.remove(output_path)




