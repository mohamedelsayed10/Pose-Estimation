import streamlit as st
import cv2
import joblib
import os
import tempfile
from moviepy.editor import VideoFileClip, AudioFileClip
from Preprocessing_Utility_Functions import *
votingModelroll = joblib.load(r'C:\Users\mohamed elsayed\Desktop\Pose Estimation\saved model\roll_model.joblib')
votingModelpitch = joblib.load(r'C:\Users\mohamed elsayed\Desktop\Pose Estimation\saved model\pitch_model.joblib')
votingModelyaw = joblib.load(r'C:\Users\mohamed elsayed\Desktop\Pose Estimation\saved model\yaw_model.joblib')
pca = joblib.load(r'C:\Users\mohamed elsayed\Desktop\Pose Estimation\saved model\pca_model.joblib')



def process_video(video_path, votingModelroll, votingModelpitch, votingModelyaw, pca):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output path in the temporary directory
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "processed_video.mp4")
    temp_dir2 = tempfile.mkdtemp()

    output_path2 = os.path.join(temp_dir2, "processed_video2.mp4")

    # Create video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'AVC1')  # Adjust codec as needed
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Predict orientation and draw axes on the frame
        roll, pitch, yaw, nose = predict_on_image(frame, votingModelroll, votingModelpitch, votingModelyaw, pca)
        if roll is not None and pitch is not None and yaw is not None and nose is not None:
            frame = draw_axis(frame, pitch[0], yaw[0], roll[0], tdx=int(nose.x * width), tdy=int(nose.y * height))

        # Write the frame to the output video
        out.write(frame)


    # Release resources
    cap.release()
    out.release()
        # Extract audio from the original video
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile("saved_audio.wav", codec='pcm_s16le', fps=44100)

    # Combine the processed video with the saved audio
    processed_video = VideoFileClip(output_path)
    final_video = processed_video.set_audio(AudioFileClip("saved_audio.wav"))
    final_video.write_videofile(output_path2, codec='libx264', audio_codec='aac')
    

    # Clean up temporary audio file
    os.remove("saved_audio.wav")
    os.remove(output_path)

    return output_path2

def save_temp_file(uploaded_file, file_extension):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, f"uploaded_file.{file_extension}")
    with open(temp_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    return temp_path

def main():
    st.title("Pose Estimation App")

    option = st.radio("What would you like to do?", ("Upload Image", "Upload Video"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        file_extension = "jpg"

        if uploaded_file is not None:
            temp_file_path = save_temp_file(uploaded_file, file_extension)
            image = cv2.imread(temp_file_path)
            roll, pitch, yaw, nose = predict_on_image(image, votingModelroll, votingModelpitch, votingModelyaw, pca)
            if roll is not None and pitch is not None and yaw is not None and nose is not None:
                image_with_axes = draw_axis(image, pitch[0], yaw[0], roll[0], tdx=int(nose.x * image.shape[1]), tdy=int(nose.y * image.shape[0]))
                st.image(cv2.cvtColor(image_with_axes, cv2.COLOR_BGR2RGB), caption="Pose Estimated Image", use_column_width=True)
            else:
                st.error("Failed to estimate pose. Try another image.")

    elif option == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
        if uploaded_file is not None:
            temp_video_path = save_temp_file(uploaded_file, "mp4")
            processed_video_path = process_video(temp_video_path, votingModelroll, votingModelpitch, votingModelyaw, pca)
            st.video(processed_video_path)

if __name__ == "__main__":
    main()

