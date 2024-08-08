import cv2
import time
import os
from datetime import datetime
import face_recognition
import pickle
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
from collections import Counter
import threading
import subprocess
DEFAULT_ENCODINGS_PATH = Path("output/recog_train_dir01_location6_cnn.pkl")
cap = cv2.VideoCapture(0)
def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding, tolerance=0.5
    )
    votes = Counter(
        name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match
    )
    if votes:
        most_common = votes.most_common(1)
        most_common_name = most_common[0][0]
        return most_common_name
    else:
        return "Unknown"
def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
    threshold: float = 0.6,
    temp_image_path: Path = Path("./temp_marked_image.jpg")
) -> str:
    
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
    min_face_size = (30, 30)  # Minimum face size
    name = "Unknown"  # Initialize name as Unknown
    input_image = face_recognition.load_image_file(image_location)
    input_face_locations = face_recognition.face_locations(input_image, model=model, number_of_times_to_upsample=1)
    face_landmarks = face_recognition.face_landmarks(input_image, input_face_locations, model="small")
    if not input_face_locations or not face_landmarks:
        name = "No_face"
        return name
    for i, input_face_location in enumerate(input_face_locations):
        top, right, bottom, left = input_face_location
        face_width = right - left
        face_height = bottom - top
        if face_width < min_face_size[0] or face_height < min_face_size[1]:
            name = "No_face"
            return name
    
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations, model="large", num_jitters=1)
    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if name != "Unknown":
            distances = face_recognition.face_distance(loaded_encodings["encodings"], unknown_encoding)
            min_distance = min(distances)
            if min_distance > threshold:
                name = "Unknown"
        top, right, bottom, left = bounding_box
        cv2.rectangle(input_image, (left, top), (right, bottom), (0, 0, 255), 3)
        cv2.putText(input_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
    # Convert the image from RGB (used by face_recognition) to BGR (used by OpenCV)
    input_image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # Save the marked image to a temporary file
    cv2.imwrite(str(temp_image_path), input_image_bgr)
    
    return name
def process_files(model: str = "hog", datapath: str = "./date_data", temp_image_path: Path = Path("./temp_marked_image.jpg")):
    while True:
        # Step 1: Create corresponding directories in process_data
        for folderpath in Path(datapath).rglob("*"):
            if folderpath.is_dir():
                # Create the corresponding folder in process_data
                relative_path = folderpath.relative_to(datapath)
                process_folder = Path("process_data") / relative_path
                process_folder.mkdir(parents=True, exist_ok=True)
                #print(f"Created folder: {process_folder}")  # Debug output

        # Step 2: Process files for face recognition and copying
        for filepath in Path(datapath).rglob("*"):
            if filepath.is_file():
                #print(f"Processing file: {filepath}")  # Debug output

                name = recognize_faces(image_location=str(filepath.absolute()), model=model, temp_image_path=temp_image_path)
                print(f"Recognition result for: {name}")  # Debug output

                # Only save the image if the name is "Unknown"
                if name == "Unknown":
                    # Construct the destination path in process_data folder
                    relative_path = filepath.relative_to(datapath)
                    dest_path = Path("process_data") / relative_path

                    # Save the marked image to the new folder
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(temp_image_path, dest_path)
                    print(f"Saved marked image")  # Debug output

                # Ensure the file is removed from the original location
                try:
                    os.remove(filepath)
                    #print(f"Deleted original file: {filepath}")  # Debug output
                except Exception as e:
                    print(f"Error deleting {filepath}: {e}")

        print("Waiting for the next check...")
        time.sleep(3)  # Check every 10 seconds
def extract_faces_from_frame(input_frame):
    timestamp = time.strftime("%H-%M-%S")  # 照片檔的時間標記，小時、分鐘、秒
    hour_folder = time.strftime("%H")
    folder_path = f'./date_data/{hour_folder}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    image_filename = f'./date_data/{hour_folder}/unknown_{timestamp}.jpg'
    cv2.imwrite(image_filename, input_frame)
def main():
    fps = 5  # 每秒幀数
    interval = 1 / fps  # 每幀的時間間隔（秒）
    #cap = cv2.VideoCapture(0)  # 打開鏡頭
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("無法打開鏡頭")
        return
    try:
        while True:
            start_time_main = datetime.now()

            ret, frame = cap.read()
            if not ret:
                print("無法讀取")
                break

            # 显示摄像头捕捉的画面
            cv2.imshow('Camera', frame)
            extract_faces_from_frame(frame)
            elapsed_time = (datetime.now() - start_time_main).total_seconds()
            time_to_wait = max(0, interval - elapsed_time)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 每次循环检查一次按键
                break
            time.sleep(time_to_wait)
    except KeyboardInterrupt:
        print("中斷程序")
    finally:
        cap.release()
        cv2.destroyAllWindows()
def record_video():
    #cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Error: Could not open camera video.")
        return
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    start_time_video = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if out is None or time.time() - start_time_video > 300:  # 每5分鐘創建一个新的影片
            if out is not None:
                out.release()  # 釋放之前的VideoWriter
            timestamp = time.strftime("%H-%M")
            filename = f"./video/{timestamp}.avi"
            out = cv2.VideoWriter(filename, fourcc, 30.0, (frame.shape[1], frame.shape[0]))  # 確保影像尺寸相符
            if not out.isOpened():
                print("Error: Could not open VideoWriter.")
                return
            out.write(frame)  # 立即寫入第一幀
            start_time_video = time.time()
        out.write(frame)
    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    
def convert_avi_mp4(input_file, output_file):
    cmd = ['ffmpeg', '-i', input_file, '-c:v', 'libx264', '-preset', 'slow', '-crf', '22', '-c:a', 'copy', output_file]
    subprocess.run(cmd)

def process_videos(input_folder, output_folder):
    try:
        while True:
            image_files = os.listdir(input_folder)
            video_files = [f for f in image_files if f.endswith('.avi')]

            if len(video_files) == 1:
                print("Only one video found. Waiting for more videos to be added...")
                time.sleep(60)
                continue
            
            if len(video_files) >= 2:
                for video_file in video_files[:-1]:  # 只處理到倒數第二個影片
                    input_file_path = os.path.join(input_folder, video_file)
                    output_file_path = os.path.join(output_folder, os.path.splitext(video_file)[0] + '.mp4')

                    convert_avi_mp4(input_file_path, output_file_path)
                    print(f"Converted and saved: {output_file_path}")

                    # 刪除原始檔案
                    os.remove(input_file_path)
                    print(f"Deleted original file: {input_file_path}")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting gracefully...")
        pass   
def copy_images_to_folder(source_folder: str, target_folder: str, image_extensions: list = ["jpg", "jpeg", "png", "gif"]):
    """
    複製指定資料夾中的所有圖片到另一個資料夾
    
    :param source_folder: 原始資料夾路徑
    :param target_folder: 目標資料夾路徑
    :param image_extensions: 圖片的擴展名列表
    """
    # 將路徑轉換為 Path 物件
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)

    # 如果目標資料夾不存在，創建它
    target_folder.mkdir(parents=True, exist_ok=True)

    # 遍歷原始資料夾中的所有文件
    for filepath in source_folder.rglob("*"):
        if filepath.is_file() and filepath.suffix[1:].lower() in image_extensions:
            try:
                # 複製文件到目標資料夾
                shutil.copy(filepath, target_folder / filepath.name)
                #print(f"複製文件: {filepath} 到 {target_folder / filepath.name}")
            except Exception as e:
                print(f"複製 {filepath} 時發生錯誤: {e}")

if __name__ == "__main__":
    video_folder='./video'
    deal_video_folder='./MP4'
    process_picture = threading.Thread(target=process_files)
    photo_picture = threading.Thread(target=main)
    video_Recode=threading.Thread(target=record_video)
    video_deal=threading.Thread(target=process_videos,args=(video_folder,deal_video_folder))  
    process_picture.start()
    photo_picture.start()
    video_Recode.start()
    video_deal.start()
    
    
    process_picture.join()
    photo_picture.join()
    video_Recode.join()
    video_deal.join()
    