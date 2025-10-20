import cv2
import requests
import threading
import numpy as np
from datetime import datetime
from playsound import playsound
from flask import Flask, Response, render_template, request, jsonify

# Thông tin bot Telegram
TELEGRAM_BOT_TOKEN = '7345755868:AAHfL-WJ0oCVNOSgtU3pg0G1X_W62nLiDsY'
TELEGRAM_CHAT_ID = '-4592015959'

# Khởi tạo Flask app
app = Flask(__name__)

# Trạng thái cảnh báo mặc định là tắt
alerting = False

# Biến toàn cục lưu thời gian cảnh báo
alert_start_time = None
alert_end_time = None

# Toạ độ vùng giữa hai cửa (cần tuỳ chỉnh theo video)
door_box = (150, 50, 400, 300)  # (start_x, start_y, end_x, end_y)

# Biến trạng thái cảnh báo
person_in_door_box = False

# Hàm kiểm tra thời gian cảnh báo
def is_within_alert_time():
    if not alert_start_time or not alert_end_time:
        return False

    now = datetime.now()
    current_time = now.strftime("%I:%M %p")  # Định dạng 12 giờ (AM/PM)

    # Kiểm tra logic thời gian
    if alert_start_time <= alert_end_time:  # Cảnh báo trong cùng ngày
        return alert_start_time <= current_time <= alert_end_time
    else:  # Cảnh báo qua ngày hôm sau
        return current_time >= alert_start_time or current_time <= alert_end_time

# Hàm kiểm tra giao cắt giữa hai bounding box
def check_intersection(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

# Hàm gửi ảnh Telegram
def send_telegram_photo(image, caption):
    url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto'
    _, buffer = cv2.imencode('.jpg', image)
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption}
    files = {'photo': ('image.jpg', buffer.tobytes(), 'image/jpeg')}
    response = requests.post(url, data=data, files=files)
    return response.json()

# Hàm phát âm thanh báo động trên luồng riêng
def play_alarm_sound():
    try:
        playsound('police.wav')
    except Exception as e:
        print(f"Lỗi khi phát âm thanh: {e}")

# Hàm gửi phát hiện qua Telegram
def send_detection(image, label):
    now = datetime.now()
    formatted_time = now.strftime("%H:%M:%S ngày %d-%m-%Y")
    caption = f"Phát hiện: {label} lúc {formatted_time}."
    send_telegram_photo(image, caption)

# Hàm xử lý phát hiện từ YOLO
def process_detections(image, outs, Width, Height):
    global person_in_door_box

    conf_threshold, nms_threshold = 0.5, 0.4
    class_ids, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Chỉ nhận diện đối tượng là "person" (class_id == 0)
            if class_id == 0 and confidence > conf_threshold:
                center_x, center_y = int(detection[0] * Width), int(detection[1] * Height)
                w, h = int(detection[2] * Width), int(detection[3] * Height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        person_box = (x, y, x + w, y + h)

        # Kiểm tra giao cắt với vùng cửa
        if check_intersection(door_box, person_box):
            if not person_in_door_box:
                person_in_door_box = True
                if alerting and is_within_alert_time():
                    cropped_image = image[max(0, y):y + h, max(0, x):x + w]
                    send_detection(cropped_image, "Người trong vùng cảnh báo")
                    threading.Thread(target=play_alarm_sound).start()
        else:
            person_in_door_box = False

        # Vẽ bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Hàm phát video stream
def generate_frames():
    config = 'yolov3.cfg'
    weights = 'yolov3.weights'
    classes_file = 'yolov3.txt'

    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNet(weights, config)
    cap = cv2.VideoCapture('test.mp4')  # Đường dẫn đến video

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image = cv2.resize(frame, (600, 400))
        Width, Height = image.shape[1], image.shape[0]
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward([net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()])

        process_detections(image, outs, Width, Height)

        # Vẽ vùng giữa hai cửa
        cv2.rectangle(image, (door_box[0], door_box[1]), (door_box[2], door_box[3]), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/set_alert_time', methods=['POST'])
def set_alert_time():
    global alert_start_time, alert_end_time

    data = request.json
    alert_start_time = datetime.strptime(data['start_time'], "%H:%M").strftime("%I:%M %p")
    alert_end_time = datetime.strptime(data['end_time'], "%H:%M").strftime("%I:%M %p")

    print(f"Thời gian cảnh báo được cập nhật: {alert_start_time} - {alert_end_time}")
    return jsonify({'start_time': alert_start_time, 'end_time': alert_end_time})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_alert', methods=['POST'])
def toggle_alert():
    global alerting
    data = request.get_json()
    alerting = data.get('alerting', False)  # Mặc định là tắt
    print(f"Cảnh báo {'bật' if alerting else 'tắt'}.")
    return jsonify({'status': 'success', 'alerting': alerting})

if __name__ == "__main__":
    app.run(debug=True)
