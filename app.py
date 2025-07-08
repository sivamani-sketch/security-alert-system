import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import threading
import time
import datetime
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageSequence
from werkzeug.utils import secure_filename
from gtts import gTTS
import pygame
import smtplib
from email.mime.text import MIMEText
from collections import Counter

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize pygame
pygame.mixer.init()

# Model definition
class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(ActionRecognitionModel, self).__init__()
        self.base_model = models.convnext_small(weights=None)
        in_features = self.base_model.classifier[2].in_features
        self.base_model.classifier[2] = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Flatten(1),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ActionRecognitionModel().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Constants
CLASS_NAMES = ['fighting', 'running', 'sitting', 'talking', 'walking']
location = "Main Campus Quad"

# Alert system
alert_active = True
last_alert_time = 0
alert_cooldown = 60  # seconds
current_status = "System initialized"
detection_history = []

# Initialize camera
camera = cv2.VideoCapture(0)

# Image transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        filename = os.path.join(app.config['UPLOAD_FOLDER'], f"alert_{int(time.time())}.mp3")
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return None

def play_alert_sound(message):
    try:
        filename = text_to_speech(message)
        if filename and os.path.exists(filename):
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            os.remove(filename)
    except Exception as e:
        print(f"Error playing alert sound: {e}")

def trigger_alerts(pred_class, confidence, source="live"):
    global alert_active, last_alert_time, current_status, detection_history
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if pred_class == 'fighting' and alert_active:
        current_time = time.time()
        if (current_time - last_alert_time) > alert_cooldown:
            last_alert_time = current_time
            
            detection_entry = {
                'timestamp': timestamp,
                'location': location,
                'action': pred_class,
                'confidence': f"{confidence:.1f}%",
                'source': source
            }
            detection_history.append(detection_entry)
            
            if len(detection_history) > 20:
                detection_history.pop(0)
            
            alert_message = f"Security alert! Fighting detected at {location}"
            threading.Thread(target=play_alert_sound, args=(alert_message,)).start()
            
            current_status = f"ALERT! Fighting detected at {timestamp} ({source})"
        else:
            current_status = f"Monitoring... (Alert on cooldown, last detection at {timestamp})"
    else:
        current_status = f"Monitoring... Detected: {pred_class} ({confidence:.1f}%)"

def process_image(image):
    try:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(probs, 1)
            pred_class = CLASS_NAMES[preds.item()]
            confidence = confidence.item() * 100
        
        label = f"{pred_class} ({confidence:.1f}%)"
        cv2.putText(img, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        result_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return result_image, pred_class, confidence
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, 0

def process_gif(gif_path):
    try:
        gif = Image.open(gif_path)
        frames = []
        results = []
        
        for frame in ImageSequence.Iterator(gif):
            frame = frame.convert('RGB')
            processed_frame, pred_class, confidence = process_image(frame)
            
            if processed_frame is None:
                continue
                
            results.append({
                'frame': processed_frame,
                'prediction': pred_class,
                'confidence': confidence,
                'timestamp': datetime.datetime.now().strftime("%H:%M:%S")
            })
            
            if pred_class == 'fighting':
                trigger_alerts(pred_class, confidence, "uploaded GIF")
        
        if not results:
            return None, None
        
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(gif_path))
        results[0]['frame'].save(
            output_path,
            save_all=True,
            append_images=[r['frame'] for r in results[1:]],
            loop=0,
            duration=gif.info.get('duration', 100)
        )
        
        return output_path, results
    except Exception as e:
        print(f"Error processing GIF: {e}")
        return None, None

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            continue
            
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, preds = torch.max(probs, 1)
                pred_class = CLASS_NAMES[preds.item()]
                confidence = confidence.item() * 100
                
            trigger_alerts(pred_class, confidence)
            
            label = f"Action: {pred_class} ({confidence:.1f}%)"
            status_text = f"Status: {current_status}"
            
            if "ALERT" in current_status:
                frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 255])
                cv2.putText(frame, "SECURITY ALERT!", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            cv2.putText(frame, label, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error processing frame: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html', location=location)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    return jsonify({
        'status': current_status,
        'location': location,
        'alert_active': alert_active,
        'history': detection_history[-5:] if detection_history else []
    })

@app.route('/toggle_alerts', methods=['POST'])
def toggle_alerts():
    global alert_active
    alert_active = not alert_active
    return jsonify({'alert_active': alert_active})

@app.route('/test_alert', methods=['POST'])
def test_alert():
    threading.Thread(target=play_alert_sound, args=("This is a test of the emergency alert system",)).start()
    return jsonify({'success': True})

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if filename.lower().endswith('.gif'):
                processed_path, results = process_gif(filepath)
                if not processed_path:
                    return render_template('error.html', message="Error processing GIF")
                
                return render_template('results.html',
                                     file_type='gif',
                                     original=filename,
                                     processed=os.path.basename(processed_path),
                                     results=results)
            else:
                image = Image.open(filepath)
                result_image, pred_class, confidence = process_image(image)
                if not result_image:
                    return render_template('error.html', message="Error processing image")
                
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
                result_image.save(result_path)
                
                if pred_class == 'fighting':
                    trigger_alerts(pred_class, confidence, "uploaded image")
                
                return render_template('results.html',
                                    file_type='image',
                                    original=filename,
                                    processed='processed_' + filename,
                                    prediction=pred_class,
                                    confidence=f"{confidence:.1f}%")
    
    return render_template('upload.html')

@app.route('/history')
def history():
    return render_template('history.html', history=detection_history)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)