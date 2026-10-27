import os, json, time, sys
import numpy as np
from PIL import Image, ImageFont
import cv2
import RPi.GPIO as GPIO

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

from picamera2 import Picamera2
from luma.core.interface.serial import i2c
from luma.oled.device import sh1106
from luma.core.render import canvas

# ---------- CONFIG ----------
BUTTON_PIN = 17
HERE = os.path.dirname(os.path.abspath(__file__))

metadata = json.load(open(os.path.join(HERE, 'model_metadata.json')))
scaler_params = json.load(open(os.path.join(HERE, 'scaler_params.json')))

CLASSES = metadata['classes']
SEV_LEVELS = metadata['severity_levels']
IMG_HEIGHT = metadata['img_height']
IMG_WIDTH = metadata['img_width']

CHL_SCALE = np.array(scaler_params['chlorophyll']['scale_'], dtype=np.float32)
CHL_MIN = np.array(scaler_params['chlorophyll']['min_'], dtype=np.float32)
COL_SCALE = np.array(scaler_params['colour_features']['scale_'], dtype=np.float32)
COL_MIN = np.array(scaler_params['colour_features']['min_'], dtype=np.float32)
TFLITE_IDX = metadata['tflite_tensor_indices']

# ---------- SETUP ----------
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

serial = i2c(port=1, address=0x3C)
device = sh1106(serial, width=128, height=64)

FIXED_IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else None

if FIXED_IMAGE_PATH is None:
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (1024, 768)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
else:
    picam2 = None

interpreter = Interpreter(model_path=os.path.join(HERE, 'model.tflite'))
interpreter.allocate_tensors()

def minmax_transform(x, scale_, min_):
    return x * scale_ + min_

def minmax_inverse(x, scale_, min_):
    return (x - min_) / scale_

def compute_colour_features(pil_img):
    img = pil_img.convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32)
    norm = arr / 255.0
    u8 = arr.astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    mask = (gray > 30) & (gray < 240)
    mm = lambda ch: (ch[mask].mean() if mask.any() else 0.0)

    R, G, B = norm[:, :, 0], norm[:, :, 1], norm[:, :, 2]
    eps = 1e-6
    exg = 2 * G - R - B
    vari = np.clip((G - R) / (G + R - B + eps), -3, 3)
    gli = np.clip((2 * G - R - B) / (2 * G + R + B + eps), -1, 1)
    mgc = G / (R + G + B + eps)

    feats = [mm(norm[:, :, 0]), mm(norm[:, :, 1]), mm(norm[:, :, 2]),
             mm(hsv[:, :, 0]) / 179, mm(hsv[:, :, 1]) / 255, mm(hsv[:, :, 2]) / 255,
             mm(lab[:, :, 1]) / 255, mm(lab[:, :, 0]) / 255,
             (norm[:, :, 1] - norm[:, :, 2]).mean(), (norm[:, :, 1] - norm[:, :, 0]).mean(),
             mm(exg), mm(vari), mm(gli), mm(mgc)]
    return norm.astype(np.float32), np.array(feats, dtype=np.float32)

def show_ready():
    with canvas(device) as draw:
        draw.rectangle(device.bounding_box, outline="white", fill="black")
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        sub_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        w = draw.textlength("LEAF SCANNER", font=title_font)
        draw.text(((128 - w) / 2, 3), "LEAF SCANNER", font=title_font, fill="white")
        draw.line((0, 20, 128, 20), fill="white")
        draw.ellipse((8, 28, 28, 48), outline="white")
        draw.line((18, 38, 18, 48), fill="white")
        draw.text((36, 28), "Press button", font=sub_font, fill="white")
        draw.text((36, 42), "to scan leaf", font=sub_font, fill="white")

def show_message(line1, line2=""):
    with canvas(device) as draw:
        draw.rectangle(device.bounding_box, outline="white", fill="black")
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        w = draw.textlength(line1, font=font)
        draw.text(((128 - w) / 2, 22), line1, font=font, fill="white")
        if line2:
            font2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
            w2 = draw.textlength(line2, font=font2)
            draw.text(((128 - w2) / 2, 40), line2, font=font2, fill="white")

def show_result(disease, dis_conf, severity, chl):
    with canvas(device) as draw:
        draw.rectangle(device.bounding_box, outline="white", fill="black")
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)

        draw.text((2, 0), disease.upper(), font=title_font, fill="white")
        draw.text((90, 2), f"{dis_conf*100:.0f}%", font=body_font, fill="white")
        draw.line((0, 16, 128, 16), fill="white")

        sev_labels = ["Low", "Medium", "High"]
        sev_text = sev_labels[severity] if severity < len(sev_labels) else str(severity)
        draw.text((2, 20), f"Severity: {sev_text}", font=body_font, fill="white")

        draw.text((2, 34), f"Chlorophyll:", font=body_font, fill="white")
        draw.text((2, 46), f"{chl:.1f} CCI", font=title_font, fill="white")

        bar_x = 70
        bar_w = 50
        fill_w = int(min(max(chl, 0), 100) / 100 * bar_w)
        draw.rectangle((bar_x, 48, bar_x + bar_w, 56), outline="white")
        draw.rectangle((bar_x, 48, bar_x + fill_w, 56), fill="white")

show_ready()
print("Ready. Press the button to capture (Ctrl+C to quit)...")

try:
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            show_message("Capturing...")
            print("\nCapturing image...")

            if FIXED_IMAGE_PATH:
                save_path = FIXED_IMAGE_PATH
            else:
                save_path = '/tmp/leaf_capture.jpg'
                picam2.capture_file(save_path)

            img = Image.open(save_path)

            show_message("Analyzing...")
            print("Processing...")
            norm_img, feats = compute_colour_features(img)
            feats_scaled = minmax_transform(feats, COL_SCALE, COL_MIN).astype(np.float32)

            interpreter.set_tensor(TFLITE_IDX['image_input'], norm_img[None])
            interpreter.set_tensor(TFLITE_IDX['colour_input'], feats_scaled[None])
            interpreter.invoke()

            dis_probs = interpreter.get_tensor(TFLITE_IDX['disease_output'])[0]
            sev_probs = interpreter.get_tensor(TFLITE_IDX['severity_output'])[0]
            chl_scaled = interpreter.get_tensor(TFLITE_IDX['chlorophyll_output'])[0]

            disease = CLASSES[int(np.argmax(dis_probs))]
            dis_conf = float(np.max(dis_probs))
            severity = SEV_LEVELS[int(np.argmax(sev_probs))]
            chl_cci = float(minmax_inverse(chl_scaled, CHL_SCALE, CHL_MIN)[0])

            print(f"DISEASE: {disease} ({dis_conf*100:.1f}%)")
            print(f"SEVERITY: {severity}")
            print(f"CHLOROPHYLL: {chl_cci:.2f} CCI")

            show_result(disease, dis_conf, severity, chl_cci)
            time.sleep(10)
            show_ready()
            print("\nReady. Press the button to capture...")
            time.sleep(0.5)

        time.sleep(0.05)

except KeyboardInterrupt:
    GPIO.cleanup()
    print("\nExiting...")