import os, sys, json, time, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from evdev import InputDevice, ecodes, list_devices

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

# ---- flip these only if something looks wrong ----
BGR_PANEL = False    # True if red and blue look swapped on screen
RB_SWAP_CAM = False  # True if the live camera picture has red/blue swapped

UW, UH = 480, 360    # UI is drawn at this size, then scaled to the panel
FS = 1.25            # text size multiplier

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
CALIB = os.path.join(HERE, 'calib.json')

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
IDX = metadata['tflite_tensor_indices']

# ---------- screen (framebuffer) ----------
def find_fb():
    for p in sorted(glob.glob('/sys/class/graphics/fb*')):
        name = open(p + '/name').read().strip().lower()
        if name.startswith('fb_') or 'ili' in name or 'st77' in name:
            return '/dev/' + os.path.basename(p), p
    return '/dev/fb1', '/sys/class/graphics/fb1'

FB_DEV, FB_SYS = find_fb()
W, H = [int(v) for v in open(FB_SYS + '/virtual_size').read().strip().split(',')]
fb = open(FB_DEV, 'wb', buffering=0)

def show(img):
    if H > W:
        img = img.rotate(90, expand=True)
    if img.size != (W, H):
        img = img.resize((W, H))
    a = np.asarray(img.convert('RGB'), dtype=np.uint16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if BGR_PANEL:
        r, b = b, r
    v = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    fb.seek(0)
    fb.write(v.astype('<u2').tobytes())

def font(size, bold=True):
    return ImageFont.truetype(FONT_B if bold else FONT_R, int(size * FS))

# ---------- touch ----------
def find_touch():
    for path in list_devices():
        d = InputDevice(path)
        if 'ads7846' in d.name.lower() or 'touch' in d.name.lower():
            return d
    raise RuntimeError('Touch device not found')

touch = find_touch()
os.set_blocking(touch.fd, False)
raw = {'x': 0, 'y': 0}

def raw_tap():
    tap = None
    try:
        for e in touch.read():
            if e.type == ecodes.EV_ABS:
                if e.code == ecodes.ABS_X:
                    raw['x'] = e.value
                elif e.code == ecodes.ABS_Y:
                    raw['y'] = e.value
            elif e.type == ecodes.EV_KEY and e.code == ecodes.BTN_TOUCH and e.value == 0:
                tap = (raw['x'], raw['y'])
    except BlockingIOError:
        pass
    return tap

def calibrate():
    pts_ui = [(50, 50), (UW - 50, 50), (UW - 50, UH - 50), (50, UH - 50)]
    pts_raw = []
    for (cx, cy) in pts_ui:
        img = Image.new('RGB', (UW, UH), (0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((UW // 2, UH // 2), "Tap the green cross", font=font(22),
               fill=(255, 255, 255), anchor="mm")
        d.line((cx - 18, cy, cx + 18, cy), fill=(0, 255, 0), width=3)
        d.line((cx, cy - 18, cx, cy + 18), fill=(0, 255, 0), width=3)
        show(img)
        time.sleep(0.5)
        raw_tap()  # drop old taps
        while True:
            t = raw_tap()
            if t:
                pts_raw.append(t)
                break
            time.sleep(0.02)
    A = np.array([[r[0], r[1], 1] for r in pts_raw], dtype=float)
    B = np.array(pts_ui, dtype=float)
    M, *_ = np.linalg.lstsq(A, B, rcond=None)
    json.dump(M.tolist(), open(CALIB, 'w'))
    return M

M = np.array(json.load(open(CALIB))) if os.path.exists(CALIB) else calibrate()

def get_tap():
    t = raw_tap()
    if not t:
        return None
    x, y = np.array([t[0], t[1], 1.0]) @ M
    return (int(x), int(y))

def inside(p, box):
    return p and box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]

# ---------- camera ----------
FIXED = sys.argv[1] if len(sys.argv) > 1 else None
if FIXED is None:
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"}))
    cam.start()
    time.sleep(1.5)

def grab():
    if FIXED:
        return Image.open(FIXED).convert('RGB')
    arr = cam.capture_array()
    if RB_SWAP_CAM:
        arr = arr[:, :, ::-1]
    return Image.fromarray(np.ascontiguousarray(arr))

# ---------- model ----------
interpreter = Interpreter(model_path=os.path.join(HERE, 'model.tflite'))
interpreter.allocate_tensors()

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

def predict(img):
    norm_img, feats = compute_colour_features(img)
    fs = (feats * COL_SCALE + COL_MIN).astype(np.float32)
    interpreter.set_tensor(IDX['image_input'], norm_img[None])
    interpreter.set_tensor(IDX['colour_input'], fs[None])
    interpreter.invoke()
    dis = interpreter.get_tensor(IDX['disease_output'])[0]
    sev = interpreter.get_tensor(IDX['severity_output'])[0]
    chl = interpreter.get_tensor(IDX['chlorophyll_output'])[0]
    disease = CLASSES[int(np.argmax(dis))]
    severity = SEV_LEVELS[int(np.argmax(sev))]
    chl_cci = float(((chl - CHL_MIN) / CHL_SCALE)[0])
    print(disease, float(np.max(dis)), severity, chl_cci)
    return disease, float(np.max(dis)), severity, chl_cci

# ---------- screens ----------
BTN_LIVE = (UW // 2 - 44, UH - 92, UW // 2 + 44, UH - 4)
BTN_AGAIN = (UW // 2 - 110, UH - 60, UW // 2 + 110, UH - 10)
WHITE = (240, 244, 241)
MUTED = (140, 160, 148)

def shade(img, box, color, alpha):
    ov = Image.new('RGBA', img.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).rectangle(box, fill=color + (alpha,))
    return Image.alpha_composite(img.convert('RGBA'), ov).convert('RGB')

def header(img, title):
    img = shade(img, (0, 0, UW, 44), (10, 60, 30), 215)
    d = ImageDraw.Draw(img)
    d.text((14, 22), "LEAF SCANNER", font=font(15), fill=WHITE, anchor="lm")
    d.text((UW - 14, 22), title, font=font(13, False), fill=(190, 230, 200), anchor="rm")
    return img

def live_screen(frame):
    img = header(frame.resize((UW, UH)), "LIVE")
    d = ImageDraw.Draw(img)
    d.ellipse((UW - 73, 17, UW - 63, 27), fill=(255, 60, 60))
    for x, y, dx, dy in [(40, 62, 1, 1), (UW - 40, 62, -1, 1),
                         (40, UH - 104, 1, -1), (UW - 40, UH - 104, -1, -1)]:
        d.line((x, y, x + dx * 30, y), fill=WHITE, width=3)
        d.line((x, y, x, y + dy * 30), fill=WHITE, width=3)
    cx, cy = UW // 2, UH - 48
    d.ellipse((cx - 38, cy - 38, cx + 38, cy + 38), outline=WHITE, width=4)
    d.ellipse((cx - 29, cy - 29, cx + 29, cy + 29), fill=(60, 200, 100))
    show(img)

def busy_screen(frame):
    img = shade(frame.resize((UW, UH)), (0, 0, UW, UH), (0, 0, 0), 170)
    img = header(img, "ANALYZING")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((UW // 2 - 150, UH // 2 - 50, UW // 2 + 150, UH // 2 + 50),
                        radius=18, fill=(20, 40, 28), outline=(60, 200, 100), width=3)
    d.text((UW // 2, UH // 2 - 12), "Analyzing leaf...", font=font(20), fill=WHITE, anchor="mm")
    d.text((UW // 2, UH // 2 + 24), "please wait", font=font(13, False), fill=MUTED, anchor="mm")
    show(img)

def pill(d, box, text, fill, fg):
    d.rounded_rectangle(box, radius=10, fill=fill)
    d.text(((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), text,
           font=font(12), fill=fg, anchor="mm")

def result_screen(photo, disease, conf, sev, chl):
    img = Image.new('RGB', (UW, UH))
    bg = ImageDraw.Draw(img)
    for y in range(UH):
        t = y / UH
        bg.line((0, y, UW, y), fill=(int(10 + 12 * t), int(22 + 20 * t), int(16 + 12 * t)))
    img = header(img, "RESULT")
    d = ImageDraw.Draw(img)
    ph = photo.resize((200, 150)).convert('RGBA')
    m = Image.new('L', ph.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, 200, 150), radius=14, fill=255)
    ph.putalpha(m)
    img.paste(ph, (14, 58), ph)
    col = (90, 220, 120) if disease.lower() == 'healthy' else (255, 120, 80)
    d.text((14, 224), "Confidence", font=font(12, False), fill=MUTED, anchor="lm")
    d.text((214, 224), f"{conf * 100:.0f}%", font=font(14), fill=WHITE, anchor="rm")
    d.rounded_rectangle((14, 238, 214, 250), radius=6, outline=MUTED)
    d.rounded_rectangle((14, 238, 14 + max(8, int(200 * conf)), 250), radius=6, fill=col)
    x0 = 238
    d.text((x0, 62), disease.upper(), font=font(24), fill=col, anchor="lt")
    d.text((x0, 112), "Severity", font=font(12, False), fill=MUTED, anchor="lm")
    cols = [(90, 200, 110), (255, 190, 60), (255, 90, 80)]
    for i, n in enumerate(["Low", "Medium", "High"]):
        bx = x0 + i * 76
        on = (i == sev)
        pill(d, (bx, 126, bx + 70, 156), n,
             cols[i] if on else (45, 60, 52), (15, 20, 16) if on else MUTED)
    d.text((x0, 182), "Chlorophyll (CCI)", font=font(12, False), fill=MUTED, anchor="lm")
    d.text((x0, 198), f"{chl:.1f}", font=font(26), fill=(120, 205, 255), anchor="lt")
    d.rounded_rectangle((x0, 248, UW - 14, 260), radius=6, outline=MUTED)
    fw = int((UW - 14 - x0) * min(max(chl, 0), 100) / 100)
    d.rounded_rectangle((x0, 248, x0 + max(8, fw), 260), radius=6, fill=(120, 205, 255))
    d.rounded_rectangle(BTN_AGAIN, radius=14, fill=(40, 120, 210))
    d.text(((BTN_AGAIN[0] + BTN_AGAIN[2]) // 2, (BTN_AGAIN[1] + BTN_AGAIN[3]) // 2),
           "SCAN AGAIN", font=font(18), fill=WHITE, anchor="mm")
    show(img)

# ---------- main loop ----------
state = 'live'
try:
    while True:
        tap = get_tap()
        if state == 'live':
            frame = grab()
            live_screen(frame)
            if inside(tap, BTN_LIVE):
                busy_screen(frame)
                res = predict(frame)
                result_screen(frame, *res)
                get_tap()  # drop taps made while analyzing
                state = 'result'
        else:
            if inside(tap, BTN_AGAIN):
                state = 'live'
        time.sleep(0.02)
except KeyboardInterrupt:
    pass