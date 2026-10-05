"""
Campus ALPR Parking Control Center
----------------------------------
- Live license plate reading (EasyOCR in a background thread, so video never lags)
- Live free-slot counter + occupancy bar + recent activity feed (dashboard panel)
- Indian plate validation + OCR error correction (O/0, I/1, S/5, B/8 ...)
- Multi-frame voting + cooldown (no more fake entries from OCR noise)
- Smart AUTO mode: plate not inside -> ENTRY, plate already inside -> EXIT
- Logs to CSV on your Desktop

Install:  pip install easyocr opencv-python numpy

Keys:  q = quit | m = cycle mode (AUTO/ENTRY/EXIT) | e / x = test entry / exit
"""

import csv
import os
import queue
import re
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
GATE_ID = 'PES_RR_MAIN_GATE'
TOTAL_CAPACITY = 100
BASE_OCCUPANCY = 42          # vehicles already inside at start (unknown plates)
CAMERA_INDEX = 0

# Scan zone as fractions of the frame (x1, y1, x2, y2). OCR only reads inside it.
# Place the plate inside this box -> faster + fewer false reads.
SCAN_ZONE = (0.12, 0.28, 0.88, 0.85)

OCR_INTERVAL = 0.25          # seconds between OCR requests
MIN_CONFIDENCE = 0.35        # minimum average OCR confidence
VOTES_NEEDED = 2             # same plate must be read this many times...
VOTE_WINDOW = 3.0            # ...within this many seconds
COOLDOWN = 15.0              # seconds before the same plate can trigger again

VIEW_W, VIEW_H = 960, 540    # video size on screen
PANEL_W = 380                # dashboard width

desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
CSV_LOG_FILE = os.path.join(desktop_path, 'campus_parking_log.csv')

# ==========================================
# COLORS (BGR)
# ==========================================
BG = (32, 26, 24)
CARD = (52, 44, 40)
WHITE = (240, 240, 240)
GREY = (150, 145, 140)
GREEN = (110, 210, 90)
AMBER = (0, 190, 255)
RED = (70, 70, 240)
BLUE = (255, 170, 60)
CYAN = (230, 200, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ==========================================
# PLATE VALIDATION (Indian format)
# ==========================================
STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'DN', 'GA', 'GJ',
    'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP',
    'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK',
    'UP', 'WB',
}
PLATE_RE = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$')   # KA51AB1234
BH_RE = re.compile(r'^\d{2}BH\d{4}[A-Z]{1,2}$')              # 22BH1234AA

# L = letter position, D = digit position
TEMPLATES = {
    10: ['LLDDLLDDDD'],
    9: ['LLDDLDDDD', 'LLDLLDDDD'],
    11: ['LLDDLLLDDDD'],
}
DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z', '6': 'G', '4': 'A'}
LETTER_TO_DIGIT = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2',
                   'S': '5', 'B': '8', 'G': '6', 'A': '4', 'T': '7'}
ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def sanitize_plate_text(raw_text):
  """KA-51-XX-1234 -> KA51XX1234"""
  return re.sub(r'[^A-Z0-9]', '', raw_text.upper())


def normalize_plate(raw_text):
  """Cleans text, fixes position-based OCR mistakes, returns valid plate or None."""
  clean = sanitize_plate_text(raw_text)
  if BH_RE.match(clean):
    return clean
  for tpl in TEMPLATES.get(len(clean), []):
    chars = []
    for ch, kind in zip(clean, tpl):
      if kind == 'L':
        chars.append(DIGIT_TO_LETTER.get(ch, ch) if ch.isdigit() else ch)
      else:
        chars.append(LETTER_TO_DIGIT.get(ch, ch) if ch.isalpha() else ch)
    cand = ''.join(chars)
    if PLATE_RE.match(cand) and cand[:2] in STATE_CODES:
      return cand
  return None


# ==========================================
# BACKGROUND OCR WORKER (keeps video smooth)
# ==========================================
class OCRWorker(threading.Thread):

  def __init__(self, reader):
    super().__init__(daemon=True)
    self.reader = reader
    self.in_q = queue.Queue(maxsize=1)
    self.out_q = queue.Queue()
    self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

  def submit(self, roi, offset):
    try:
      self.in_q.put_nowait((roi, offset))
      return True
    except queue.Full:
      return False

  def run(self):
    while True:
      roi, (ox, oy) = self.in_q.get()
      try:
        self.out_q.put(self._read(roi, ox, oy))
      except Exception as exc:  # never let the thread die
        print(f'[OCR ERROR] {exc}')
        self.out_q.put(None)

  def _read(self, roi, ox, oy):
    gray = self.clahe.apply(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
    dets = self.reader.readtext(gray, allowlist=ALLOWLIST)
    if not dets:
      return None

    # reading order: top-to-bottom, left-to-right (handles 2-row plates too)
    dets.sort(key=lambda d: (round(d[0][0][1] / 25), d[0][0][0]))
    dets = dets[:8]

    best = None
    for i in range(len(dets)):
      for j in range(i, len(dets)):
        seg = dets[i:j + 1]
        plate = normalize_plate(''.join(d[1] for d in seg))
        if not plate:
          continue
        conf = sum(d[2] for d in seg) / len(seg)
        if conf < MIN_CONFIDENCE:
          continue
        pts = np.array([p for d in seg for p in d[0]])
        box = (int(pts[:, 0].min()) + ox, int(pts[:, 1].min()) + oy,
               int(pts[:, 0].max()) + ox, int(pts[:, 1].max()) + oy)
        if best is None or conf > best[1]:
          best = (plate, conf, box)
    return best


# ==========================================
# PARKING STATE + CSV LOGGING
# ==========================================
class ParkingState:

  def __init__(self):
    self.inside = set()
    self.base_unknown = BASE_OCCUPANCY
    self.events = deque(maxlen=7)       # (time, plate, action)
    self.banner = ('', WHITE, 0.0)      # text, color, show-until
    self.last_plate = ('---', 0.0)      # plate, confidence
    self._init_csv()
    self._restore_from_csv()

  @property
  def occupancy(self):
    return self.base_unknown + len(self.inside)

  @property
  def free(self):
    return max(0, TOTAL_CAPACITY - self.occupancy)

  def _init_csv(self):
    if not os.path.exists(CSV_LOG_FILE):
      with open(CSV_LOG_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(['Timestamp', 'Plate_Number', 'Gate_ID',
                                'Action_Type', 'Occupancy_Status', 'Confidence'])

  def _restore_from_csv(self):
    """Rebuild who is inside from previous runs, so restarts don't lose state."""
    try:
      with open(CSV_LOG_FILE, newline='') as f:
        for row in csv.DictReader(f):
          plate, action = row.get('Plate_Number', ''), row.get('Action_Type', '')
          if action == 'ENTRY':
            self.inside.add(plate)
          elif action == 'EXIT':
            self.inside.discard(plate)
    except Exception as exc:
      print(f'[WARN] Could not restore state: {exc}')
    if self.occupancy > TOTAL_CAPACITY:
      self.base_unknown = max(0, TOTAL_CAPACITY - len(self.inside))

  def _log(self, plate, action, conf):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    occ = f'{self.occupancy}/{TOTAL_CAPACITY}'
    with open(CSV_LOG_FILE, 'a', newline='') as f:
      csv.writer(f).writerow([ts, plate, GATE_ID, action, occ, f'{conf * 100:.1f}%'])
    self.events.appendleft((datetime.now().strftime('%H:%M:%S'), plate, action))
    print(f'[AUDIT LOG] {ts} | Plate: {plate} | Action: {action} | Slots: {occ}')

  def _flash(self, text, color, secs=2.5):
    self.banner = (text, color, time.time() + secs)

  def handle(self, plate, mode, conf):
    """Decide ENTRY / EXIT / DENIED and update counters."""
    self.last_plate = (plate, conf)

    if mode == 'AUTO':
      action = 'EXIT' if plate in self.inside else 'ENTRY'
    else:
      action = mode

    if action == 'ENTRY':
      if plate in self.inside:
        self._flash(f'{plate} ALREADY INSIDE', AMBER)
        return
      if self.occupancy >= TOTAL_CAPACITY:
        self._log(plate, 'DENIED_FULL', conf)
        self._flash(f'{plate}  PARKING FULL', RED)
        return
      self.inside.add(plate)
      self._log(plate, 'ENTRY', conf)
      self._flash(f'{plate}  ENTRY GRANTED', GREEN)
    else:
      if plate in self.inside:
        self.inside.discard(plate)
      elif self.base_unknown > 0:
        self.base_unknown -= 1
      else:
        self._flash(f'{plate} NOT FOUND INSIDE', AMBER)
        return
      self._log(plate, 'EXIT', conf)
      self._flash(f'{plate}  EXIT LOGGED', BLUE)


# ==========================================
# UI DRAWING
# ==========================================
def put(img, text, org, scale=0.55, color=WHITE, thick=1):
  cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def centered(img, text, cx, y, scale, color, thick):
  (w, _), _ = cv2.getTextSize(text, FONT, scale, thick)
  put(img, text, (cx - w // 2, y), scale, color, thick)


def chip(img, text, x, y, color):
  (w, h), _ = cv2.getTextSize(text, FONT, 0.5, 1)
  cv2.rectangle(img, (x, y), (x + w + 20, y + 26), color, -1)
  put(img, text, (x + 10, y + 18), 0.5, (20, 20, 20), 1)
  return x + w + 30


def free_color(state):
  ratio = state.free / TOTAL_CAPACITY
  if state.free == 0:
    return RED
  return GREEN if ratio > 0.25 else AMBER


def draw_panel(state, mode, fps):
  p = np.full((VIEW_H, PANEL_W, 3), BG, dtype=np.uint8)
  cx = PANEL_W // 2

  # Header
  put(p, 'CAMPUS PARKING', (20, 32), 0.8, WHITE, 2)
  put(p, GATE_ID, (20, 52), 0.42, GREY, 1)

  # Big free-slots number
  col = free_color(state)
  cv2.rectangle(p, (15, 66), (PANEL_W - 15, 186), CARD, -1)
  centered(p, str(state.free), cx, 150, 3.2, col, 6)
  centered(p, 'FREE SLOTS', cx, 176, 0.55, GREY, 1)

  # Occupancy bar
  bx1, bx2, by = 20, PANEL_W - 20, 202
  cv2.rectangle(p, (bx1, by), (bx2, by + 16), CARD, -1)
  fill = int((bx2 - bx1) * state.occupancy / TOTAL_CAPACITY)
  cv2.rectangle(p, (bx1, by), (bx1 + min(fill, bx2 - bx1), by + 16), col, -1)
  put(p, f'{state.occupancy}/{TOTAL_CAPACITY} occupied', (bx1, by + 36), 0.48, GREY, 1)

  # Status chips
  x = chip(p, 'OPEN' if state.free > 0 else 'FULL', 20, 250, col)
  mode_col = {'AUTO': CYAN, 'ENTRY': GREEN, 'EXIT': BLUE}[mode]
  chip(p, f'MODE: {mode}', x, 250, mode_col)

  # Last plate card
  cv2.rectangle(p, (15, 288), (PANEL_W - 15, 358), CARD, -1)
  put(p, 'LAST PLATE READ', (25, 308), 0.42, GREY, 1)
  plate, conf = state.last_plate
  put(p, plate, (25, 345), 0.95, WHITE, 2)
  if conf > 0:
    put(p, f'{conf * 100:.0f}%', (PANEL_W - 70, 345), 0.55, GREY, 1)

  # Recent activity
  put(p, 'RECENT ACTIVITY', (20, 382), 0.45, GREY, 1)
  cv2.line(p, (20, 390), (PANEL_W - 20, 390), CARD, 1)
  colors = {'ENTRY': GREEN, 'EXIT': BLUE, 'DENIED_FULL': RED}
  if not state.events:
    put(p, 'Waiting for vehicles...', (20, 415), 0.5, GREY, 1)
  for i, (t, plate, action) in enumerate(state.events):
    y = 411 + i * 17
    put(p, t, (20, y), 0.42, GREY, 1)
    put(p, plate, (95, y), 0.45, WHITE, 1)
    put(p, action.replace('_FULL', ''), (255, y), 0.42, colors.get(action, WHITE), 1)

  # Footer
  put(p, f'{fps:4.1f} FPS   |   q quit  m mode  e/x test', (20, VIEW_H - 10), 0.4, GREY, 1)
  return p


def draw_scan_zone(view, active):
  x1, y1 = int(SCAN_ZONE[0] * VIEW_W), int(SCAN_ZONE[1] * VIEW_H)
  x2, y2 = int(SCAN_ZONE[2] * VIEW_W), int(SCAN_ZONE[3] * VIEW_H)
  col = CYAN if active else GREY
  L = 28
  for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
    cv2.line(view, (px, py), (px + dx * L, py), col, 3)
    cv2.line(view, (px, py), (px, py + dy * L), col, 3)
  put(view, 'SCAN ZONE - align plate here', (x1 + 6, y1 - 8), 0.5, col, 1)


def draw_banner(view, state):
  text, color, until = state.banner
  if time.time() > until:
    return
  overlay = view.copy()
  cv2.rectangle(overlay, (0, 0), (VIEW_W, 56), color, -1)
  cv2.addWeighted(overlay, 0.75, view, 0.25, 0, view)
  centered(view, text, VIEW_W // 2, 38, 0.95, (15, 15, 15), 2)


# ==========================================
# MAIN LOOP
# ==========================================
def run_alpr_system():
  try:
    import easyocr
  except ImportError:
    print('[ERROR] EasyOCR not installed. Run: pip install easyocr')
    return

  print('[SYSTEM] Loading EasyOCR (first run downloads models)...')
  worker = OCRWorker(easyocr.Reader(['en'], gpu=False))
  worker.start()
  print('[SYSTEM] EasyOCR ready.')

  state = ParkingState()
  cap = cv2.VideoCapture(CAMERA_INDEX)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  if not cap.isOpened():
    print('[ERROR] Cannot open camera. Try CAMERA_INDEX = 1.')
    return

  print('=== ALPR LIVE CAMERA GATE SYSTEM STARTED ===')
  print(f'Logging to: {CSV_LOG_FILE}')

  modes = ['AUTO', 'ENTRY', 'EXIT']
  mode = 'AUTO'
  votes = {}               # plate -> deque of read timestamps
  last_event = {}          # plate -> time of last triggered event
  detection = None         # (plate, conf, box, time) for drawing
  last_submit = 0.0
  fps, t_prev = 0.0, time.time()

  while True:
    ret, frame = cap.read()
    if not ret:
      break
    now = time.time()
    h0, w0 = frame.shape[:2]
    sx, sy = VIEW_W / w0, VIEW_H / h0

    # ---- send ROI to OCR thread (non-blocking) ----
    if now - last_submit >= OCR_INTERVAL:
      rx1, ry1 = int(SCAN_ZONE[0] * w0), int(SCAN_ZONE[1] * h0)
      rx2, ry2 = int(SCAN_ZONE[2] * w0), int(SCAN_ZONE[3] * h0)
      if worker.submit(frame[ry1:ry2, rx1:rx2].copy(), (rx1, ry1)):
        last_submit = now

    # ---- collect OCR results ----
    while not worker.out_q.empty():
      res = worker.out_q.get()
      if not res:
        continue
      plate, conf, box = res
      detection = (plate, conf, box, now)

      dq = votes.setdefault(plate, deque())
      dq.append(now)
      while dq and now - dq[0] > VOTE_WINDOW:
        dq.popleft()

      if len(dq) >= VOTES_NEEDED and now - last_event.get(plate, 0) > COOLDOWN:
        last_event[plate] = now
        dq.clear()
        state.handle(plate, mode, conf)

    # ---- draw video side ----
    view = cv2.resize(frame, (VIEW_W, VIEW_H))
    fresh = detection is not None and now - detection[3] < 1.2
    draw_scan_zone(view, fresh)
    if fresh:
      plate, conf, (bx1, by1, bx2, by2), _ = detection
      p1, p2 = (int(bx1 * sx), int(by1 * sy)), (int(bx2 * sx), int(by2 * sy))
      cv2.rectangle(view, p1, p2, GREEN, 2)
      label = f'{plate}  {conf * 100:.0f}%'
      (tw, th), _ = cv2.getTextSize(label, FONT, 0.7, 2)
      ty = max(p1[1] - 8, th + 62)
      cv2.rectangle(view, (p1[0], ty - th - 8), (p1[0] + tw + 12, ty + 6), GREEN, -1)
      put(view, label, (p1[0] + 6, ty), 0.7, (15, 15, 15), 2)
    draw_banner(view, state)

    # ---- fps + compose window ----
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
    t_prev = now
    cv2.imshow('ALPR Campus Parking Control Center',
               np.hstack([view, draw_panel(state, mode, fps)]))

    # ---- keys ----
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
      break
    elif key == ord('m'):
      mode = modes[(modes.index(mode) + 1) % len(modes)]
    elif key == ord('e'):
      state.handle('KA51TEST12', 'ENTRY', 0.95)
    elif key == ord('x'):
      state.handle('KA51TEST12', 'EXIT', 0.95)

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  run_alpr_system()