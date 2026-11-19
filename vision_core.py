"""
Motor compartido de Vision IA (camara y video).

Pipeline por frame:
  1. YOLO11x  -> 80 objetos comunes (COCO) con cajas precisas.
  2. YOLOE    -> objetos que COCO no tiene (audifonos, llaves, cartera, lentes...).
                 Si su caja coincide con una de YOLO11 se usa como "voto" extra.
  3. ByteTrack -> asigna un ID estable a cada objeto entre frames.
  4. CLIP     -> mira el recorte de cada objeto de una familia confusa
                 (llave de auto / celular / control / mouse / audifonos...) y vota.
  5. Roboflow -> modelo propio "llave-auto" (en la nube) suma votos a "Llave de auto".
  6. Votacion temporal por objeto -> la etiqueta no parpadea entre frames.
"""
import os
import sys
import time
import base64
import colorsys
import hashlib
import threading
from collections import Counter, defaultdict, deque
from functools import lru_cache

import cv2
import numpy as np

import config as C
from vocabulario import (COCO_ES, YOLOE_VOCAB, ETIQUETAS_COCO, GRUPOS_CLIP,
                         DISTRACTORES_CLIP, DISTRACTORES_GRUPO, GRUPO_DE,
                         ESCENA_ETIQUETAS, DISTRACTORES_ESCENA, FAMILIA_DE)

os.chdir(C.BASE_DIR)  # rutas relativas y descargas de modelos siempre en la carpeta del proyecto

import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.engine.results import Boxes  # noqa: E402
from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: E402
from ultralytics.utils import IterableSimpleNamespace  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
GPU = DEVICE != "cpu"
NOMBRE_DISPOSITIVO = torch.cuda.get_device_name(0) if GPU else "CPU"

# Pesos de la votacion
A_DET = 0.30       # velocidad con la que el voto del detector sigue al frame actual
PESO_ALT = 0.70    # voto de YOLOE cuando coincide con una caja de YOLO11
PESO_RF = 1.0      # voto del modelo Roboflow "llave-auto"
HISTERESIS = 1.15  # la etiqueta nueva debe superar a la actual en 15% para cambiar
OLVIDAR_TRAS = 90  # frames sin ver un objeto antes de borrar su historial


# ══════════════════════════════════════════════════════════════════════════════
#  Utilidades de dibujo
# ══════════════════════════════════════════════════════════════════════════════
def color_de(etiqueta):
    """Color BGR estable para cada etiqueta (mismo objeto = mismo color siempre)."""
    fijos = {"Persona": (235, 160, 60), "Mano": (200, 120, 230), "Llave de auto": (90, 230, 90)}
    if etiqueta in fijos:
        return fijos[etiqueta]
    h = int(hashlib.md5(etiqueta.encode("utf-8")).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.70, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def color_texto(bgr):
    return (0, 0, 0) if 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2] > 140 else (255, 255, 255)


def rect_alpha(img, x1, y1, x2, y2, color, alpha):
    """Rectangulo relleno semitransparente (solo procesa la region, no copia todo el frame)."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    capa = np.empty_like(roi)
    capa[:] = color
    cv2.addWeighted(capa, alpha, roi, 1 - alpha, 0, dst=roi)


class Texto:
    """Texto con acentos y ene (OpenCV putText no los soporta). Usa la fuente de Windows."""
    FUENTES = [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\segoeui.ttf",
               r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf",
               "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]

    def __init__(self):
        self._fuentes = {}

    def _fuente(self, tam):
        if tam not in self._fuentes:
            fuente = None
            for ruta in self.FUENTES:
                try:
                    fuente = ImageFont.truetype(ruta, tam)
                    break
                except OSError:
                    continue
            self._fuentes[tam] = fuente or ImageFont.load_default()
        return self._fuentes[tam]

    @lru_cache(maxsize=2048)
    def mascara(self, texto, tam):
        f = self._fuente(tam)
        x0, _, x1, _ = f.getbbox(texto)
        asc, desc = f.getmetrics()
        img = Image.new("L", (max(1, x1 - min(0, x0) + 1), asc + desc), 0)
        ImageDraw.Draw(img).text((-min(0, x0), 0), texto, font=f, fill=255)
        return np.asarray(img, dtype=np.float32) / 255.0

    def medida(self, texto, tam):
        m = self.mascara(texto, tam)
        return m.shape[1], m.shape[0]

    def dibujar(self, img, texto, x, y, color, tam=18):
        m = self.mascara(texto, tam)
        h, w = m.shape
        H, W = img.shape[:2]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x1 <= x0 or y1 <= y0:
            return
        a = m[y0 - y:y1 - y, x0 - x:x1 - x, None]
        roi = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = (roi * (1 - a) + np.float32(color) * a).astype(np.uint8)


def iou_vs(b, cajas):
    """IoU y fraccion de la caja chica contenida en la grande, de b contra N cajas."""
    if len(cajas) == 0:
        return np.zeros(0), np.zeros(0)
    ix1 = np.maximum(b[0], cajas[:, 0]); iy1 = np.maximum(b[1], cajas[:, 1])
    ix2 = np.minimum(b[2], cajas[:, 2]); iy2 = np.minimum(b[3], cajas[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    a1 = (b[2] - b[0]) * (b[3] - b[1])
    a2 = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
    iou = inter / (a1 + a2 - inter + 1e-9)
    contenido = inter / (np.minimum(a1, a2) + 1e-9)
    return iou, contenido


# ══════════════════════════════════════════════════════════════════════════════
#  Estado de cada objeto rastreado
# ══════════════════════════════════════════════════════════════════════════════
class Deteccion:
    __slots__ = ("caja", "conf", "etiqueta", "fuente", "alts")

    def __init__(self, caja, conf, etiqueta, fuente):
        self.caja = np.asarray(caja, dtype=np.float32)
        self.conf = conf
        self.etiqueta = etiqueta
        self.fuente = fuente
        self.alts = []  # [(etiqueta, conf)] votos de YOLOE sobre la misma caja


class Objeto:
    def __init__(self, tid):
        self.tid = tid
        self.det = defaultdict(float)  # votos del detector (media movil)
        self.clip = {}                 # votos de CLIP (media movil)
        self.grupo_clip = None
        self.fondo = 0.0               # prob. de que CLIP vea "mano/mesa/fondo" en vez de un objeto
        self.rf = 0.0                  # votos del modelo Roboflow
        self.etiqueta = None
        self.conf = 0.0
        self.caja = None
        self.fuente = "coco"
        self.visto = 0
        self.n_clip = 0
        self.ultimo_clip = -10 ** 9
        self.rastro = deque(maxlen=C.MAX_RASTRO)

    # -- votos -----------------------------------------------------------------
    def votar_detector(self, d):
        for k in self.det:
            self.det[k] *= 1 - A_DET
        self.det[d.etiqueta] += A_DET * d.conf
        for etq, c in d.alts:
            self.det[etq] += A_DET * PESO_ALT * c
        self.conf = d.conf
        self.fuente = d.fuente

    def votar_clip(self, grupo, probs, fondo):
        if grupo != self.grupo_clip:
            self.clip, self.grupo_clip, self.n_clip = {}, grupo, 0
        b = 1.0 if self.n_clip == 0 else 0.5
        if grupo == "persona" and fondo < 0.85:
            # Persona vs Mano: decide la proporcion entre ambas (la mesa del fondo no importa)
            total = sum(probs.values()) or 1.0
            probs = {e: p / total for e, p in probs.items()}
        for e, p in probs.items():
            self.clip[e] = (1 - b) * self.clip.get(e, 0.0) + b * p
        self.fondo = (1 - b) * self.fondo + b * fondo
        self.n_clip += 1

    def puntaje(self, e):
        # YOLO dice "person" tanto para una persona como para una mano: su voto vale para ambas
        det = self.det.get("Persona", 0.0) if e == "Mano" else self.det.get(e, 0.0)
        return (det + C.CLIP_PESO * self.clip.get(e, 0.0)
                + (PESO_RF * self.rf if e == "Llave de auto" else 0.0))

    def decidir(self):
        cands = set(self.det) | set(self.clip)
        if self.rf > 0.05:
            cands.add("Llave de auto")
        if not cands:
            return
        mejor = max(cands, key=self.puntaje)
        if self.etiqueta is None or self.etiqueta not in cands:
            self.etiqueta = mejor
        elif mejor != self.etiqueta and self.puntaje(mejor) > HISTERESIS * self.puntaje(self.etiqueta) + 0.02:
            self.etiqueta = mejor

    # -- consultas -------------------------------------------------------------
    @property
    def grupo(self):
        """Familia de confusion segun lo que ve el detector (decide si CLIP debe mirar)."""
        if self.det:
            g = GRUPO_DE.get(max(self.det, key=self.det.get))
            if g:
                return g
        return GRUPO_DE.get(self.etiqueta)

    @property
    def verificado_clip(self):
        return self.n_clip > 0 and self.clip.get(self.etiqueta, 0.0) >= 0.45

    @property
    def verificado_rf(self):
        return self.etiqueta == "Llave de auto" and self.rf >= 0.25

    @property
    def oculto(self):
        """Falsa alarma: CLIP confirmo varias veces que el recorte es fondo (piso, mesa, sombra...).
        A las cajas que solo vienen de YOLOE (mas ruidoso) se les exige menos evidencia."""
        if self.n_clip == 0 or max(self.clip.values(), default=0.0) >= 0.25:
            return False
        if self.n_clip == 1:
            return self.fondo >= 0.75
        return self.fondo >= (0.50 if self.fuente == "yoloe" else 0.65)


# ══════════════════════════════════════════════════════════════════════════════
#  CLIP: segunda opinion sobre el recorte
# ══════════════════════════════════════════════════════════════════════════════
def _vector(salida):
    """transformers <5 devuelve un tensor; >=5 devuelve un objeto con pooler_output."""
    return salida if torch.is_tensor(salida) else salida.pooler_output


class RefinadorCLIP:
    def __init__(self, sincrono):
        from transformers import CLIPModel, CLIPProcessor
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()

        print(f"  Cargando CLIP ({C.MODELO_CLIP})...")
        self.modelo = CLIPModel.from_pretrained(C.MODELO_CLIP).to(DEVICE).eval()
        if GPU:
            self.modelo = self.modelo.half()
        self.dtype = next(self.modelo.parameters()).dtype
        self.proc = CLIPProcessor.from_pretrained(C.MODELO_CLIP)
        self.escala = float(self.modelo.logit_scale.exp().item())

        # Embeddings de texto calculados UNA vez (antes se recalculaban en cada recorte)
        self.grupos = {}
        grupos = [(g, etqs, DISTRACTORES_GRUPO.get(g, DISTRACTORES_CLIP)) for g, etqs in GRUPOS_CLIP.items()]
        grupos.append(("escena", ESCENA_ETIQUETAS, DISTRACTORES_ESCENA))
        with torch.no_grad():
            for g, etqs, distractores in grupos:
                nombres = list(etqs)
                filas = ([self._texto([f"a photo of {p}." for p in etqs[e]]) for e in nombres]
                         + [self._texto([f"a photo of {d}."]) for d in distractores])
                M = torch.stack(filas)
                self.grupos[g] = (nombres, M / M.norm(dim=-1, keepdim=True))

        self.sincrono = sincrono
        self.lock = threading.Lock()
        self.evento = threading.Event()
        self.pendientes = {}
        self.resultados = {}
        self.escena_pendiente = None
        self.escena = (-10 ** 9, None, 0.0, [])  # (frame, etiqueta_centro, prob, [(caja, etq, prob)])
        self.parar = False
        self.error = None
        if not sincrono:
            threading.Thread(target=self._hilo, daemon=True, name="clip").start()

    @torch.no_grad()
    def _texto(self, prompts):
        tok = self.proc(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
        f = _vector(self.modelo.get_text_features(**tok)).float()
        f = f / f.norm(dim=-1, keepdim=True)
        return f.mean(0)

    @torch.no_grad()
    def _clasificar(self, items):
        imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for _, _, c in items]
        px = self.proc(images=imgs, return_tensors="pt")["pixel_values"].to(DEVICE, self.dtype)
        f = _vector(self.modelo.get_image_features(pixel_values=px)).float()
        f = f / f.norm(dim=-1, keepdim=True)
        salida = {}
        for i, (tid, g, _) in enumerate(items):
            nombres, M = self.grupos[g]
            p = (self.escala * f[i] @ M.T).softmax(-1).cpu().numpy()
            salida[tid] = (g, {n: float(p[k]) for k, n in enumerate(nombres)},
                           float(p[len(nombres):].sum()))
        return salida

    def solicitar(self, tid, grupo, recorte):
        with self.lock:
            self.pendientes[tid] = (grupo, recorte)
        self.evento.set()

    def solicitar_escena(self, frame, propuestas, fi):
        with self.lock:
            self.escena_pendiente = (frame, propuestas, fi)
        self.evento.set()

    def _procesar_escena(self, frame, propuestas, fi):
        """Busca objetos que los detectores no encontraron: clasifica el centro de la imagen
        y las cajas de baja confianza que se habian descartado."""
        H, W = frame.shape[:2]
        mx, my = int(W * 0.12), int(H * 0.12)
        recortes = [frame[my:H - my, mx:W - mx]]
        for x1, y1, x2, y2 in propuestas:
            px, py = int(0.08 * (x2 - x1)), int(0.08 * (y2 - y1))
            recortes.append(frame[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)])
        res = self._clasificar([(i, "escena", r) for i, r in enumerate(recortes)])

        _, probs, _ = res[0]
        etq_centro = max(probs, key=probs.get)
        p_centro = probs[etq_centro]

        por_etq = defaultdict(list)
        for i, caja in enumerate(propuestas, start=1):
            _, probs, _ = res[i]
            e = max(probs, key=probs.get)
            if probs[e] >= C.ESCENA_MIN_PROB:
                por_etq[e].append((caja, probs[e]))
        objetos = []
        for e, lista in por_etq.items():  # varias piezas del mismo objeto (p.ej. 2 orejeras) -> 1 caja
            cajas = np.array([c for c, _ in lista])
            caja = (int(cajas[:, 0].min()), int(cajas[:, 1].min()), int(cajas[:, 2].max()), int(cajas[:, 3].max()))
            objetos.append((caja, e, max(p for _, p in lista)))
        with self.lock:
            self.escena = (fi, etq_centro, p_centro, objetos)

    def procesar(self):
        """Procesa todo lo pendiente (en lotes). Lo llama el hilo o, en modo video, el bucle."""
        while True:
            with self.lock:
                lote = list(self.pendientes.items())[:C.CLIP_LOTE]
                for tid, _ in lote:
                    del self.pendientes[tid]
                escena, self.escena_pendiente = self.escena_pendiente, None
            if escena is not None:
                self._procesar_escena(*escena)
            if not lote:
                if escena is None:
                    return
                continue
            res = self._clasificar([(tid, g, c) for tid, (g, c) in lote])
            with self.lock:
                self.resultados.update(res)

    def recoger(self):
        with self.lock:
            r, self.resultados = self.resultados, {}
        return r

    def _hilo(self):
        while not self.parar:
            if not self.evento.wait(0.1):
                continue
            self.evento.clear()
            try:
                self.procesar()
            except Exception as e:  # no tumbar la app por un recorte raro
                if self.error is None:
                    print(f"[CLIP] error: {e}")
                self.error = e

    def cerrar(self):
        self.parar = True


# ══════════════════════════════════════════════════════════════════════════════
#  Roboflow: modelo propio "llave-auto"
# ══════════════════════════════════════════════════════════════════════════════
class ClienteRoboflow:
    def __init__(self):
        import requests
        self.sesion = requests.Session()
        self.url = f"https://detect.roboflow.com/{C.ROBOFLOW_MODELO}"
        self.lock = threading.Lock()
        self.evento = threading.Event()
        self.trabajo = None
        self.ocupado = False
        self.ultimo_envio = 0.0
        self.resultado = ([], -1, 0.0)  # (cajas, frame_origen, hora)
        self.errores = 0
        self.activo = bool(C.ROBOFLOW_API_KEY)
        if not self.activo:
            print("  [AVISO] Sin API key de Roboflow (roboflow_key.txt o variable ROBOFLOW_API_KEY):"
                  " se continua sin el modelo llave-auto.")
        self.parar = False
        threading.Thread(target=self._hilo, daemon=True, name="roboflow").start()

    def enviar(self, frame, frame_idx):
        if not self.activo or self.ocupado or time.time() - self.ultimo_envio < C.ROBOFLOW_INTERVALO:
            return
        h, w = frame.shape[:2]
        esc = min(1.0, 640 / max(h, w))  # lado mayor a 640 px (sirve tambien para video vertical)
        peq = cv2.resize(frame, (int(w * esc), int(h * esc)), interpolation=cv2.INTER_AREA)
        with self.lock:
            self.trabajo = (peq, esc, frame_idx)
            self.ocupado = True
            self.ultimo_envio = time.time()
        self.evento.set()

    def obtener(self, frame_idx, max_frames, max_seg):
        with self.lock:
            cajas, fi, hora = self.resultado
        if frame_idx - fi > max_frames or time.time() - hora > max_seg:
            return []
        return cajas

    def _inferir(self, img):
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        r = self.sesion.post(
            self.url,
            params={"api_key": C.ROBOFLOW_API_KEY, "confidence": C.ROBOFLOW_CONF, "overlap": 30},
            data=base64.b64encode(buf).decode("ascii"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=C.ROBOFLOW_TIMEOUT,
        )
        if r.status_code in (401, 403):
            self.errores = 99
            raise RuntimeError("API key de Roboflow invalida o sin permiso")
        r.raise_for_status()
        return r.json().get("predictions", [])

    def _hilo(self):
        while not self.parar:
            if not self.evento.wait(0.1):
                continue
            self.evento.clear()
            with self.lock:
                t, self.trabajo = self.trabajo, None
            if t is None:
                continue
            peq, esc, fi = t
            try:
                cajas = []
                for p in self._inferir(peq):
                    if p.get("class", "").lower() != "llave-auto" or p["confidence"] * 100 < C.ROBOFLOW_CONF:
                        continue
                    x, y, w, h = p["x"] / esc, p["y"] / esc, p["width"] / esc, p["height"] / esc
                    cajas.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2, float(p["confidence"])))
                with self.lock:
                    self.resultado = (cajas, fi, time.time())
                self.errores = 0
            except Exception as e:
                self.errores += 1
                if self.errores >= 3 and self.activo:
                    self.activo = False
                    print(f"[Roboflow] desactivado ({e}). Se sigue detectando sin el. Tecla r para reintentar.")
            finally:
                with self.lock:
                    self.ocupado = False

    def reactivar(self):
        self.errores = 0
        self.activo = bool(C.ROBOFLOW_API_KEY)

    def cerrar(self):
        self.parar = True


# ══════════════════════════════════════════════════════════════════════════════
#  Motor principal
# ══════════════════════════════════════════════════════════════════════════════
class Motor:
    def __init__(self, modo="camara", tta=None):
        assert modo in ("camara", "video")
        self.modo = modo
        self.sincrono = modo == "video"  # en video CLIP corre en el mismo hilo: resultado reproducible
        self.imgsz = C.IMGSZ_CAMARA if modo == "camara" else C.IMGSZ_VIDEO
        self.tta = (C.TTA_VIDEO if tta is None else tta) and modo == "video"
        self.conf = C.CONF_MOSTRAR
        self.mostrar_rastro = C.MOSTRAR_RASTRO
        self.preprocesar = False
        self.ayuda = False
        self.texto = Texto()
        self.frame_idx = 0
        self.fps = 0.0
        self.objetos = {}
        self.conteo = Counter()
        self.escena_txt = None
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        print(f"Dispositivo: {NOMBRE_DISPOSITIVO}")
        self._cargar_yolo()
        self.yoloe = self._cargar_yoloe() if C.USAR_YOLOE else None
        self.usar_yoloe = self.yoloe is not None

        self.clip = None
        if C.USAR_CLIP:
            try:
                self.clip = RefinadorCLIP(self.sincrono)
            except Exception as e:
                print(f"  [AVISO] CLIP no disponible ({e}). Se continua sin el.")
        self.usar_clip = self.clip is not None

        self.rf = ClienteRoboflow() if C.USAR_ROBOFLOW else None
        self.usar_rf = self.rf is not None

        self.tracker = self._nuevo_tracker()
        print("Todo listo.\n")

    # -- carga -----------------------------------------------------------------
    def _cargar_yolo(self):
        nombre = C.MODELO_YOLO
        if not GPU:
            nombre = C.MODELO_YOLO_CPU
            self.imgsz = min(self.imgsz, 640)
            print("  [AVISO] Sin GPU CUDA: se usa un modelo mas ligero y resolucion 640.")
        print(f"  Cargando {nombre}...")
        self.yolo = YOLO(nombre)
        self.nombre_yolo = os.path.splitext(nombre)[0].upper()
        self.yolo.predict(np.zeros((640, 640, 3), np.uint8), device=DEVICE, half=GPU, verbose=False)

    def _cargar_yoloe(self):
        if not GPU:
            print("  [AVISO] YOLOE desactivado en CPU (seria demasiado lento).")
            return None
        try:
            from ultralytics import YOLOE
            print(f"  Cargando {C.MODELO_YOLOE} (vocabulario abierto, {len(YOLOE_VOCAB)} objetos extra)...")
            m = YOLOE(C.MODELO_YOLOE)
            self.yoloe_nombres = list(YOLOE_VOCAB)
            m.set_classes(self.yoloe_nombres)
            # half=False: la version actual de ultralytics falla con FP16 en modelos -seg
            m.predict(np.zeros((640, 640, 3), np.uint8), device=DEVICE, half=False, verbose=False)
            return m
        except Exception as e:
            print(f"  [AVISO] YOLOE no disponible ({e}). Se continua sin el.")
            return None

    def _nuevo_tracker(self):
        cfg = IterableSimpleNamespace(
            tracker_type="bytetrack", track_high_thresh=self.conf, track_low_thresh=0.10,
            new_track_thresh=self.conf, track_buffer=45, match_thresh=0.8, fuse_score=True)
        return BYTETracker(cfg)

    def cambiar_conf(self, delta):
        self.conf = round(min(0.90, max(0.10, self.conf + delta)), 2)
        self.tracker.args.track_high_thresh = self.conf
        self.tracker.args.new_track_thresh = self.conf
        print(f"Confianza minima: {self.conf * 100:.0f}%")

    # -- deteccion -------------------------------------------------------------
    def _detectar(self, img):
        """Devuelve (detecciones, propuestas). Las propuestas son cajas de baja confianza que
        CLIP revisa en la busqueda de "escena" (objetos que ningun detector reconocio)."""
        dets, propuestas = [], []
        r = self.yolo.predict(img, conf=0.10, iou=C.IOU_NMS, imgsz=self.imgsz, device=DEVICE,
                              half=GPU, agnostic_nms=True, augment=self.tta, max_det=100,
                              verbose=False)[0]
        if len(r.boxes):
            for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                               r.boxes.cls.cpu().numpy().astype(int)):
                en = r.names[int(k)]
                dets.append(Deteccion(b, float(c), COCO_ES.get(en, en.capitalize()), "coco"))
                if c < self.conf and en != "person":
                    propuestas.append((float(c), b))

        if self.usar_yoloe:
            r2 = self.yoloe.predict(img, conf=0.05, iou=C.IOU_NMS, imgsz=C.IMGSZ_YOLOE,
                                    device=DEVICE, half=False, agnostic_nms=True, max_det=50,
                                    verbose=False)[0]
            if len(r2.boxes):
                base = np.array([d.caja for d in dets]) if dets else np.zeros((0, 4), np.float32)
                for b, c, k in zip(r2.boxes.xyxy.cpu().numpy(), r2.boxes.conf.cpu().numpy(),
                                   r2.boxes.cls.cpu().numpy().astype(int)):
                    etq = YOLOE_VOCAB.get(r2.names[int(k)])
                    if etq is None:
                        continue
                    c = float(c)
                    iou, cont = iou_vs(b, base)
                    j = int(np.argmax(iou)) if len(iou) else -1
                    misma = j >= 0 and (iou[j] >= 0.45 or (cont[j] >= 0.85 and iou[j] >= 0.30))
                    if c < 0.15:
                        if not misma and etq != "Persona":
                            propuestas.append((c, b))
                    elif misma:
                        d = dets[j]  # misma caja que YOLO11: se suma como voto
                        d.alts.append((etq, c))
                        if etq not in ETIQUETAS_COCO:
                            d.conf = max(d.conf, c)
                    elif c >= (C.CONF_YOLOE if etq not in ETIQUETAS_COCO else C.CONF_YOLOE + 0.10):
                        dets.append(Deteccion(b, c, etq, "yoloe"))  # objeto que YOLO11 no vio
                    elif etq != "Persona":
                        propuestas.append((c, b))

        # propuestas: sin duplicados, de tamano razonable, las mas confiables primero
        lado_min = 0.10 * min(img.shape[:2])
        finales = []
        for c, b in sorted(propuestas, key=lambda x: -x[0]):
            w, h = b[2] - b[0], b[3] - b[1]
            if min(w, h) < lado_min or max(w, h) > 4 * min(w, h):  # tiras delgadas = cables/bordes
                continue
            if finales and iou_vs(b, np.array(finales))[0].max() >= 0.6:
                continue
            finales.append(b)
            if len(finales) >= C.ESCENA_MAX_PROPUESTAS:
                break
        return dets, finales

    def _rastrear(self, dets, shape):
        if dets:  # columna "clase" = indice de la deteccion, para recuperarla despues
            arr = np.array([[*d.caja, d.conf, i] for i, d in enumerate(dets)], dtype=np.float32)
        else:
            arr = np.zeros((0, 6), dtype=np.float32)
        return self.tracker.update(Boxes(arr, shape[:2]))

    # -- un frame completo -----------------------------------------------------
    def procesar(self, frame):
        t0 = time.perf_counter()
        self.frame_idx += 1
        fi = self.frame_idx
        H, W = frame.shape[:2]

        entrada = self._mejorar(frame) if self.preprocesar else frame
        dets, propuestas = self._detectar(entrada)
        tracks = self._rastrear(dets, frame.shape)

        visibles = []
        for t in tracks:
            x1, y1, x2, y2 = t[:4]
            # t[7] (idx) es relativo a un subconjunto interno del tracker; por eso el indice
            # original de la deteccion viaja en la columna de clase (t[6]).
            tid, idx = int(t[4]), int(t[6])
            d = dets[idx]
            obj = self.objetos.get(tid)
            if obj is None:
                obj = self.objetos[tid] = Objeto(tid)
            obj.votar_detector(d)
            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = int(min(W - 1, x2)), int(min(H - 1, y2))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            obj.caja = (x1, y1, x2, y2)
            obj.visto = fi
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if obj.rastro:  # suavizado: evita el "garabato" cuando el objeto casi no se mueve
                px, py = obj.rastro[-1]
                cx, cy = 0.6 * px + 0.4 * cx, 0.6 * py + 0.4 * cy
            obj.rastro.append((int(cx), int(cy)))
            visibles.append(obj)

        # CLIP: pedir verificacion a los objetos de familias confusas
        if self.usar_clip:
            for obj in visibles:
                g = obj.grupo
                if g is None:
                    continue
                intervalo = C.CLIP_INTERVALO_INICIAL if obj.n_clip < 3 else C.CLIP_INTERVALO
                if fi - obj.ultimo_clip < intervalo:
                    continue
                x1, y1, x2, y2 = obj.caja
                if min(x2 - x1, y2 - y1) < 16:
                    continue
                px, py = int(0.10 * (x2 - x1)), int(0.10 * (y2 - y1))
                rec = frame[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)].copy()
                self.clip.solicitar(obj.tid, g, rec)
                obj.ultimo_clip = fi
            intervalo = C.ESCENA_INTERVALO_VIDEO if self.modo == "video" else C.ESCENA_INTERVALO_CAMARA
            if fi % intervalo == 0:
                cajas_vis = np.array([o.caja for o in visibles], dtype=np.float32).reshape(-1, 4)
                libres = []
                for b in propuestas:  # solo zonas donde no hay ya un objeto rastreado
                    iou, cont = iou_vs(b, cajas_vis)
                    if len(iou) == 0 or (iou.max() < 0.5 and cont.max() < 0.9):
                        libres.append(tuple(int(v) for v in b))
                self.clip.solicitar_escena(frame.copy(), libres, fi)
            if self.sincrono:
                self.clip.procesar()
            for tid, (g, probs, fondo) in self.clip.recoger().items():
                if tid in self.objetos:
                    self.objetos[tid].votar_clip(g, probs, fondo)

        # Roboflow: llave de auto
        rf_sueltas = []
        for obj in visibles:
            obj.rf *= 0.85
        if self.usar_rf and self.rf.activo:
            self.rf.enviar(entrada, fi)
            max_frames, max_seg = (12, 60.0) if self.modo == "video" else (10 ** 9, 1.0)
            cajas_obj = np.array([o.caja for o in visibles], dtype=np.float32).reshape(-1, 4)
            for x1, y1, x2, y2, c in self.rf.obtener(fi, max_frames, max_seg):
                iou, cont = iou_vs(np.array([x1, y1, x2, y2]), cajas_obj)
                j = int(np.argmax(iou)) if len(iou) else -1
                if j >= 0 and (iou[j] >= 0.25 or cont[j] >= 0.7):
                    o = visibles[j]
                    o.rf = max(o.rf, c)
                elif c >= 0.60:
                    rf_sueltas.append((int(x1), int(y1), int(x2), int(y2), c))

        for obj in visibles:
            obj.decidir()

        # Limpiar objetos que ya no estan (evita fuga de memoria en sesiones largas)
        if fi % 30 == 0:
            for tid in [t for t, o in self.objetos.items() if fi - o.visto > OLVIDAR_TRAS]:
                del self.objetos[tid]

        # Que mostrar: sin falsas alarmas y sin cajas duplicadas del mismo objeto
        candidatos = sorted((o for o in visibles if o.etiqueta and not o.oculto), key=lambda o: -o.conf)
        mostrar = []
        for o in candidatos:
            dup = False
            for m in mostrar:
                if m.etiqueta == o.etiqueta:
                    iou, cont = iou_vs(np.array(o.caja, np.float32), np.array([m.caja], np.float32))
                    dup = iou[0] >= 0.6 or cont[0] >= 0.85
                    if dup:
                        break
            if not dup:
                mostrar.append(o)

        # Objetos encontrados por la busqueda de escena (CLIP) donde los detectores no vieron nada
        escena_cajas = []
        self.escena_txt = None
        if self.usar_clip:
            efi, etq_c, p_c, objs = self.clip.escena
            vigencia = 2 * (C.ESCENA_INTERVALO_VIDEO if self.modo == "video" else C.ESCENA_INTERVALO_CAMARA) + 2
            if fi - efi <= vigencia:
                if p_c >= 0.5:
                    self.escena_txt = f"{etq_c} {p_c * 100:.0f}%"
                cajas_m = np.array([o.caja for o in mostrar], dtype=np.float32).reshape(-1, 4)
                ya = set()
                for o in mostrar:
                    ya |= FAMILIA_DE.get(o.etiqueta, {o.etiqueta})
                # objetos pequenos ya detectados (llave, celular...): una caja de escena que los
                # toca suele ser un pedazo de ellos, no un objeto nuevo
                chicos = np.array([o.caja for o in mostrar if GRUPO_DE.get(o.etiqueta) == "mano"],
                                  dtype=np.float32).reshape(-1, 4)
                for caja, e, p in objs:
                    if e in ya:  # ese objeto ya tiene su caja: no duplicar
                        continue
                    b = np.array(caja, np.float32)
                    iou, cont = iou_vs(b, cajas_m)
                    if len(iou) and (iou.max() >= 0.3 or cont.max() >= 0.7):
                        continue
                    if len(chicos):
                        inter = (np.clip(np.minimum(b[2], chicos[:, 2]) - np.maximum(b[0], chicos[:, 0]), 0, None)
                                 * np.clip(np.minimum(b[3], chicos[:, 3]) - np.maximum(b[1], chicos[:, 1]), 0, None))
                        if inter.max() > 0.10 * (b[2] - b[0]) * (b[3] - b[1]):
                            continue
                    escena_cajas.append((caja, e, p))

        salida = frame.copy()
        self.conteo = Counter()
        mostrar.sort(key=lambda o: -(o.caja[2] - o.caja[0]) * (o.caja[3] - o.caja[1]))
        if self.mostrar_rastro:
            for o in mostrar:
                self._dibujar_rastro(salida, o.rastro, color_de(o.etiqueta))
        for o in mostrar:
            insignias = []
            if o.verificado_clip:
                insignias.append("IA")
            if o.verificado_rf:
                insignias.append("RF")
            self._dibujar_caja(salida, o.caja, o.etiqueta, o.conf, f"#{o.tid}", insignias)
            self.conteo[o.etiqueta] += 1
        for caja, e, p in escena_cajas:
            self._dibujar_caja(salida, caja, e, p, "", ["IA"])
            self.conteo[e] += 1
        for x1, y1, x2, y2, c in rf_sueltas:
            self._dibujar_caja(salida, (x1, y1, x2, y2), "Llave de auto", c, "", ["RF"])
            self.conteo["Llave de auto"] += 1

        dt = time.perf_counter() - t0
        self.fps = 1 / dt if self.fps == 0 else 0.9 * self.fps + 0.1 / max(dt, 1e-6)
        return salida

    def _mejorar(self, frame):
        """Mejora de contraste (CLAHE) para escenas oscuras. Tecla p."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # -- dibujo ----------------------------------------------------------------
    def _escala_ui(self, img):
        return max(0.75, min(img.shape[:2]) / 720)

    def _dibujar_caja(self, img, caja, etiqueta, conf, prefijo, insignias):
        x1, y1, x2, y2 = caja
        H, W = img.shape[:2]
        s = self._escala_ui(img)
        color = color_de(etiqueta)
        grosor = max(2, int(round(2 * s)))
        rect_alpha(img, x1, y1, x2, y2, color, 0.12)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, grosor, cv2.LINE_AA)
        L = max(6, min(int(22 * s), (x2 - x1) // 4, (y2 - y1) // 4))
        for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (px, py), (px + dx * L, py), color, grosor + 2, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + dy * L), color, grosor + 2, cv2.LINE_AA)

        tam = int(18 * s)
        texto = f"{prefijo + ' ' if prefijo else ''}{etiqueta}  {conf * 100:.0f}%"
        if insignias:
            texto += "  " + " ".join(f"[{i}]" for i in insignias)
        tw, th = self.texto.medida(texto, tam)
        pad = max(3, int(4 * s))
        lx = min(max(0, x1), max(0, W - tw - 2 * pad))
        ly = y1 - th - 2 * pad
        if ly < 0:  # sin espacio arriba: etiqueta dentro de la caja
            ly = y1
        cv2.rectangle(img, (lx, ly), (lx + tw + 2 * pad, ly + th + 2 * pad), color, -1)
        self.texto.dibujar(img, texto, lx + pad, ly + pad, color_texto(color), tam)

        # barra de confianza
        by = min(H - 4, y2 + 3)
        alto = max(3, int(4 * s))
        cv2.rectangle(img, (x1, by), (x2, by + alto), (40, 40, 40), -1)
        bc = (60, 220, 60) if conf >= 0.7 else (0, 200, 255) if conf >= 0.5 else (60, 90, 255)
        cv2.rectangle(img, (x1, by), (x1 + int((x2 - x1) * min(conf, 1.0)), by + alto), bc, -1)

    @staticmethod
    def _dibujar_rastro(img, puntos, color):
        pts = list(puntos)
        for i in range(1, len(pts)):
            a = i / len(pts)
            cv2.line(img, pts[i - 1], pts[i], tuple(int(v * a) for v in color),
                     max(1, int(4 * a)), cv2.LINE_AA)

    def dibujar_panel(self, img, extra=()):
        s = self._escala_ui(img)
        tam, tam_c = int(17 * s), int(16 * s)
        lh = int(tam * 1.45)
        on = lambda v: "ON" if v else "OFF"  # noqa: E731
        lineas = [
            (f"Vision IA  ·  {NOMBRE_DISPOSITIVO[:30]}", (255, 200, 80)),
            (f"FPS {self.fps:4.1f}   Confianza {self.conf * 100:.0f}%   Objetos {sum(self.conteo.values())}",
             (200, 255, 200)),
            (f"YOLOE {on(self.usar_yoloe)}  CLIP {on(self.usar_clip)}  "
             f"Roboflow {on(self.usar_rf and self.rf.activo)}  Contraste {on(self.preprocesar)}",
             (180, 220, 255)),
            *[(t, (180, 220, 255)) for t in extra],
        ]
        if self.escena_txt:
            lineas.append((f"Objeto al centro (CLIP): {self.escena_txt}", (120, 255, 255)))
        cuenta = [(f"   {n}: {c}", (255, 255, 255)) for n, c in self.conteo.most_common(10)]
        todo = lineas + ([("", None)] + cuenta if cuenta else []) + [("H = ayuda", (150, 150, 150))]
        ancho = max(self.texto.medida(t, tam)[0] for t, _ in todo if t) + int(20 * s)
        alto = lh * len(todo) + int(12 * s)
        rect_alpha(img, 0, 0, ancho, alto, (12, 12, 12), 0.72)
        for i, (t, col) in enumerate(todo):
            if t:
                self.texto.dibujar(img, t, int(10 * s), int(6 * s) + i * lh, col,
                                   tam if i < len(lineas) else tam_c)
        if self.ayuda:
            self._dibujar_ayuda(img)

    def _dibujar_ayuda(self, img):
        s = self._escala_ui(img)
        tam = int(18 * s)
        filas = [("q / ESC", "salir"), ("s", "guardar captura"), ("+ / -", "subir / bajar confianza"),
                 ("t", "rastros on/off"), ("e", "YOLOE (objetos extra) on/off"),
                 ("c", "CLIP (verificación) on/off"), ("r", "Roboflow (llave de auto) on/off"),
                 ("p", "mejorar contraste on/off"), ("h", "mostrar / ocultar ayuda")]
        if self.modo == "video":
            filas.insert(1, ("ESPACIO", "pausar / continuar"))
        lh = int(tam * 1.5)
        col = max(self.texto.medida(k, tam)[0] for k, _ in filas) + int(18 * s)
        ancho = col + max(self.texto.medida(d, tam)[0] for _, d in filas) + int(40 * s)
        alto = lh * (len(filas) + 1) + int(24 * s)
        H, W = img.shape[:2]
        x0, y0 = max(0, (W - ancho) // 2), max(0, (H - alto) // 2)
        xt, yt = x0 + int(20 * s), y0 + int(12 * s)
        rect_alpha(img, x0, y0, x0 + ancho, y0 + alto, (20, 20, 20), 0.85)
        self.texto.dibujar(img, "CONTROLES", xt, yt, (80, 220, 255), tam)
        for i, (k, d) in enumerate(filas, start=1):
            self.texto.dibujar(img, k, xt, yt + i * lh, (120, 255, 255), tam)
            self.texto.dibujar(img, d, xt + col, yt + i * lh, (240, 240, 240), tam)

    # -- teclado ---------------------------------------------------------------
    def tecla(self, k, imagen):
        """Devuelve False si hay que salir."""
        if k in (ord("q"), ord("Q"), 27):
            return False
        if k in (ord("s"), ord("S")):
            C.CARPETA_CAPTURAS.mkdir(exist_ok=True)
            ruta = C.CARPETA_CAPTURAS / f"captura_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(ruta), imagen)
            print(f"Captura guardada: {ruta}")
        elif k in (ord("t"), ord("T")):
            self.mostrar_rastro = not self.mostrar_rastro
        elif k in (ord("+"), ord("=")):
            self.cambiar_conf(+0.05)
        elif k in (ord("-"), ord("_")):
            self.cambiar_conf(-0.05)
        elif k in (ord("e"), ord("E")) and self.yoloe is not None:
            self.usar_yoloe = not self.usar_yoloe
        elif k in (ord("c"), ord("C")) and self.clip is not None:
            self.usar_clip = not self.usar_clip
            if not self.usar_clip:
                for o in self.objetos.values():
                    o.clip, o.n_clip, o.fondo = {}, 0, 0.0
        elif k in (ord("r"), ord("R")) and self.rf is not None:
            if self.usar_rf and not self.rf.activo:
                self.rf.reactivar()
                print("[Roboflow] reintentando...")
            else:
                self.usar_rf = not self.usar_rf
        elif k in (ord("p"), ord("P")):
            self.preprocesar = not self.preprocesar
        elif k in (ord("h"), ord("H")):
            self.ayuda = not self.ayuda
        return True

    def cerrar(self):
        if self.clip:
            self.clip.cerrar()
        if self.rf:
            self.rf.cerrar()


# ══════════════════════════════════════════════════════════════════════════════
#  Ventana
# ══════════════════════════════════════════════════════════════════════════════
class Ventana:
    def __init__(self, titulo):
        self.titulo = titulo
        self._ajustada = False
        cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)

    def mostrar(self, img):
        if not self._ajustada:
            h, w = img.shape[:2]
            esc = min(1.0, C.ALTO_MAX_VENTANA / h, 1600 / w)
            cv2.resizeWindow(self.titulo, int(w * esc), int(h * esc))
            self._ajustada = True
        cv2.imshow(self.titulo, img)

    def cerrada(self):
        """True si el usuario cerro la ventana con la X (antes el programa seguia corriendo)."""
        try:
            return cv2.getWindowProperty(self.titulo, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True


def consola_utf8():
    """Evita errores al imprimir acentos en la consola de Windows."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
