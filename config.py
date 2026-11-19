"""
Configuracion central de Vision IA.
Cambia aqui los valores; detectar_camara.py y detectar_video.py los leen de este archivo.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CARPETA_CAPTURAS = BASE_DIR / "capturas"
CARPETA_RESULTADOS = BASE_DIR / "resultados"

# ── MODELOS ───────────────────────────────────────────────────────────────────
MODELO_YOLO = "yolo11x.pt"            # detector principal: 80 clases COCO (celular, mouse, laptop...)
MODELO_YOLO_CPU = "yolo11s.pt"        # se usa si no hay GPU (mucho mas rapido en CPU)
MODELO_YOLOE = "yoloe-26l-seg.pt"     # vocabulario abierto: objetos que COCO NO tiene (audifonos, llaves...)
MODELO_CLIP = "openai/clip-vit-large-patch14"  # "segunda opinion" para objetos que se confunden

USAR_YOLOE = True      # tecla e
USAR_CLIP = True       # tecla c
USAR_ROBOFLOW = True   # tecla r  (modelo propio "llave-auto" en la nube)

# ── DETECCION ─────────────────────────────────────────────────────────────────
CONF_MOSTRAR = 0.35    # confianza minima para mostrar un objeto nuevo (teclas +/-)
CONF_YOLOE = 0.30      # YOLOE es mas ruidoso: umbral propio
IOU_NMS = 0.50
# Resolucion de entrada de YOLO. Medido con este proyecto: 640-960 reconoce mejor los objetos
# cercanos a la camara; 1280 solo ayuda con objetos muy pequenos/lejanos (y corto la llave).
IMGSZ_CAMARA = 640
IMGSZ_VIDEO = 960
IMGSZ_YOLOE = 640
TTA_VIDEO = True       # Test-Time Augmentation en video (mas preciso, ~2x mas lento)

# ── CLIP (refinamiento de etiqueta) ───────────────────────────────────────────
CLIP_PESO = 1.5             # cuanto pesa CLIP frente al detector al decidir la etiqueta
CLIP_INTERVALO_INICIAL = 6  # frames entre verificaciones mientras el objeto es nuevo
CLIP_INTERVALO = 30         # frames entre re-verificaciones despues
CLIP_LOTE = 16              # recortes por lote en la GPU

# Busqueda de "escena": objetos que YOLO no reconoce (p.ej. audifonos) se buscan con CLIP
ESCENA_INTERVALO_CAMARA = 8  # frames entre busquedas
ESCENA_INTERVALO_VIDEO = 4
ESCENA_MIN_PROB = 0.60       # prob. minima de CLIP para dibujar la caja
ESCENA_MAX_PROPUESTAS = 8

# ── ROBOFLOW ──────────────────────────────────────────────────────────────────
# La API key NUNCA va en el codigo. Se lee de (en este orden):
#   1. la variable de entorno ROBOFLOW_API_KEY
#   2. el archivo roboflow_key.txt junto a este script (ignorado por git)
# Sin key, la app funciona igual pero sin el modelo de Roboflow.
def _leer_api_key():
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    archivo = BASE_DIR / "roboflow_key.txt"
    if not key and archivo.exists():
        key = archivo.read_text(encoding="utf-8").strip()
    return key


ROBOFLOW_API_KEY = _leer_api_key()
ROBOFLOW_MODELO = "llave-auto/1"
ROBOFLOW_CONF = 40          # % minimo (antes 5%: generaba muchas falsas "llaves")
ROBOFLOW_INTERVALO = 0.35   # segundos minimos entre consultas
ROBOFLOW_TIMEOUT = 6

# ── CAMARA ────────────────────────────────────────────────────────────────────
CAMARA = 0
CAM_ANCHO, CAM_ALTO, CAM_FPS = 1280, 720, 30

# ── INTERFAZ ──────────────────────────────────────────────────────────────────
MOSTRAR_RASTRO = False      # tecla t (linea que sigue el movimiento de cada objeto)
MAX_RASTRO = 40
ALTO_MAX_VENTANA = 900      # la ventana se escala para caber en pantalla (videos verticales)
