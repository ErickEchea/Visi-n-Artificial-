"""
Debug del modelo Roboflow "llave-auto": muestra TODAS sus predicciones (confianza >= 1%).

Uso:  python debug_modelo.py                -> toma una foto con la camara
      python debug_modelo.py imagen.jpg     -> analiza una imagen
"""
import base64
import sys
import time

import cv2
import requests

import config as C

WARM_UP = 5  # frames descartados para que la camara ajuste exposicion


def foto_camara():
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(C.CAMARA, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(C.CAMARA)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la camara.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAM_ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAM_ALTO)
    frame = None
    for _ in range(WARM_UP):
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.05)
    cap.release()
    if frame is None:
        print("ERROR: La camara no entrego ninguna imagen.")
    return frame


def main():
    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
        if frame is None:
            print(f"ERROR: no se pudo leer la imagen {sys.argv[1]}")
            return 1
    else:
        frame = foto_camara()
        if frame is None:
            return 1
        C.CARPETA_CAPTURAS.mkdir(exist_ok=True)
        ruta = C.CARPETA_CAPTURAS / "debug_frame.jpg"
        cv2.imwrite(str(ruta), frame)
        print(f"Frame guardado en {ruta}")

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Consultando modelo {C.ROBOFLOW_MODELO} con confianza >= 1%...")
    try:
        r = requests.post(
            f"https://detect.roboflow.com/{C.ROBOFLOW_MODELO}",
            params={"api_key": C.ROBOFLOW_API_KEY, "confidence": 1, "overlap": 30},
            data=base64.b64encode(buf).decode("ascii"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=C.ROBOFLOW_TIMEOUT + 4,
        )
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.ConnectionError:
        print("ERROR: Sin conexion a internet o Roboflow no responde.")
        return 1
    except requests.exceptions.Timeout:
        print("ERROR: Roboflow tardo demasiado (timeout).")
        return 1
    except requests.exceptions.HTTPError as e:
        extra = " (API key invalida o sin permiso)" if r.status_code in (401, 403) else ""
        print(f"ERROR HTTP {r.status_code}{extra}: {e}")
        return 1
    except ValueError:
        print("ERROR: La respuesta de Roboflow no es JSON valido.")
        return 1

    preds = sorted(data.get("predictions", []), key=lambda p: p["confidence"], reverse=True)
    print(f"\nPredicciones con confianza >= 1%: {len(preds)}  "
          f"(la app muestra solo las >= {C.ROBOFLOW_CONF}%)")
    for p in preds:
        marca = "  <- se mostraria" if p["confidence"] * 100 >= C.ROBOFLOW_CONF else ""
        print(f"  clase={p['class']:<14} confianza={p['confidence'] * 100:5.1f}%  "
              f"cx={p['x']:.0f} cy={p['y']:.0f} w={p['width']:.0f} h={p['height']:.0f}{marca}")
    if not preds:
        print("  (ninguna prediccion)\n\nPosibles causas:")
        print("  - La llave no esta visible o esta muy lejos")
        print("  - Poca iluminacion")
        print("  - El modelo se entreno con pocas imagenes (50) de una sola llave")
    return 0


if __name__ == "__main__":
    sys.exit(main())
