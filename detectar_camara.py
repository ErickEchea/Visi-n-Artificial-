"""
Vision IA - Camara en tiempo real
YOLO11x + YOLOE (objetos extra) + CLIP (verificacion) + Roboflow (llave de auto)

Uso:  python detectar_camara.py [--camara N]
Controles: q/ESC salir · s captura · +/- confianza · t rastros · e YOLOE · c CLIP
           r Roboflow · p contraste · h ayuda
"""
import argparse
import sys
import threading
import time

import cv2

import config as C
from vision_core import Motor, Ventana, consola_utf8

TITULO = "Vision IA - Camara"


class Camara:
    """Lee la camara en un hilo aparte y entrega siempre el frame mas reciente
    (sin esto, el retraso crece porque los frames se acumulan en el buffer)."""

    def __init__(self, indice):
        self.cap = self._abrir(indice)
        self.frame = None
        self.num = 0
        self.vivo = self.cap is not None
        self.cond = threading.Condition()
        if self.vivo:
            threading.Thread(target=self._leer, daemon=True, name="camara").start()

    @staticmethod
    def _abrir(indice):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        indices = [indice] + [i for i in range(4) if i != indice]
        for i in indices:
            for be in backends:
                cap = cv2.VideoCapture(i, be)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAM_ANCHO)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAM_ALTO)
                cap.set(cv2.CAP_PROP_FPS, C.CAM_FPS)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok = False
                for _ in range(10):  # la camara puede tardar en entregar el primer frame
                    ok, _f = cap.read()
                    if ok:
                        break
                    time.sleep(0.05)
                if ok:
                    if i != indice:
                        print(f"[AVISO] La camara {indice} no respondio; usando la camara {i}.")
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"Camara {i} abierta: {w}x{h}")
                    return cap
                cap.release()
        return None

    def _leer(self):
        fallos = 0
        while self.vivo:
            ok, f = self.cap.read()
            if not ok:
                fallos += 1
                if fallos > 60:
                    print("ERROR: la camara dejo de enviar imagen (¿desconectada o en uso por otra app?).")
                    self.vivo = False
                time.sleep(0.03)
                continue
            fallos = 0
            with self.cond:
                self.frame, self.num = f, self.num + 1
                self.cond.notify_all()
        with self.cond:
            self.cond.notify_all()

    def leer(self, ultimo, timeout=2.0):
        """Espera un frame nuevo (num > ultimo)."""
        with self.cond:
            self.cond.wait_for(lambda: self.num > ultimo or not self.vivo, timeout=timeout)
            return self.frame, self.num

    def cerrar(self):
        self.vivo = False
        time.sleep(0.05)
        if self.cap is not None:
            self.cap.release()


def main():
    consola_utf8()
    ap = argparse.ArgumentParser(description="Deteccion de objetos con la camara")
    ap.add_argument("--camara", type=int, default=C.CAMARA, help="indice de la camara (0, 1, ...)")
    args = ap.parse_args()

    motor = Motor("camara")
    cam = Camara(args.camara)
    if cam.cap is None:
        print("ERROR: no se pudo abrir ninguna camara. Revisa que este conectada y que")
        print("       ninguna otra aplicacion (Zoom, Teams, navegador) la este usando.")
        motor.cerrar()
        return 1

    print("Controles: q/ESC salir · s captura · +/- confianza · h ayuda")
    print("[IA] = confirmado por CLIP   [RF] = confirmado por el modelo Roboflow\n")
    ventana = Ventana(TITULO)
    ultimo = 0
    try:
        while cam.vivo:
            frame, num = cam.leer(ultimo)
            if frame is None or num == ultimo:
                continue
            ultimo = num
            salida = motor.procesar(frame)
            motor.dibujar_panel(salida)
            ventana.mostrar(salida)
            k = cv2.waitKey(1) & 0xFF
            if (k != 255 and not motor.tecla(k, salida)) or ventana.cerrada():
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.cerrar()
        motor.cerrar()
        cv2.destroyAllWindows()
    print("Camara cerrada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
