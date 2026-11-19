"""
Vision IA - Deteccion en video (maxima precision, guarda el resultado)
YOLO11x + TTA + YOLOE (objetos extra) + CLIP (verificacion) + Roboflow (llave de auto)

Uso:  python detectar_video.py [ruta_video] [--sin-tta] [--sin-ventana] [--max-frames N]
      Sin ruta, usa el primer .mp4/.mov/.avi de la carpeta del proyecto.
Controles: q/ESC salir · ESPACIO pausar · s captura · +/- confianza · h ayuda
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

import config as C
from vision_core import Motor, Ventana, consola_utf8

TITULO = "Vision IA - Video"
EXTENSIONES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def buscar_video():
    for p in sorted(C.BASE_DIR.iterdir()):
        if p.suffix.lower() in EXTENSIONES and not p.stem.endswith("_detectado"):
            return p
    return None


def main():
    consola_utf8()
    ap = argparse.ArgumentParser(description="Deteccion de objetos en un video")
    ap.add_argument("video", nargs="?", help="ruta del video (opcional)")
    ap.add_argument("--sin-tta", action="store_true", help="mas rapido, un poco menos preciso")
    ap.add_argument("--sin-ventana", action="store_true", help="procesar sin mostrar la ventana")
    ap.add_argument("--max-frames", type=int, default=0, help="procesar solo los primeros N frames")
    args = ap.parse_args()

    ruta = Path(args.video).expanduser() if args.video else buscar_video()
    if ruta is None or not ruta.exists():
        print(f"ERROR: no se encontro el video{f': {ruta}' if ruta else ''}.")
        print(f"       Copia un video a {C.BASE_DIR} o pasa la ruta: python detectar_video.py mi_video.mp4")
        return 1

    video = cv2.VideoCapture(str(ruta))
    if not video.isOpened():
        print(f"ERROR: OpenCV no pudo abrir el video: {ruta}")
        return 1

    fps = video.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and 1 <= fps <= 240 else 30.0
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if args.max_frames:
        total = min(total, args.max_frames) if total else args.max_frames

    ok, frame = video.read()
    if not ok:
        print("ERROR: el video esta vacio o daniado.")
        return 1
    alto, ancho = frame.shape[:2]

    motor = Motor("video", tta=False if args.sin_tta else None)

    C.CARPETA_RESULTADOS.mkdir(exist_ok=True)
    salida_ruta = C.CARPETA_RESULTADOS / f"{ruta.stem}_detectado.mp4"
    writer = cv2.VideoWriter(str(salida_ruta), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))
    if not writer.isOpened():
        print(f"[AVISO] No se pudo crear {salida_ruta}; el resultado no se guardara.")
        writer = None

    print(f"Video: {ruta.name}  {ancho}x{alto} @ {fps:.1f} fps  |  {total or '?'} frames")
    print(f"TTA: {'ON' if motor.tta else 'OFF'}   Salida: {salida_ruta}")
    print("Controles: q/ESC salir · ESPACIO pausar · s captura · h ayuda\n")

    ventana = None if args.sin_ventana else Ventana(TITULO)
    n = 0
    pausado = False
    anotado = salida = None
    t_inicio = time.time()
    ultimo_reporte = -1

    def componer():
        """Frame anotado + panel + barra de progreso (se rehace en pausa al pulsar teclas)."""
        vista = anotado.copy()
        prog = n / total if total else 0
        motor.dibujar_panel(vista, extra=[f"Frame {n}/{total or '?'}  ({prog * 100:.1f}%)"
                                          f"   TTA {'ON' if motor.tta else 'OFF'}"])
        if total:
            cv2.rectangle(vista, (2, alto - 8), (2 + int((ancho - 4) * prog), alto - 2), (0, 210, 80), -1)
        return vista

    try:
        while True:
            if not pausado:
                if n > 0:
                    ok, frame = video.read()
                    if not ok:
                        break
                n += 1
                anotado = motor.procesar(frame)
                salida = componer()
                prog = n / total if total else 0
                if writer:
                    writer.write(salida)
                pct = int(prog * 20)
                if total and pct != ultimo_reporte:
                    ultimo_reporte = pct
                    trans = time.time() - t_inicio
                    resta = trans / n * (total - n)
                    print(f"  {prog * 100:5.1f}%  frame {n}/{total}  ~{resta:.0f}s restantes", flush=True)
                if args.max_frames and n >= args.max_frames:
                    break

            if ventana:
                ventana.mostrar(salida)
                k = cv2.waitKey(30 if pausado else 1) & 0xFF
                if k == ord(" "):
                    pausado = not pausado
                    print("PAUSADO" if pausado else "Continuando")
                elif k != 255:
                    if not motor.tecla(k, salida):
                        break
                    if pausado:  # refrescar panel/ayuda mientras esta en pausa
                        salida = componer()
                if ventana.cerrada():
                    break
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    finally:
        video.release()
        if writer:
            writer.release()
        motor.cerrar()
        cv2.destroyAllWindows()

    dur = time.time() - t_inicio
    print(f"\nProcesados {n} frames en {dur:.0f}s ({n / max(dur, 1e-6):.1f} fps)")
    if writer:
        print(f"Video guardado: {salida_ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
