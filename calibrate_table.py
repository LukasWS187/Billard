"""
Kalibrier-Tool: Tischecken per Mausklick bestimmen
=====================================================

Zeigt ein Bild der Spielfläche und lässt dich die 4 Ecken anklicken.
Das Ergebnis wird als JSON gespeichert und kann direkt in
`pool_referee.py` (TableConfig.corners_px) verwendet werden.

DREI BILDQUELLEN (nur eine wird benötigt)
-------------------------------------------
1. --device N   : Live-Aufnahme von einer angeschlossenen Kamera (V4L2/USB).
2. --picamera2   : Live-Aufnahme über Raspberry-Pi-CSI (picamera2).
3. --image PFAD  : Vorhandenes Einzelbild (z.B. Handyfoto von der exakt
                    gleichen Position/Höhe/Winkel, an der die OV9281 später
                    montiert wird, oder ein zuvor exportierter Kameraframe).
4. --video PFAD  : Vorhandene Videodatei. Mit den Tasten '.'/',' kannst du
                    vor-/zurückspulen, um ein geeignetes Standbild zu wählen.

Du brauchst also die tatsächliche OV9281 NICHT angeschlossen zu haben,
solange du ein Bild aus exakt der späteren Kameraperspektive hast.
Wichtig: Nur die Perspektive (Position/Höhe/Neigewinkel) muss stimmen,
nicht die Bildquelle.

Bedienung
---------
- Bild öffnet sich in einem Fenster.
- Klicke der Reihe nach:
    1. oben links
    2. oben rechts
    3. unten rechts
    4. unten links
  (Reihenfolge ist wichtig! Es sind die Ecken der SPIELFLÄCHE, nicht der
   Bande/des Holzrahmens.)
- 'r'     -> letzten Punkt zurücknehmen
- 's'     -> Kalibrierung speichern (erst aktiv, wenn 4 Punkte gesetzt sind)
- 'n'     -> neues Standbild holen (bei --device/--picamera2) bzw. ohne
              Wirkung bei --image
- '.' / ',' -> bei --video: 10 Frames vor / zurück springen
- 'q'     -> Abbrechen ohne Speichern

Nach dem Speichern zusätzlich eine Vorschau der entzerrten Top-Down-Ansicht,
damit du die Ecken visuell prüfen kannst, bevor du sie übernimmst.

Aufruf-Beispiele
----------------
    # Live, USB/V4L2:
    python3 calibrate_table.py --device 0 --width 1280 --height 800 \
        --out table_calibration.json

    # Live, Raspberry-Pi-CSI:
    python3 calibrate_table.py --picamera2 --out table_calibration.json

    # Ohne angeschlossene Kamera, mit vorhandenem Foto/Frame:
    python3 calibrate_table.py --image tisch_foto.jpg \
        --out table_calibration.json

    # Ohne angeschlossene Kamera, aus einer Videoaufnahme:
    python3 calibrate_table.py --video aufnahme.mp4 \
        --out table_calibration.json
"""

import argparse
import json
import sys

import cv2
import numpy as np


POINT_LABELS = ["oben links", "oben rechts", "unten rechts", "unten links"]
WINDOW = "Kalibrierung - Tischecken anklicken"


class Calibrator:
    def __init__(self):
        self.points: list[tuple[int, int]] = []
        self.frame = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"Punkt {len(self.points)} ({POINT_LABELS[len(self.points)-1]}): ({x}, {y})")

    def draw_overlay(self, frame):
        img = frame.copy()
        for i, p in enumerate(self.points):
            cv2.circle(img, p, 6, (0, 0, 255), -1)
            cv2.putText(img, f"{i+1}:{POINT_LABELS[i]}", (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                cv2.line(img, self.points[i], self.points[i + 1], (0, 255, 0), 1)
        if len(self.points) == 4:
            cv2.line(img, self.points[3], self.points[0], (0, 255, 0), 1)

        hint = "Naechster Punkt: " + (POINT_LABELS[len(self.points)]
                                       if len(self.points) < 4 else "fertig (s = speichern)")
        cv2.putText(img, hint, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "r=zuruck  n=neues Bild  s=speichern  q=abbrechen",
                    (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1, cv2.LINE_AA)
        return img


def grab_frame(args):
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"FEHLER: Bild konnte nicht geladen werden: {args.image}")
            sys.exit(1)
        return frame

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"FEHLER: Video konnte nicht geoeffnet werden: {args.video}")
            sys.exit(1)
        raw_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total = int(raw_count) if raw_count == raw_count and raw_count > 0 else 0  # raw_count!=raw_count erkennt NaN
        idx = min(total // 2, total - 1) if total > 0 else 0  # mittleren Frame als Start
        frame = video_frame_picker(cap, idx)
        cap.release()
        return frame

    if args.picamera2:
        if args.width <= 0 or args.height <= 0:
            print(f"FEHLER: --width/--height muessen positiv sein (waren: "
                  f"{args.width}/{args.height}).")
            sys.exit(1)
        try:
            from picamera2 import Picamera2  # type: ignore
            cam = Picamera2()
            config = cam.create_still_configuration(
                main={"size": (args.width, args.height), "format": "RGB888"}
            )
            cam.configure(config)
            cam.start()
            frame = cam.capture_array()
            cam.stop()
            return frame
        except ImportError:
            print("FEHLER: picamera2-Bibliothek nicht installiert. Installieren "
                  "mit: pip install picamera2 --break-system-packages (oder ueber "
                  "apt: sudo apt install python3-picamera2).")
            sys.exit(1)
        except Exception as e:
            print(f"FEHLER: CSI-Kamera konnte nicht angesprochen werden: {e}. Ist "
                  f"sie korrekt angeschlossen und in raspi-config aktiviert?")
            sys.exit(1)

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("FEHLER: Kamera konnte nicht geoeffnet werden.")
        sys.exit(1)
    # ein paar Frames verwerfen, bis Belichtung sich eingependelt hat
    frame = None
    for _ in range(10):
        ok, frame = cap.read()
        if not ok:
            print("FEHLER: Kein Bild von der Kamera erhalten.")
            sys.exit(1)
    cap.release()
    return frame


def video_frame_picker(cap: cv2.VideoCapture, start_idx: int):
    """Lässt den Nutzer mit '.'/',' durch ein Video scrollen, um ein
    geeignetes Standbild für die Kalibrierung auszuwählen. Bestätigt mit
    Leertaste/Enter.
    """
    raw_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total = int(raw_count) if raw_count == raw_count and raw_count > 0 else 0  # raw_count!=raw_count erkennt NaN
    idx = max(0, min(start_idx, total - 1)) if total > 0 else 0
    win = "Video-Standbild waehlen (,/. = zurueck/vor, Leertaste = uebernehmen)"
    cv2.namedWindow(win)
    frame = None
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        display = frame.copy()
        cv2.putText(display, f"Frame {idx}/{max(total - 1, 0)}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, display)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('.'):
            idx = min(idx + 10, max(total - 1, 0))
        elif key == ord(','):
            idx = max(idx - 10, 0)
        elif key in (32, 13):  # Leertaste oder Enter
            break
        elif key == ord('q'):
            print("Abgebrochen.")
            sys.exit(0)
    cv2.destroyWindow(win)
    if frame is None:
        print("FEHLER: Es konnte kein lesbares Frame aus dem Video geholt werden "
              "(evtl. beschaedigte Datei oder ungewoehnliches Format).")
        sys.exit(1)
    return frame


def preview_warp(frame, points, out_w=1000, out_h=500):
    src = np.array(points, dtype=np.float32)
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, H, (out_w, out_h))
    cv2.imshow("Vorschau: entzerrte Top-Down-Ansicht", warped)
    print("Vorschau geoeffnet. Beliebige Taste im Vorschaufenster druecken zum Schliessen.")
    cv2.waitKey(0)
    cv2.destroyWindow("Vorschau: entzerrte Top-Down-Ansicht")


def main():
    parser = argparse.ArgumentParser(description="Tischecken-Kalibrierung per Mausklick")
    parser.add_argument("--device", type=int, default=0, help="V4L2 Geraeteindex")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--picamera2", action="store_true",
                         help="Raspberry-Pi-CSI-Kamera statt V4L2/USB verwenden")
    parser.add_argument("--image", type=str, default=None,
                         help="Vorhandenes Einzelbild statt Live-Kamera verwenden")
    parser.add_argument("--video", type=str, default=None,
                         help="Vorhandene Videodatei statt Live-Kamera verwenden")
    parser.add_argument("--out", type=str, default="table_calibration.json")
    args = parser.parse_args()

    if args.image and args.video:
        print("FEHLER: --image und --video koennen nicht gleichzeitig genutzt werden.")
        sys.exit(1)

    calib = Calibrator()
    calib.frame = grab_frame(args)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, calib.on_mouse)

    while True:
        display = calib.draw_overlay(calib.frame)
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            print("Abgebrochen, nichts gespeichert.")
            break

        elif key == ord('r'):
            if calib.points:
                removed = calib.points.pop()
                print(f"Punkt zurueckgenommen: {removed}")

        elif key == ord('n'):
            if args.image:
                print("'n' hat bei --image keine Wirkung (festes Einzelbild).")
                continue
            print("Hole neues Standbild...")
            calib.frame = grab_frame(args)

        elif key == ord('s'):
            if len(calib.points) != 4:
                print(f"Es sind erst {len(calib.points)}/4 Punkte gesetzt.")
                continue
            # Sanity-Check 1: liegen zwei Punkte verdaechtig nah beieinander
            # (z.B. Doppelklick-Verwackler)? Das wuerde eine entartete,
            # unbrauchbare Perspektiv-Transformation erzeugen.
            too_close = False
            for i in range(4):
                for j in range(i + 1, 4):
                    dist = ((calib.points[i][0] - calib.points[j][0]) ** 2 +
                            (calib.points[i][1] - calib.points[j][1]) ** 2) ** 0.5
                    if dist < 15:
                        print(f"WARNUNG: Punkt {i+1} und Punkt {j+1} liegen nur "
                              f"{dist:.0f}px auseinander - vermutlich ein "
                              f"Klickfehler. Mit 'r' zuruecknehmen und neu setzen.")
                        too_close = True
            if too_close:
                continue

            pts_arr = np.array(calib.points, dtype=np.float32)
            shoelace_area = 0.0
            for i in range(4):
                x1, y1 = pts_arr[i]
                x2, y2 = pts_arr[(i + 1) % 4]
                shoelace_area += x1 * y2 - x2 * y1
            shoelace_area = abs(shoelace_area) / 2.0
            hull = cv2.convexHull(pts_arr)
            hull_area = cv2.contourArea(hull)

            # Sanity-Check 2b: Ist die Flaeche des Vierecks ueberhaupt
            # sinnvoll gross? Vier nahezu kollineare Punkte (z.B. alle 4 fast
            # auf einer Linie) fallen weder durch den Mindestabstand-Check
            # (Punkte koennen weit auseinander liegen) noch durch den
            # folgenden Bowtie-Check (beide Flaechen sind dann fast gleich
            # UND fast Null) - das Ergebnis waere aber ein komplett leeres,
            # unbrauchbares entzerrtes Bild (im Testprotokoll reproduziert:
            # 0 sichtbare Pixel). Deshalb zusaetzlich eine Mindestflaeche
            # relativ zur Bildgroesse verlangen.
            frame_area = calib.frame.shape[0] * calib.frame.shape[1]
            if hull_area < 0.01 * frame_area:
                print("WARNUNG: Die 4 Punkte umschliessen eine verdaechtig "
                      "kleine/entartete Flaeche (liegen fast auf einer Linie). "
                      "Das wuerde ein leeres, unbrauchbares Ergebnisbild "
                      "erzeugen. Mit 'r' zuruecknehmen und die 4 echten "
                      "Tischecken neu anklicken.")
                continue

            # Sanity-Check 2: wurden zwei Ecken in FALSCHER REIHENFOLGE geklickt
            # (z.B. oben-rechts und unten-rechts vertauscht)? Das ergibt ein
            # sich selbst ueberkreuzendes "Bowtie"-Viereck statt eines echten
            # Rechtecks -- die Perspektiv-Transformation wird dann komplett
            # unbrauchbar (getestet: das Ergebnisbild wird schlicht schwarz).
            # Erkennung: die Flaeche des Vierecks IN DER GEKLICKTEN REIHENFOLGE
            # (Shoelace-Formel) muss der Flaeche der konvexen Huelle derselben
            # 4 Punkte entsprechen -- bei vertauschter Reihenfolge weichen
            # beide deutlich voneinander ab.
            if shoelace_area < 0.8 * hull_area:
                print("WARNUNG: Die 4 Punkte scheinen in FALSCHER REIHENFOLGE "
                      "geklickt zu sein (ueberkreuztes statt einfaches Viereck) "
                      "- vermutlich wurden zwei Ecken vertauscht. Das wuerde die "
                      "Kalibrierung unbrauchbar machen. Mit 'r' zuruecknehmen "
                      "und in der Reihenfolge oben-links -> oben-rechts -> "
                      "unten-rechts -> unten-links neu klicken.")
                continue

            preview_warp(calib.frame, calib.points)
            confirm = input("Kalibrierung so speichern? (j/n): ").strip().lower()
            if confirm == "j":
                data = {
                    "corners_px": calib.points,
                    "labels": POINT_LABELS,
                    "image_width": calib.frame.shape[1],
                    "image_height": calib.frame.shape[0],
                }
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"Gespeichert nach: {args.out}")
                print("In pool_referee.py verwenden:")
                print(f"  corners_px={calib.points}")
                break
            else:
                print("Nicht gespeichert, weiter kalibrieren (r = Punkt zuruecknehmen).")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
