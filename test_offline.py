"""
Offline-Test: Ball-Erkennung/Tracking/Regel-Events an einer Videodatei prüfen
================================================================================

Spielt eine vorhandene Videodatei (egal welche Qualität) Frame für Frame
durch die gleiche Pipeline wie `pool_referee.py` (BallDetector -> BallTracker
-> ShotEventDetector) und zeichnet die Ergebnisse zur Kontrolle ein:

- erkannte Kugeln als Kreise (mit ID + vermuteter Kugelart)
- Bandenkontakt-Markierung
- Fenster-Konsole: ausgelöste Ereignisse (Versenkt, Erstkontakt, ...)

So kannst du VOR dem Einsatz der echten OV9281 am Rechner prüfen, ob die
Parameter (Ballradius, Kontaktschwelle, Hintergrundmodell) grob passen,
und die Pipeline debuggen, ohne live am Tisch zu stehen.

Voraussetzung: `pool_referee.py` liegt im selben Verzeichnis (wird importiert).

Aufruf-Beispiele
----------------
    # Mit Kalibrierdatei aus calibrate_table.py:
    python3 test_offline.py --video aufnahme.mp4 \
        --calibration table_calibration.json --ball-radius-px 14

    # Ohne Kalibrierdatei, Ecken manuell angeben (x1,y1 x2,y2 x3,y3 x4,y4):
    python3 test_offline.py --video aufnahme.mp4 \
        --corners 50,40 1230,40 1230,760 50,760 --ball-radius-px 14

    # Playback verlangsamen/anhalten testen:
    #   Leertaste = Pause/Weiter, '.'/',' = 1 Frame vor/zurueck, 'q' = Ende

Ausgabe (optional): mit --save-annotated wird ein annotiertes Video
gespeichert, das du dir später ohne GUI ansehen kannst.
"""

import argparse
import json
import sys
import time

import cv2
import numpy as np

from pool_referee import (
    TableConfig, TableCalibration, BallDetector, BallTracker,
    ShotEventDetector, NotificationManager, RuleEngine, GamePhase,
)


KIND_COLORS = {
    "cue": (255, 255, 255),
    "eight": (0, 0, 0),
    "solid": (0, 165, 255),
    "stripe": (255, 0, 255),
    "unknown": (0, 255, 255),
}


def parse_corners(args) -> list:
    if args.calibration and args.corners:
        print(f"HINWEIS: sowohl --calibration als auch --corners angegeben - "
              f"--calibration ('{args.calibration}') wird verwendet, --corners "
              f"wird ignoriert.")
    if args.calibration:
        try:
            with open(args.calibration, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"FEHLER: Kalibrierdatei nicht gefunden: '{args.calibration}'.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"FEHLER: Kalibrierdatei '{args.calibration}' ist kein gültiges "
                  f"JSON (vermutlich beschädigt oder falsche Datei): {e}")
            sys.exit(1)
        pts = [tuple(p) for p in data["corners_px"]]
        if len(pts) != 4:
            print(f"FEHLER: Kalibrierdatei enthält {len(pts)} Punkte statt der "
                  f"erwarteten 4 (corners_px).")
            sys.exit(1)
        return pts
    if args.corners:
        pts = []
        for token in args.corners:
            parts = token.split(",")
            if len(parts) != 2:
                print(f"FEHLER: --corners Punkt '{token}' hat falsches Format. "
                      f"Erwartet wird 'x,y' (z.B. '150,80'), mit genau einem Komma.")
                sys.exit(1)
            x_str, y_str = parts
            try:
                pts.append((float(x_str), float(y_str)))
            except ValueError:
                print(f"FEHLER: --corners Punkt '{token}': 'x' und 'y' müssen Zahlen sein.")
                sys.exit(1)
        if len(pts) != 4:
            print("FEHLER: --corners braucht genau 4 Punkte.")
            sys.exit(1)
        return pts
    print("FEHLER: entweder --calibration oder --corners angeben.")
    sys.exit(1)


def draw_annotations(frame_topdown, tracker: BallTracker, events_summary: str,
                      ball_radius_px: float):
    img = frame_topdown.copy()
    for b in tracker.balls.values():
        if b.pocketed or b.off_table:
            continue
        color = KIND_COLORS.get(b.kind, (0, 255, 0))
        center = (int(b.pos[0]), int(b.pos[1]))
        # Kreisgroesse = konfigurierter ball_radius_px, NICHT fix -> so laesst
        # sich visuell pruefen, ob der Wert zur tatsaechlichen Kugelgroesse passt.
        cv2.circle(img, center, int(ball_radius_px), color, 2)
        cv2.putText(img, f"{b.id}:{b.kind}", (center[0] - 20, center[1] - int(ball_radius_px) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        speed = np.linalg.norm(b.vel)
        if speed > 1.0:
            end = (int(b.pos[0] + b.vel[0] * 3), int(b.pos[1] + b.vel[1] * 3))
            cv2.arrowedLine(img, center, end, color, 1, tipLength=0.3)

    cv2.rectangle(img, (0, img.shape[0] - 26), (img.shape[1], img.shape[0]),
                  (30, 30, 30), -1)
    cv2.putText(img, events_summary, (6, img.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser(description="Offline-Test der Erkennungs-Pipeline")
    parser.add_argument("--video", required=True, help="Pfad zur Testvideodatei")
    parser.add_argument("--calibration", type=str, default=None,
                         help="JSON-Datei aus calibrate_table.py")
    parser.add_argument("--corners", nargs=4, default=None,
                         help="4 Punkte manuell, Format x,y (falls keine --calibration)")
    parser.add_argument("--out-w", type=int, default=1000,
                         help="Breite der entzerrten Top-Down-Ansicht. Skaliert "
                              "stark auf die Erkennungsgeschwindigkeit, siehe "
                              "pool_referee.py --help fuer Messwerte.")
    parser.add_argument("--out-h", type=int, default=500)
    parser.add_argument("--table-w-mm", type=float, default=2540.0)
    parser.add_argument("--table-h-mm", type=float, default=1270.0)
    parser.add_argument("--ball-radius-px", type=float, required=True,
                         help="Ballradius IN DER ENTZERRTEN TOP-DOWN-ANSICHT (Pixel)")
    parser.add_argument("--background-frames", type=int, default=20,
                         help="Anzahl Frames am Videoanfang zum Hintergrund-Lernen "
                              "(Tisch sollte dort moeglichst leer/unbewegt sein)")
    parser.add_argument("--max-missing-frames", type=int, default=30,
                         help="Wie viele Frames eine Kugel fehlen darf, bevor sie als "
                              "versenkt/vom Tisch gilt. Kontakt zweier Kugeln kann durch "
                              "Video-Kompression kurzzeitig (teils >60 Frames bei 30fps "
                              "im Test!) zu KEINER Erkennung fuehren. Eher grosszuegig "
                              "waehlen (30-60) und hier am eigenen Video verifizieren.")
    parser.add_argument("--match-dist-px", type=float, default=40.0,
                         help="Max. Distanz (Pixel), damit eine Erkennung noch "
                              "derselben Kugel zugeordnet wird. Bei schnellen Stoessen "
                              "kann eine Kugel in mehrere IDs fragmentieren, wenn dieser "
                              "Wert zu klein ist -- im Fenster auf haeufig wechselnde "
                              "IDs achten und ggf. erhoehen.")
    parser.add_argument("--save-annotated", type=str, default=None,
                         help="Pfad, um annotiertes Video zu speichern (optional)")
    parser.add_argument("--no-display", action="store_true",
                         help="Kein GUI-Fenster oeffnen (z.B. auf Server ohne Display)")
    args = parser.parse_args()

    if args.ball_radius_px <= 0:
        print(f"FEHLER: --ball-radius-px muss positiv sein (war: {args.ball_radius_px}).")
        sys.exit(1)
    if args.background_frames <= 0:
        print(f"FEHLER: --background-frames muss positiv sein (war: {args.background_frames}). "
              f"Ohne echte Hintergrund-Lernphase wird nichts korrekt erkannt.")
        sys.exit(1)
    if args.max_missing_frames <= 0:
        print(f"FEHLER: --max-missing-frames muss positiv sein (war: {args.max_missing_frames}). "
              f"Bei 0 oder negativ wuerde JEDE Kugel sofort als versenkt/vom Tisch gelten.")
        sys.exit(1)
    if args.match_dist_px <= 0:
        print(f"FEHLER: --match-dist-px muss positiv sein (war: {args.match_dist_px}). "
              f"Bei 0 oder negativ koennte niemals eine Kugel wiedererkannt werden.")
        sys.exit(1)

    corners = parse_corners(args)
    table_config = TableConfig(corners_px=corners, table_w_mm=args.table_w_mm,
                                table_h_mm=args.table_h_mm)
    calib = TableCalibration(table_config, out_w=args.out_w, out_h=args.out_h)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"FEHLER: Video konnte nicht geoeffnet werden: {args.video}")
        sys.exit(1)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector = BallDetector(ball_radius_px=args.ball_radius_px)
    tracker = BallTracker(max_missing_frames=args.max_missing_frames,
                          match_dist_px=args.match_dist_px)
    shot_detector = ShotEventDetector(calib, table_config,
                                      ball_radius_px=args.ball_radius_px)
    notifier = NotificationManager()
    rules = RuleEngine(notifier)

    # -- Hintergrund aus den ersten N Frames lernen --
    print(f"Lerne Hintergrund aus den ersten {args.background_frames} Frames...")
    learned = 0
    while learned < args.background_frames:
        ok, frame = cap.read()
        if not ok:
            print("FEHLER: Video zu kurz fuer Hintergrund-Lernphase.")
            sys.exit(1)
        topdown = calib.warp(frame)
        detector.bg_subtractor.apply(topdown, learningRate=0.5)
        learned += 1
    print("Hintergrund gelernt. Starte Auswertung...\n")

    if not args.no_display:
        print("Tastenbelegung im Fenster:")
        print("  Leertaste = Pause/Weiter   . = 1 Frame vor (im Pausemodus)")
        print("  1-6 = Tasche fuer die angesagte 8 waehlen (vor dem Stoss auf die 8)")
        print("  c   = Taschenansage zuruecksetzen   q = Beenden\n")

    writer = None
    if args.save_annotated:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_annotated, fourcc, src_fps,
                                  (args.out_w, args.out_h))

    shot_active = False
    last_time = time.time()
    paused = False
    step_once = False
    events_summary = "Warte auf Bewegung..."
    frame_idx = args.background_frames

    while True:
        if not paused or step_once:
            ok, frame = cap.read()
            if not ok:
                print("Video zu Ende.")
                break
            frame_idx += 1
            step_once = False

            now = time.time()
            dt = 1.0 / src_fps if src_fps > 0 else (now - last_time)
            last_time = now

            topdown = calib.warp(frame)
            detections = detector.detect(topdown)
            tracker.update(detections, detector.classify_group, dt)

            moving = not tracker.all_stationary(px_to_mm=calib.scale_x)

            if moving and not shot_active:
                shot_active = True
                shot_detector.reset()
                print(f"[Frame {frame_idx}] Stoss beginnt (Bewegung erkannt).")

            if shot_active:
                shot_detector.process_frame(tracker)

            if shot_active and not moving:
                shot_active = False
                ev = shot_detector.events
                print(f"[Frame {frame_idx}] Stoss beendet. "
                      f"Erstkontakt={ev.first_contact_kind}, "
                      f"Banden={len(ev.cushions_touched)}, "
                      f"Versenkt={[k for _, k, _ in ev.pocketed]}, "
                      f"Scratch={ev.cue_pocketed}")
                rules.evaluate_shot(ev)
                rules.called_pocket_for_eight = None  # Ansage gilt nur fuer 1 Stoss
                events_summary = (
                    f"Phase={rules.phase.name} | Dran={rules.current_player.name}"
                )
                if rules.phase == GamePhase.GAME_OVER:
                    print(f"\n>>> SPIELENDE: {rules.winner} gewinnt <<<")

            else:
                n_balls = len([b for b in tracker.balls.values()
                               if not b.pocketed and not b.off_table])
                pocket_info = (rules.called_pocket_for_eight + 1
                                if rules.called_pocket_for_eight is not None else "-")
                events_summary = (f"Kugeln erkannt: {n_balls} | Bewegung: {moving} | "
                                   f"Angesagte Tasche(8)={pocket_info}")

        annotated = draw_annotations(topdown, tracker, events_summary,
                                     args.ball_radius_px)

        if writer is not None:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Offline-Test: Ball-Erkennung", annotated)
            wait_ms = 0 if paused else 1
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'):
                break
            elif key == 32:  # Leertaste: Pause umschalten
                paused = not paused
            elif key == ord('.') and paused:
                step_once = True  # naechster Schleifendurchlauf verarbeitet 1 Frame, bleibt danach pausiert
            elif key == ord('c'):
                rules.called_pocket_for_eight = None
            elif ord('1') <= key <= ord('6'):
                idx = key - ord('1')
                if 0 <= idx < len(table_config.pocket_positions_mm):
                    rules.called_pocket_for_eight = idx
                    print(f"Angesagte Tasche fuer die 8: Nr. {idx + 1}")

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print("\nFertig.")
    if rules.winner:
        print(f"Ergebnis: {rules.winner} hat gewonnen.")
    else:
        print(f"Kein Spielende erreicht. Letzte Phase: {rules.phase.name}")


if __name__ == "__main__":
    main()
