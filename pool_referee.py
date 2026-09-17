"""
8-Ball Pool – Regelüberwachung per Kamera (OV9281)
====================================================

Dieses Programm überwacht eine 8-Ball-Partie über eine OV9281-Kamera
(Global Shutter, ideal für schnelle Kugelbewegungen ohne Rolling-Shutter-
Verzerrung) und meldet Regelverstöße (Fouls) automatisch.

WICHTIGE HINWEISE / GRENZEN
----------------------------
1. Die OV9281 gibt es als Mono- und als Farbvariante.
   - Mono: Gruppenklassifikation (Volle/Halbe) erfolgt heuristisch über
     Textur/Kantendichte auf der Kugeloberfläche (gestreift = mehr Kanten).
     Das ist fehleranfälliger als eine Farbklassifikation.
   - Farbe: In `BallDetector.classify_group()` stattdessen HSV-Histogramm
     verwenden (Codepfad ist vorbereitet, siehe TODO dort).
2. "Antritt mit weniger als einem Fuß am Boden" kann eine Kamera über dem
   Tisch grundsätzlich nicht erkennen. Dieser Foul-Typ ist als Platzhalter
   vorgesehen (`foot_foul`-Flag), du müsstest ihn über eine zusätzliche
   Kamera/einen Bodensensor liefern.
3. "Weiße Kugel springt vom Tisch" wird nur indirekt erkannt: Wenn ein Ball
   verschwindet, aber NICHT in der Nähe einer Tasche – das ist eine
   Heuristik und kann mit Verdeckungen (Hand, Queue) verwechselt werden.

BENÖTIGTE BIBLIOTHEKEN (pip install ...)
------------------------------------------
    opencv-python
    numpy
    scipy

Optional (empfohlen für robustere Erkennung statt Hough-Kreise):
    ultralytics   (YOLOv8, trainiertes Modell für Billardkugeln nötig)

Falls die Kamera per Raspberry-Pi-CSI angeschlossen ist, zusätzlich:
    picamera2

Kamera-Zugriff unter Linux/PC (USB-Board für OV9281, z.B. Arducam):
    cv2.VideoCapture(index, cv2.CAP_V4L2)
"""

from __future__ import annotations

import time
import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ============================================================================
# 1. KONFIGURATION
# ============================================================================

@dataclass
class TableConfig:
    """Geometrische Kalibrierdaten des Tisches.

    corners_px: 4 Eckpunkte des Spielfelds im Kamerabild, Reihenfolge
    [oben-links, oben-rechts, unten-rechts, unten-links] in Pixel-Koordinaten.
    Diese musst du einmalig durch Anklicken in einem Kalibrier-Tool ermitteln.
    """
    corners_px: list
    table_w_mm: float = 2540.0   # Spielfläche 8-Fuß-Tisch (Beispielwert)
    table_h_mm: float = 1270.0
    ball_diameter_mm: float = 57.15
    pocket_radius_mm: float = 65.0
    # Nur fuer nicht-standardmaessige Taschenanordnungen setzen (z.B. exotische
    # Tischformen). Im Normalfall None lassen - dann werden die 6 Taschen
    # automatisch aus table_w_mm/table_h_mm berechnet (siehe Property unten).
    pocket_positions_mm_override: Optional[list] = None

    @property
    def pocket_positions_mm(self) -> list:
        """6 Taschenpositionen in Tisch-Koordinaten (mm), Ursprung oben-links.
        Wird aus table_w_mm/table_h_mm abgeleitet, damit sie bei einem
        abweichend konfigurierten Tisch (z.B. 9-Fuss statt 8-Fuss, oder
        vertauschten Massen fuer eine Hochkant-Kalibrierung) automatisch
        mitskalieren, statt auf den 8-Fuss-Standardwerten stehen zu bleiben
        (frueherer Fehler: bei geaendertem table_w_mm/table_h_mm blieben die
        Taschen faelschlich an den alten, festen Koordinaten haengen).

        WICHTIG: Die beiden Seitentaschen liegen bei einem echten Billardtisch
        immer auf den LANGEN Banden, an deren Mittelpunkt - NIE auf den
        kurzen Banden. Welche der beiden Achsen (table_w_mm oder table_h_mm)
        die lange ist, haengt davon ab, wie kalibriert wurde (Querformat:
        table_w_mm ist die Laenge; Hochformat: table_h_mm ist die Laenge) -
        das wird hier automatisch anhand des groesseren Wertes erkannt,
        statt table_w_mm immer als Laengsachse anzunehmen (frueherer Fehler:
        bei einer Hochformat-Kalibrierung landeten die Seitentaschen dadurch
        faelschlich auf den kurzen statt den langen Banden)."""
        if self.pocket_positions_mm_override is not None:
            return self.pocket_positions_mm_override
        w, h = self.table_w_mm, self.table_h_mm
        if w >= h:
            # Querformat: w ist die Laengsachse, Seitentaschen mittig auf
            # der oberen/unteren (kurzen) Bandenrichtung ragend.
            overhang = h * 0.0157
            return [
                (0, 0), (w / 2, -overhang), (w, 0),
                (0, h), (w / 2, h + overhang), (w, h),
            ]
        else:
            # Hochformat: h ist die Laengsachse, Seitentaschen mittig auf
            # der linken/rechten (kurzen) Bandenrichtung ragend.
            overhang = w * 0.0157
            return [
                (0, 0), (-overhang, h / 2), (0, h),
                (w, 0), (w + overhang, h / 2), (w, h),
            ]


# ============================================================================
# 2. KAMERA-ERFASSUNG (OV9281)
# ============================================================================

class OV9281Capture:
    """Kapselt den Zugriff auf die OV9281-Kamera.

    use_picamera2=True  -> Raspberry Pi CSI-Anschluss (empfohlen für RPi)
    use_picamera2=False -> USB/V4L2 (z.B. Arducam USB-Board an PC/Linux)
    """

    def __init__(self, device_index: int = 0, width: int = 1280,
                 height: int = 800, fps: int = 120,
                 use_picamera2: bool = False):
        self.use_picamera2 = use_picamera2
        if use_picamera2:
            try:
                from picamera2 import Picamera2  # type: ignore
                self.cam = Picamera2()
                config = self.cam.create_video_configuration(
                    main={"size": (width, height), "format": "RGB888"},
                    controls={"FrameRate": fps},
                )
                self.cam.configure(config)
                self.cam.start()
            except ImportError:
                raise RuntimeError(
                    "picamera2-Bibliothek nicht installiert. Auf dem "
                    "Raspberry Pi installieren mit: pip install picamera2 "
                    "--break-system-packages (oder ueber apt: "
                    "sudo apt install python3-picamera2)."
                )
            except Exception as e:
                raise RuntimeError(
                    f"CSI-Kamera konnte nicht gestartet werden: {e}. Ist die "
                    f"Kamera korrekt am Raspberry Pi angeschlossen und in "
                    f"raspi-config aktiviert?"
                )
        else:
            self.cam = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
            self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cam.set(cv2.CAP_PROP_FPS, fps)
            if not self.cam.isOpened():
                raise RuntimeError(
                    "Kamera konnte nicht geöffnet werden. Geräteindex/"
                    "Treiber prüfen (v4l2-ctl --list-devices)."
                )

    def read(self) -> Optional[np.ndarray]:
        if self.use_picamera2:
            frame = self.cam.capture_array()
            return frame
        ok, frame = self.cam.read()
        return frame if ok else None

    def release(self):
        if self.use_picamera2:
            self.cam.stop()
        else:
            self.cam.release()


# ============================================================================
# 3. KALIBRIERUNG (Perspektiv-Entzerrung Kamerabild -> Tisch-Koordinaten)
# ============================================================================

class TableCalibration:
    def __init__(self, config: TableConfig, out_w: int = 1000, out_h: int = 500,
                 margin_frac: float = 0.025):
        """
        margin_frac: Die Tischecken werden NICHT auf die exakte Bildecke (0,0)
            des entzerrten Bildes gemappt, sondern mit einem Randabstand von
            margin_frac * out_w bzw. margin_frac * out_h leicht nach innen
            versetzt (PROPORTIONAL zu Breite/Hoehe, nicht ein fester Pixel-
            wert - sonst wuerde bei ungleichen Raendern das Seitenverhaeltnis
            leicht verzerrt, da scale_x != scale_y entstehen wuerde).
            Grund fuer den Rand ueberhaupt: Video-Kompression kann exakt an
            der Bildecke (Pixel 0,0 etc.) andere Artefakte erzeugen als im
            restlichen Bild (z.B. kurzlebige Geister-Fragmente genau beim
            Verschwinden einer Kugel dort) - im Test klar reproduziert und
            auf die literale Bildecke eingegrenzt (Kantenmitte war nicht
            betroffen). Da Eck-Taschen genau auf die Tischecken fallen,
            waeren sie ohne diesen Rand am staerksten betroffen.
        """
        self.config = config
        self.out_w, self.out_h = out_w, out_h
        self.margin_x = out_w * margin_frac
        self.margin_y = out_h * margin_frac
        src = np.array(config.corners_px, dtype=np.float32)
        dst = np.array(
            [[self.margin_x, self.margin_y], [out_w - self.margin_x, self.margin_y],
             [out_w - self.margin_x, out_h - self.margin_y], [self.margin_x, out_h - self.margin_y]],
            dtype=np.float32
        )
        self.H = cv2.getPerspectiveTransform(src, dst)
        self.scale_x = config.table_w_mm / (out_w - 2 * self.margin_x)
        self.scale_y = config.table_h_mm / (out_h - 2 * self.margin_y)

    def warp(self, frame: np.ndarray) -> np.ndarray:
        # Manche Kamera-Backends/Bildquellen liefern 4 Kanaele (BGRA) statt
        # der erwarteten 3 (BGR) - z.B. bestimmte Aufnahmepfade oder Bilder
        # mit Alpha-Kanal. Ein 4-Kanal-Bild hat ebenfalls ndim==3 (ndim
        # zaehlt nur Achsen, nicht Kanaele!), wuerde also spaeter in
        # BallDetector.detect() faelschlich durch die BGR2GRAY-Konvertierung
        # laufen und ein fast komplett schwarzes, unbrauchbares Ergebnis
        # erzeugen (im Testprotokoll reproduziert) - hier normalisieren,
        # BEVOR irgendetwas Nachgelagertes (Hintergrundlernen, Erkennung)
        # das Bild zu Gesicht bekommt.
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return cv2.warpPerspective(frame, self.H, (self.out_w, self.out_h))

    def px_to_mm(self, pt_px: np.ndarray) -> np.ndarray:
        """Punkt im entzerrten (Top-Down) Bild -> mm auf dem Tisch."""
        return np.array([(pt_px[0] - self.margin_x) * self.scale_x,
                          (pt_px[1] - self.margin_y) * self.scale_y])


# ============================================================================
# 4. BALL-ERKENNUNG
# ============================================================================

class BallDetector:
    """Erkennt Kugeln per Hintergrundsubtraktion + Hough-Kreise.

    Für produktiven Einsatz empfiehlt sich stattdessen ein trainiertes
    YOLOv8-Modell (ultralytics) für robustere Erkennung bei wechselndem
    Licht/Reflexionen. Diese Klasse ist ein funktionierender Startpunkt.
    """

    def __init__(self, ball_radius_px: float, is_color_camera: bool = False):
        self.ball_radius_px = ball_radius_px
        self.is_color_camera = is_color_camera
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=False
        )
        self.background_learned = False

    def detect(self, frame_topdown: np.ndarray):
        """Gibt Liste von Detektionen zurück: (x, y, radius, mean_gray, roi)."""
        gray = cv2.cvtColor(frame_topdown, cv2.COLOR_BGR2GRAY) \
            if frame_topdown.ndim == 3 else frame_topdown

        fg_mask = self.bg_subtractor.apply(frame_topdown, learningRate=0)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,
                                    np.ones((3, 3), np.uint8))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE,
                                    np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        r = self.ball_radius_px
        for c in contours:
            area = cv2.contourArea(c)
            expected_area = np.pi * r * r
            # Obergrenze bewusst grosszuegig (nicht zu eng): zwei nah beieinander-
            # liegende Kugeln koennen durch Kompressionsartefakte kurz zu einem
            # ca. 2x-grossen Blob verschmelzen. Eine zu enge Obergrenze wuerde
            # genau im Kontaktmoment (wenn Kugeln sich beruehren) zu einem
            # kompletten Erkennungsausfall fuehren und den Erstkontakt verpassen
            # (im Testprotokoll reproduziert). Die eigentliche Absicherung gegen
            # Identitaetsverwechslung bei einer verschmolzenen Erkennung passiert
            # NICHT hier, sondern art-bewusst in BallTracker.update() (Kosten-
            # Strafe bei Kind-Mismatch), was den Kontaktmoment nicht beeintraechtigt.
            if not (0.4 * expected_area < area < 2.5 * expected_area):
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (int(x), int(y)), int(radius * 0.7), 255, -1)
            mean_gray = cv2.mean(gray, mask=mask)[0]
            x0, y0 = int(x - radius), int(y - radius)
            x1, y1 = int(x + radius), int(y + radius)
            roi = frame_topdown[max(0, y0):y1, max(0, x0):x1]
            detections.append((x, y, radius, mean_gray, roi))
        return detections

    def classify_group(self, mean_gray: float, roi: np.ndarray) -> str:
        """Sehr grobe Heuristik zur Klassifikation.

        - Sehr dunkel  -> vermutlich die schwarze 8
        - Sehr hell + wenig Kantenanteil -> vermutlich Weiße (Cue-Ball)
        - Sonst: Kantendichte im ROI als Proxy für "gestreift" vs "voll"
          (gestreift hat mehr Kontrastkanten durch den weißen Ring)

        Wichtig: die Kantenanalyse wird NUR auf einen inneren Kreisbereich
        (Radius = 60% des tatsächlichen Kugelradius) angewendet. Würde man
        das gesamte quadratische ROI nehmen, würde der Kugel-Außenrand
        selbst als "Kante" gezählt und jede Kugel fälschlich als
        "texturiert" erscheinen.

        TODO Farbkamera: hier stattdessen HSV-Histogramm des ROI auswerten
        und über Farbtabelle (rot=Voll 3/11, gelb=Voll 1/9, ...) klassifizieren.
        """
        if mean_gray < 40:
            return "eight"
        if roi.size == 0:
            return "unknown"
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        h, w = gray_roi.shape[:2]
        cy, cx = h // 2, w // 2
        inner_r = int(min(h, w) * 0.30)  # Radius = 60% des Kugelradius (min(h,w)=~2*Kugelradius)
        mask = np.zeros_like(gray_roi, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), max(inner_r, 1), 255, -1)
        edges = cv2.Canny(gray_roi, 60, 150)
        edges_inner = cv2.bitwise_and(edges, edges, mask=mask)
        inner_pixel_count = max(int(np.count_nonzero(mask)), 1)
        edge_density = np.count_nonzero(edges_inner) / inner_pixel_count
        if mean_gray > 180 and edge_density < 0.03:
            return "cue"
        return "stripe" if edge_density > 0.06 else "solid"


# ============================================================================
# 5. TRACKING
# ============================================================================

@dataclass
class TrackedBall:
    id: int
    kind: str  # 'cue', 'eight', 'solid', 'stripe', 'unknown'
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_seen: float = field(default_factory=time.time)
    missing_frames: int = 0
    frames_seen: int = 1
    pocketed: bool = False
    off_table: bool = False

    @property
    def confirmed(self) -> bool:
        """Ein einzelner, nie wiederholter Blitz-Blob (z.B. ein Kompressions-
        Artefakt genau im Moment, in dem eine ANDERE Kugel an derselben
        Stelle verschwindet) soll nicht als 'versenkt/vom Tisch' zaehlen
        koennen. Erst nach mind. 2 aufeinanderfolgenden Sichtungen gilt eine
        Kugel als echt genug, um ueberhaupt als 'verschwunden' gewertet zu
        werden - eine echte Kugel wird ohnehin ueber viele Frames verfolgt,
        bevor sie in eine Tasche rollt."""
        return self.frames_seen >= 2


class BallTracker:
    def __init__(self, max_missing_frames: int = 30, match_dist_px: float = 40.0):
        self.balls: dict[int, TrackedBall] = {}
        self._next_id = itertools.count()
        self.max_missing_frames = max_missing_frames
        self.match_dist_px = match_dist_px

    def update(self, detections, classify_fn, dt: float):
        positions = [np.array([d[0], d[1]]) for d in detections]
        active_ids = [bid for bid, b in self.balls.items() if not b.pocketed and not b.off_table]

        # Kind-bewusste Kostenberechnung: eine Erkennung, deren Kugelart klar
        # NICHT zur bisherigen Kennung einer Kugel passt, darf deren Position
        # nicht einfach per kuerzester Distanz "erben". Ohne diese Pruefung
        # kann z.B. eine schnell bewegte, vom Tracking kurz verlorene Kugel
        # (die dabei mehrfach eine neue ID bekam) am Ende zufaellig naeher an
        # einer ganz ANDEREN, ruhenden Kugel landen als deren eigene, aber
        # schon lange nicht mehr aktualisierte Vorhersage - und ihr so
        # faelschlich deren Position/Kennung "stehlen" (im Testprotokoll
        # reproduziert: eine schnelle Zielkugel hat am Ende die Position der
        # unbewegten Weissen uebernommen). Quick-Klassifikation wird pro
        # Erkennung nur einmal berechnet (gecacht) und nur bei Bedarf.
        quick_kinds = [None] * len(detections)

        def quick_kind(j):
            if quick_kinds[j] is None:
                _, _, _, mean_gray, roi = detections[j]
                quick_kinds[j] = classify_fn(mean_gray, roi)
            return quick_kinds[j]

        if active_ids and positions:
            assign_cost = np.zeros((len(active_ids), len(positions)))
            dist_cost = np.zeros((len(active_ids), len(positions)))
            for i, bid in enumerate(active_ids):
                predicted = self.balls[bid].pos + self.balls[bid].vel * dt
                tracked_kind = self.balls[bid].kind
                for j, p in enumerate(positions):
                    base_cost = np.linalg.norm(predicted - p)
                    dist_cost[i, j] = base_cost
                    penalty = 0.0
                    if tracked_kind != "unknown":
                        qk = quick_kind(j)
                        if qk != "unknown" and qk != tracked_kind:
                            # Straf-Aufschlag NUR fuer die Zuordnungs-Optimierung
                            # (bevorzugt bei mehreren Kandidaten die kind-passende
                            # Erkennung), NICHT fuer die Akzeptanz-Schwelle unten.
                            # Ein einzelner Klassifikations-"Flacker" (z.B. kurzer
                            # Schatten laesst eine Kugel fuer 1 Frame als andere
                            # Art erscheinen) soll NICHT dazu fuehren, dass die
                            # naheliegendste (und einzig plausible) Erkennung
                            # abgelehnt wird, nur weil sie kurzzeitig anders
                            # klassifiziert wurde (im Testprotokoll reproduziert).
                            penalty = 500.0
                    assign_cost[i, j] = base_cost + penalty
            row_ind, col_ind = linear_sum_assignment(assign_cost)
        else:
            dist_cost = None
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)

        matched_det = set()
        for r, c in zip(row_ind, col_ind):
            if dist_cost[r, c] > self.match_dist_px:
                continue
            bid = active_ids[r]
            new_pos = positions[c]
            b = self.balls[bid]
            b.vel = (new_pos - b.pos) / dt if dt > 0 else np.zeros(2)
            b.pos = new_pos
            b.last_seen = time.time()
            b.missing_frames = 0
            b.frames_seen += 1
            matched_det.add(c)
            # Nachklassifikation: nicht nur fuer "unknown", sondern fuer JEDE
            # Kugel in ihren ersten Sichtungen (frames_seen < 5). Grund: eine
            # Kugel kann beim ALLERERSTEN Erkennen konfident, aber FALSCH
            # klassifiziert werden (z.B. die Weisse erscheint durch einen
            # kurzen Schatten als "solid" statt "cue") - ohne diese
            # Korrekturmoeglichkeit wuerde das dauerhaft haengen bleiben,
            # weil eine bereits "sichere" (nicht-unknown) Klassifikation
            # sonst nie wieder ueberprueft wird. Im Testprotokoll reproduziert:
            # eine falsch als "solid" erkannte Weisse fuehrte dazu, dass
            # cue_ball nie gefunden wird und JEDER Stoss der Partie faelschlich
            # als "trifft nichts" (Foul) gemeldet wird. Nach den ersten 5
            # Sichtungen wird die Klassifikation wie bisher final eingefroren,
            # damit spaeteres Lichtflackern eine etablierte Kugel nicht mehr
            # umklassifizieren kann.
            if b.kind == "unknown" or b.frames_seen < 5:
                new_kind = quick_kind(c)
                if new_kind != "unknown":
                    b.kind = new_kind

        matched_ids = {active_ids[r] for r, c in zip(row_ind, col_ind)
                       if dist_cost[r, c] <= self.match_dist_px}
        for bid in active_ids:
            if bid not in matched_ids:
                self.balls[bid].missing_frames += 1

        for j, d in enumerate(detections):
            if j in matched_det:
                continue
            kind = quick_kind(j)
            new_id = next(self._next_id)
            self.balls[new_id] = TrackedBall(id=new_id, kind=kind,
                                              pos=positions[j])

    def all_stationary(self, speed_thresh_mm_s: float = 15.0,
                        px_to_mm: float = 1.0) -> bool:
        for b in self.balls.values():
            if b.pocketed or b.off_table:
                continue
            speed = np.linalg.norm(b.vel) * px_to_mm
            if speed > speed_thresh_mm_s:
                return False
        return True


# ============================================================================
# 6. EREIGNISSE WÄHREND EINES STOSSES
# ============================================================================

@dataclass
class ShotEvents:
    first_contact_kind: Optional[str] = None
    cushions_touched: set = field(default_factory=set)   # ball ids
    pocketed: list = field(default_factory=list)          # (id, kind, pocket_idx)
    cue_pocketed: bool = False
    cue_off_table: bool = False
    eight_off_table: bool = False
    any_ball_off_table: bool = False
    foot_foul: bool = False  # Platzhalter, siehe Modulkopf


class ShotEventDetector:
    """Beobachtet einen einzelnen Stoß (von Antritt bis Stillstand aller
    Kugeln) und protokolliert die für die Regelauswertung nötigen Ereignisse.
    """

    def __init__(self, calib: TableCalibration, table_config: TableConfig,
                 ball_radius_px: float, cushion_margin_mm: float = 30.0,
                 contact_margin_factor: float = 1.3):
        """
        ball_radius_px: Kugelradius IN DER ENTZERRTEN TOP-DOWN-ANSICHT
            (gleicher Wert wie bei BallDetector). Wird genutzt, um den
            Kontaktabstand korrekt zu berechnen: zwei Kugeln beruehren sich,
            wenn ihr Mittelpunktabstand ca. 2 * ball_radius_px betraegt.
        contact_margin_factor: Sicherheitsspielraum wegen Erkennungsrauschen
            (Standard 1.3 = 30% Toleranz ueber dem theoretischen Kontaktabstand).
        """
        self.calib = calib
        self.config = table_config
        self.cushion_margin_mm = cushion_margin_mm
        self.contact_thresh_px = 2.0 * ball_radius_px * contact_margin_factor
        self.events = ShotEvents()
        self._contact_registered = False
        self._prev_positions: dict[int, np.ndarray] = {}

    def reset(self):
        self.events = ShotEvents()
        self._contact_registered = False
        self._prev_positions = {}

    def process_frame(self, tracker: BallTracker):
        cue_ball = next((b for b in tracker.balls.values() if b.kind == "cue"
                          and not b.pocketed and not b.off_table), None)

        # -- Bewegungsstatus je Kugel einmal bestimmen (mehrfach gebraucht) --
        moved_since_last = {}
        for b in tracker.balls.values():
            if b.pocketed or b.off_table:
                continue
            moved_since_last[b.id] = (
                b.id in self._prev_positions and
                np.linalg.norm(b.pos - self._prev_positions[b.id]) > 0.5
            )

        # -- Bandenkontakt pruefen: nur fuer Kugeln, die sich GERADE bewegen.
        # Sonst wuerde eine Kugel, die von einem frueheren Stoss schon an
        # der Bande liegt (aber sich in DIESEM Stoss nicht ruehrt), faelschlich
        # als Bandenkontakt in diesem Stoss durchgehen.
        for b in tracker.balls.values():
            if b.pocketed or b.off_table:
                continue
            if not moved_since_last.get(b.id, False):
                continue
            mm = self.calib.px_to_mm(b.pos)
            near_edge = (
                mm[0] < self.cushion_margin_mm or
                mm[0] > self.config.table_w_mm - self.cushion_margin_mm or
                mm[1] < self.cushion_margin_mm or
                mm[1] > self.config.table_h_mm - self.cushion_margin_mm
            )
            if near_edge:
                self.events.cushions_touched.add(b.id)

        # -- Erstkontakt der Weißen prüfen --
        if cue_ball is not None and not self._contact_registered:
            for b in tracker.balls.values():
                # b.kind == "cue" wird zusaetzlich zur gewaehlten cue_ball-ID
                # ausgeschlossen: sollten durch eine Fehlklassifikation zwei
                # Kugeln gleichzeitig als "cue" erkannt werden, wuerde ein
                # Treffer auf die ZWEITE sonst faelschlich first_contact_kind
                # ="cue" melden - ein Wert, der nie zu einer Gruppe passen
                # kann und IMMER faelschlich ein Foul ausloesen wuerde. Ein
                # zweiter "cue"-Kandidat ist per Definition ein
                # Klassifikationsfehler, kein gueltiges Kontaktziel.
                if b.id == cue_ball.id or b.kind == "cue" or b.pocketed or b.off_table:
                    continue
                dist = np.linalg.norm(cue_ball.pos - b.pos)
                if dist < self.contact_thresh_px and moved_since_last.get(b.id, False):
                    self.events.first_contact_kind = b.kind
                    self._contact_registered = True
                    break

        # -- Verschwundene Kugeln: versenkt oder vom Tisch gesprungen --
        # HINWEIS: hier bewusst KEINE Sonderbehandlung fuer "unbestaetigte"
        # (nur 1x gesehene) Kugeln mehr. Eine fruehere Fassung hat solche
        # Kugeln nach nur 2 fehlenden Frames stillschweigend verworfen, um
        # Kompressions-Geisterartefakte nahe der Bildecke abzufangen - das
        # eigentliche Problem (Artefakte exakt an der Bildecke) wird aber
        # bereits durch den Rand in TableCalibration geloest (Taschen liegen
        # dort nicht mehr exakt auf der Bildecke). Die fruehe Verwerfung hatte
        # dagegen ein ernsteres Risiko: eine Kugel, die durch kurzzeitigen
        # Tracking-Verlust (z.B. Ueberlappung) kurz vor dem Einlochen eine
        # NEUE ID bekommt, waere als "unbestaetigt" verworfen worden UND der
        # eigentliche Pot waere komplett verloren gegangen (im Testprotokoll
        # reproduziert). Alle Kugeln bekommen daher gleichermassen die volle
        # max_missing_frames-Frist.
        for b in tracker.balls.values():
            if b.pocketed or b.off_table:
                continue
            if b.missing_frames >= tracker.max_missing_frames:
                # Verwaistes Duplikat erkennen: liegt GENAU JETZT eine ANDERE
                # aktive Kugel derselben Art nahe an dieser (laengst
                # eingefrorenen) Position? Dann ist dies hoechstwahrscheinlich
                # ein Ueberbleibsel eines Verschmelzungs-Ereignisses (die echte
                # Kugel laeuft unter einer anderen ID weiter) und KEIN eigenes
                # Versenkt-/Vom-Tisch-Ereignis - stillschweigend entfernen,
                # ohne ein (falsches) Ereignis zu buchen.
                is_orphan_duplicate = any(
                    other.id != b.id and other.kind == b.kind and
                    not other.pocketed and not other.off_table and
                    np.linalg.norm(other.pos - b.pos) < tracker.match_dist_px
                    for other in tracker.balls.values()
                )
                if is_orphan_duplicate:
                    b.off_table = True  # nur intern aus dem Tracking entfernen
                    continue
                mm = self.calib.px_to_mm(b.pos)
                pocket_idx = self._nearest_pocket(mm)
                if pocket_idx is not None:
                    b.pocketed = True
                    self.events.pocketed.append((b.id, b.kind, pocket_idx))
                    if b.kind == "cue":
                        self.events.cue_pocketed = True
                else:
                    b.off_table = True
                    self.events.any_ball_off_table = True
                    if b.kind == "cue":
                        self.events.cue_off_table = True
                    elif b.kind == "eight":
                        self.events.eight_off_table = True

        self._prev_positions = {bid: b.pos.copy() for bid, b in tracker.balls.items()}

    def _nearest_pocket(self, pos_mm: np.ndarray) -> Optional[int]:
        best_idx, best_dist = None, None
        for idx, p in enumerate(self.config.pocket_positions_mm):
            dist = np.linalg.norm(pos_mm - np.array(p))
            if dist < self.config.pocket_radius_mm * 1.5:
                if best_dist is None or dist < best_dist:
                    best_idx, best_dist = idx, dist
        return best_idx


# ============================================================================
# 7. REGELWERK / STATE MACHINE
# ============================================================================

class GamePhase(Enum):
    BREAK = auto()
    OPEN_TABLE = auto()
    GROUPS_ASSIGNED = auto()
    GAME_OVER = auto()


class Group(Enum):
    SOLID = "Volle"
    STRIPE = "Halbe/Gestreifte"


@dataclass
class Player:
    name: str
    group: Optional[Group] = None
    remaining_balls: int = 7  # sinkt bei jeder legal versenkten eigenen Kugel


class Foul(Enum):
    WRONG_BALL_FIRST = "Falsche Kugel zuerst getroffen (oder gar keine)."
    SCRATCH = "Weiße Kugel versenkt (Scratch)."
    NO_RAIL_CONTACT = "Nach dem Stoß wurde keine Bande berührt und keine Kugel versenkt."
    CUE_OFF_TABLE = "Weiße Kugel ist vom Tisch gesprungen."
    FOOT_FOUL = "Antritt mit weniger als einem Fuß am Boden."
    BREAK_FOUL = "Ungültiger Break: weniger als 4 Bandenkontakte und keine Kugel versenkt."
    EIGHT_EARLY = "Verlust: Schwarze 8 wurde versenkt, bevor die eigene Gruppe leer war."
    EIGHT_WRONG_POCKET = "Verlust: Schwarze 8 wurde in die falsche Tasche versenkt."
    EIGHT_SCRATCH = "Verlust: Schwarze 8 wurde gleichzeitig mit einem Scratch versenkt."
    EIGHT_OFF_TABLE = "Verlust: Schwarze 8 ist vom Tisch gesprungen."


class NotificationManager:
    """Zentrale Stelle für Regelverstoß-Meldungen.

    Standardmäßig Konsolenausgabe. Für echte Benachrichtigungen (Telegram,
    Desktop-Popup, o.ä.) hier den Versand ergänzen, z.B.:

        import requests
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                       json={"chat_id": CHAT_ID, "text": message})
    """

    def send(self, message: str):
        print(f"[REGELVERSTOSS] {message}")


class RuleEngine:
    def __init__(self, notifier: NotificationManager,
                 player_names=("Spieler 1", "Spieler 2")):
        self.notifier = notifier
        self.phase = GamePhase.BREAK
        self.players = [Player(n) for n in player_names]
        self.current_idx = 0
        self.called_pocket_for_eight: Optional[int] = None  # von außen gesetzt
        self.winner: Optional[str] = None

    @property
    def current_player(self) -> Player:
        return self.players[self.current_idx]

    @property
    def opponent(self) -> Player:
        return self.players[1 - self.current_idx]

    def _switch_turn(self):
        self.current_idx = 1 - self.current_idx

    def _foul(self, foul: Foul):
        self.notifier.send(f"{foul.value} -> Ball in Hand für {self.opponent.name}.")
        self._switch_turn()

    def _win(self, foul_reason: Optional[Foul] = None):
        if foul_reason is not None:
            self.notifier.send(f"{foul_reason.value} {self.opponent.name} gewinnt!")
            self.winner = self.opponent.name
        else:
            self.notifier.send(f"{self.current_player.name} versenkt die 8 regelgerecht und gewinnt!")
            self.winner = self.current_player.name
        self.phase = GamePhase.GAME_OVER

    def evaluate_shot(self, events: ShotEvents):
        """Wird nach jedem abgeschlossenen Stoß (alle Kugeln stehen still)
        aufgerufen und wertet die gesammelten `ShotEvents` gegen das
        Regelwerk aus.
        """
        if self.phase == GamePhase.GAME_OVER:
            return

        pocketed_eight = next(((pid, kind, pk) for pid, kind, pk in events.pocketed
                                if kind == "eight"), None)
        pocketed_own = [p for p in events.pocketed
                         if self._kind_matches_group(p[1], self.current_player.group)]
        pocketed_any_ball = len(events.pocketed) > 0

        # -- Fußfoul geht immer vor (Stoß ungültig) --
        if events.foot_foul:
            self._foul(Foul.FOOT_FOUL)
            return

        # ---------------- Break-Phase ----------------
        if self.phase == GamePhase.BREAK:
            if pocketed_eight is not None:
                scratch_note = (" Zusätzlich wurde im selben Stoß auch die Weiße "
                                 "versenkt (Scratch)." if events.cue_pocketed else "")
                self.notifier.send(
                    "Die 8 wurde beim Break versenkt." + scratch_note +
                    " Das Regelwerk legt für diesen Fall keine automatische "
                    "Entscheidung fest (übliche Varianten: neu aufbauen und "
                    "erneut anstoßen, oder die 8 zurücksetzen und normal "
                    "weiterspielen) - bitte manuell klären. Phase bleibt auf "
                    "BREAK, bis erneut angestoßen wird."
                )
                return
            valid_break = len(events.cushions_touched) >= 4 or pocketed_any_ball
            if not valid_break:
                self._foul(Foul.BREAK_FOUL)
                return
            self.phase = GamePhase.OPEN_TABLE
            if events.cue_pocketed:
                self._foul(Foul.SCRATCH)
                return
            if not pocketed_any_ball:
                self._switch_turn()
            # sonst: Stoß war regulär und etwas wurde versenkt -> gleicher Spieler bleibt dran
            return

        # ---------------- 8 im Spiel ----------------
        if pocketed_eight is not None:
            _, _, pocket_idx = pocketed_eight
            own_group_cleared = self.current_player.remaining_balls <= 0
            if not own_group_cleared:
                self._win(Foul.EIGHT_EARLY)
                return
            if (self.called_pocket_for_eight is not None and
                    pocket_idx != self.called_pocket_for_eight):
                self._win(Foul.EIGHT_WRONG_POCKET)
                return
            if events.cue_pocketed:
                self._win(Foul.EIGHT_SCRATCH)
                return
            self._win(None)
            return

        if events.eight_off_table:
            self._win(Foul.EIGHT_OFF_TABLE)
            return

        # ---------------- Normale Fouls ----------------
        contact_kind = events.first_contact_kind
        contact_ok = self._contact_matches_group(contact_kind)

        if contact_kind is None or not contact_ok:
            self._foul(Foul.WRONG_BALL_FIRST)
            return

        # Ab hier war der Erstkontakt legal. Buchhaltung (Gruppenzuweisung,
        # verbleibende Kugeln) muss JETZT erfolgen -- unabhaengig davon, ob
        # der Stoss zusaetzlich noch einen Scratch-/Off-Table-Foul enthaelt.
        # Einmal legal versenkte Kugeln bleiben unten, auch wenn im selben
        # Stoss zusaetzlich gescratcht wird.
        if self.phase == GamePhase.OPEN_TABLE and pocketed_own:
            distinct_kinds = {p[1] for p in pocketed_own}
            if len(distinct_kinds) > 1:
                self.notifier.send(
                    "Hinweis: Beim offenen Tisch wurden im selben Stoß sowohl "
                    "eine volle als auch eine gestreifte Kugel versenkt. Das "
                    "Regelwerk legt hierfür keine eindeutige Reihenfolge fest "
                    "-- es wird vereinfachend die zuerst erkannte Kugel für "
                    "die Gruppenzuweisung verwendet. Bei Uneinigkeit bitte "
                    "manuell klären."
                )
            self._assign_groups_from_first_pot(pocketed_own[0][1])
        self._update_remaining_balls(events)

        if events.cue_pocketed:
            self._foul(Foul.SCRATCH)
            return

        if events.cue_off_table:
            self._foul(Foul.CUE_OFF_TABLE)
            return

        if not events.cushions_touched and not pocketed_any_ball:
            self._foul(Foul.NO_RAIL_CONTACT)
            return

        if pocketed_own:
            return  # gleicher Spieler bleibt dran
        self._switch_turn()

    # -- Hilfsfunktionen --------------------------------------------------

    def _kind_matches_group(self, kind: str, group: Optional[Group]) -> bool:
        if group is None:
            return kind in ("solid", "stripe")  # offener Tisch: alles außer 8/cue erlaubt
        if group == Group.SOLID:
            return kind == "solid"
        return kind == "stripe"

    def _contact_matches_group(self, kind: Optional[str]) -> bool:
        if kind is None:
            return False
        group = self.current_player.group
        if group is None:
            return kind in ("solid", "stripe")
        if self.current_player.remaining_balls <= 0:
            return kind == "eight"
        return self._kind_matches_group(kind, group)

    def _assign_groups_from_first_pot(self, kind: str):
        self.phase = GamePhase.GROUPS_ASSIGNED
        if kind == "solid":
            self.current_player.group = Group.SOLID
            self.opponent.group = Group.STRIPE
        else:
            self.current_player.group = Group.STRIPE
            self.opponent.group = Group.SOLID
        self.notifier.send(
            f"Gruppen zugewiesen: {self.current_player.name} = "
            f"{self.current_player.group.value}, {self.opponent.name} = "
            f"{self.opponent.group.value}."
        )

    def _update_remaining_balls(self, events: ShotEvents):
        for _, kind, _ in events.pocketed:
            if kind == "eight":
                continue
            for player in self.players:
                if self._kind_matches_group(kind, player.group):
                    was_positive = player.remaining_balls > 0
                    player.remaining_balls = max(0, player.remaining_balls - 1)
                    if was_positive and player.remaining_balls == 0:
                        self.notifier.send(
                            f"{player.name} hat alle eigenen Kugeln versenkt! "
                            f"Vor dem Stoss auf die 8 jetzt die Tasche ansagen "
                            f"(Taste 1-6 im Live-Fenster)."
                        )


# ============================================================================
# 8. HAUPTPROGRAMM
# ============================================================================

def _load_corners(calibration_path: Optional[str], corners_arg: Optional[list]):
    """Liefert die 4 Eck-Punkte entweder aus einer JSON-Datei (von
    calibrate_table.py) oder aus manuell übergebenen 'x,y'-Strings."""
    if calibration_path and corners_arg:
        print(f"HINWEIS: sowohl --calibration als auch --corners angegeben - "
              f"--calibration ('{calibration_path}') wird verwendet, --corners "
              f"wird ignoriert.")
    if calibration_path:
        import json
        try:
            with open(calibration_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Kalibrierdatei nicht gefunden: '{calibration_path}'.")
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Kalibrierdatei '{calibration_path}' ist kein gültiges JSON "
                f"(vermutlich beschädigt oder falsche Datei): {e}"
            )
        if "corners_px" not in data:
            raise ValueError(
                f"Kalibrierdatei '{calibration_path}' enthält keinen "
                f"'corners_px'-Schlüssel - ist das wirklich eine von "
                f"calibrate_table.py erzeugte Datei?"
            )
        pts = [tuple(p) for p in data["corners_px"]]
        if len(pts) != 4:
            raise ValueError(
                f"Kalibrierdatei '{calibration_path}' enthält {len(pts)} "
                f"Punkte statt der erwarteten 4 (corners_px)."
            )
        return pts
    if corners_arg:
        pts = []
        for token in corners_arg:
            parts = token.split(",")
            if len(parts) != 2:
                raise ValueError(
                    f"--corners Punkt '{token}' hat falsches Format. "
                    f"Erwartet wird 'x,y' (z.B. '150,80'), mit genau einem Komma."
                )
            x_str, y_str = parts
            try:
                pts.append((float(x_str), float(y_str)))
            except ValueError:
                raise ValueError(
                    f"--corners Punkt '{token}': 'x' und 'y' müssen Zahlen sein."
                )
        if len(pts) != 4:
            raise ValueError("--corners braucht genau 4 Punkte.")
        return pts
    raise ValueError("Entweder --calibration oder --corners angeben.")


def _build_pocket_map_image(table_config: TableConfig) -> np.ndarray:
    """Erzeugt eine kleine Uebersichtsgrafik des Tisches mit nummerierten
    Taschen (1-6), passend zur Tastenbelegung im Live-Fenster. Dient nur
    als visuelle Gedaechtnisstuetze, welche Zahl zu welcher physischen
    Tasche gehoert - wird einmalig erzeugt und in einem eigenen kleinen
    Fenster angezeigt.

    WICHTIG: Die Grafik richtet sich automatisch nach dem tatsaechlich
    konfigurierten table_w_mm/table_h_mm-Seitenverhaeltnis aus (Hoch- oder
    Querformat), statt Querformat fest anzunehmen. Massgeblich ist, was du
    bei der Kalibrierung tatsaechlich als 'oben-links' etc. angeklickt hast
    - je nachdem faellt die Grafik quer (Laenge horizontal, Seitentaschen
    oben-Mitte/unten-Mitte) oder hochkant (Laenge vertikal, Seitentaschen
    Mitte-links/Mitte-rechts) aus. Die Positionen ergeben sich rein
    geometrisch aus table_w_mm/table_h_mm - es wird nichts angenommen."""
    margin = 20
    long_side_px = 220   # Zielgroesse der laengeren Tischseite in Pixeln
    is_landscape = table_config.table_w_mm >= table_config.table_h_mm
    ratio = (min(table_config.table_w_mm, table_config.table_h_mm) /
             max(table_config.table_w_mm, table_config.table_h_mm))
    short_side_px = max(40, int(long_side_px * ratio))

    if is_landscape:
        table_rect_w, table_rect_h = long_side_px, short_side_px
    else:
        table_rect_w, table_rect_h = short_side_px, long_side_px

    img_w = table_rect_w + 2 * margin
    img_h = table_rect_h + 2 * margin + 20  # +20 fuer die Beschriftungszeile
    img = np.full((img_h, img_w, 3), (40, 40, 40), dtype=np.uint8)

    scale_x = table_rect_w / table_config.table_w_mm
    scale_y = table_rect_h / table_config.table_h_mm

    cv2.rectangle(img, (margin, margin),
                  (margin + table_rect_w, margin + table_rect_h),
                  (60, 140, 20), -1)
    cv2.rectangle(img, (margin, margin),
                  (margin + table_rect_w, margin + table_rect_h),
                  (210, 210, 210), 2)

    for idx, (mx, my) in enumerate(table_config.pocket_positions_mm):
        px = margin + int(mx * scale_x)
        py = margin + int(my * scale_y)
        cv2.circle(img, (px, py), 10, (10, 10, 10), -1)
        cv2.circle(img, (px, py), 10, (255, 255, 255), 1)
        label = str(idx + 1)
        tsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.putText(img, label, (px - tsize[0] // 2, py + tsize[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img, "Tasten 1-6 = Tasche fuer die 8", (8, img_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Live-Regelüberwachung 8-Ball (OV9281)")
    parser.add_argument("--calibration", type=str, default=None,
                         help="JSON-Datei aus calibrate_table.py")
    parser.add_argument("--corners", nargs=4, default=None,
                         help="Alternativ zu --calibration: 4 Punkte 'x,y' manuell")
    parser.add_argument("--device", type=int, default=0, help="V4L2 Geraeteindex")
    parser.add_argument("--picamera2", action="store_true",
                         help="Raspberry-Pi-CSI-Kamera statt V4L2/USB verwenden")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--out-w", type=int, default=1000,
                         help="Breite der entzerrten Top-Down-Ansicht in Pixeln. "
                              "WICHTIG fuer Performance: die Erkennung (v.a. "
                              "Hintergrundsubtraktion) skaliert etwa quadratisch "
                              "mit der Aufloesung. Im Test: 1000x500 ~19ms/Frame, "
                              "500x250 ~4.5ms/Frame (ca. 4x schneller). Bei 120fps "
                              "Zielrate bleiben nur ~8.3ms/Frame - ggf. verkleinern, "
                              "falls die Verarbeitung nicht mit der Kamera mithaelt.")
    parser.add_argument("--out-h", type=int, default=500)
    parser.add_argument("--table-w-mm", type=float, default=2540.0)
    parser.add_argument("--table-h-mm", type=float, default=1270.0)
    parser.add_argument("--ball-radius-px", type=float, required=True,
                         help="Ballradius IN DER ENTZERRTEN TOP-DOWN-ANSICHT (Pixel); "
                              "mit test_offline.py ermitteln/pruefen")
    parser.add_argument("--background-frames", type=int, default=40,
                         help="Anzahl ECHTER Kameraframes am Start zum Hintergrund-"
                              "Lernen. Tisch MUSS in dieser Zeit leer sein.")
    parser.add_argument("--max-missing-frames", type=int, default=30,
                         help="Wie viele Frames eine Kugel fehlen darf, bevor sie als "
                              "versenkt/vom Tisch gilt. WICHTIG: haengt von --fps ab, "
                              "UND vom Kontaktmoment zweier Kugeln - Video-Kompression "
                              "kann zwei nah beieinanderliegende Kugeln kurz zu einem "
                              "Blob verschmelzen lassen, was fuer mehrere Frames zu "
                              "KEINER Erkennung fuehren kann. Im Test (30fps) waren das "
                              "teils >60 Frames. Eher grosszuegig waehlen (z.B. 30-60) "
                              "und mit test_offline.py an echtem Material verifizieren "
                              "-- ein zu hoher Wert kostet nur Reaktionszeit, ein zu "
                              "niedriger kann Kugeln faelschlich als versenkt melden.")
    parser.add_argument("--match-dist-px", type=float, default=40.0,
                         help="Max. Distanz (Pixel in der Top-Down-Ansicht) zwischen "
                              "vorhergesagter und tatsaechlicher Position, damit eine "
                              "Erkennung noch derselben Kugel zugeordnet wird. Bei "
                              "SCHNELLEN Stoessen (v.a. harter Break) kann eine Kugel "
                              "zwischen zwei Frames weiter wandern als dieser Wert -- "
                              "dann fragmentiert eine einzelne Kugel in mehrere IDs. "
                              "Mit test_offline.py am eigenen Material prophylaktisch "
                              "pruefen (auf haeufig wechselnde IDs achten) und ggf. "
                              "erhoehen (v.a. bei niedrigerer fps).")
    parser.add_argument("--player-names", nargs=2, default=["Spieler 1", "Spieler 2"])
    parser.add_argument("--no-display", action="store_true",
                         help="Kein Live-Fenster (dann ist die Taschenansage fuer "
                              "die 8 per Tastatur nicht moeglich)")
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
    if args.out_w <= 0 or args.out_h <= 0:
        print(f"FEHLER: --out-w/--out-h muessen positiv sein (waren: "
              f"{args.out_w}/{args.out_h}).")
        sys.exit(1)
    if args.table_w_mm <= 0 or args.table_h_mm <= 0:
        print(f"FEHLER: --table-w-mm/--table-h-mm muessen positiv sein (waren: "
              f"{args.table_w_mm}/{args.table_h_mm}). Bei 0 oder negativ wuerden "
              f"alle mm-Umrechnungen (Bandenkontakt, Taschenerkennung) stillschweigend "
              f"falsch, ohne dass ein Fehler sichtbar wuerde.")
        sys.exit(1)
    if args.width <= 0 or args.height <= 0:
        print(f"FEHLER: --width/--height (Kamera-Aufloesung) muessen positiv "
              f"sein (waren: {args.width}/{args.height}).")
        sys.exit(1)
    if args.fps <= 0:
        print(f"FEHLER: --fps muss positiv sein (war: {args.fps}).")
        sys.exit(1)
    if args.picamera2 and args.device != 0:
        print(f"HINWEIS: --picamera2 ist gesetzt, --device wird dabei ignoriert "
              f"(nur fuer den V4L2/USB-Pfad relevant).")
    if not args.player_names[0].strip() or not args.player_names[1].strip():
        print(f"FEHLER: Spielernamen duerfen nicht leer sein (waren: "
              f"{args.player_names!r}).")
        sys.exit(1)
    if args.player_names[0].strip().lower() == args.player_names[1].strip().lower():
        print(f"HINWEIS: Beide Spieler heissen '{args.player_names[0]}' (oder "
              f"nur in Gross-/Kleinschreibung/Leerzeichen unterschiedlich) - die "
              f"Spielverfolgung selbst bleibt korrekt, aber die finale "
              f"Sieger-Meldung waere nicht mehr eindeutig lesbar. Empfehlung: "
              f"--player-names mit zwei klar unterschiedlichen Namen angeben.")

    corners = _load_corners(args.calibration, args.corners)
    table_config = TableConfig(corners_px=corners, table_w_mm=args.table_w_mm,
                                table_h_mm=args.table_h_mm)
    calib = TableCalibration(table_config, out_w=args.out_w, out_h=args.out_h)

    cam = OV9281Capture(device_index=args.device, width=args.width,
                         height=args.height, fps=args.fps,
                         use_picamera2=args.picamera2)
    detector = BallDetector(ball_radius_px=args.ball_radius_px)
    tracker = BallTracker(max_missing_frames=args.max_missing_frames,
                          match_dist_px=args.match_dist_px)
    shot_detector = ShotEventDetector(calib, table_config,
                                      ball_radius_px=args.ball_radius_px)
    notifier = NotificationManager()
    rules = RuleEngine(notifier, player_names=tuple(args.player_names))

    # -- Hintergrund aus mehreren ECHTEN Kameraframes lernen --------------
    print(f"Lerne Hintergrund aus {args.background_frames} echten Kameraframes.")
    print("WICHTIG: Der Tisch muss dabei komplett leer sein (keine Kugeln, keine Hand).")
    print("(Strg+C zum Abbrechen, auch waehrend dieser Phase moeglich)")
    MAX_CONSECUTIVE_FAILED_READS = 150  # ~1-5s je nach fps, statt endlos zu haengen
    try:
        learned = 0
        failed_reads = 0
        while learned < args.background_frames:
            frame = cam.read()
            if frame is None:
                failed_reads += 1
                if failed_reads >= MAX_CONSECUTIVE_FAILED_READS:
                    print(f"\nFEHLER: {MAX_CONSECUTIVE_FAILED_READS} Kameraframes in "
                          f"Folge fehlgeschlagen - Kamera getrennt oder abgestuerzt? "
                          f"Programm wird beendet.")
                    cam.release()
                    return
                continue
            failed_reads = 0
            topdown = calib.warp(frame)
            detector.bg_subtractor.apply(topdown, learningRate=0.5)
            learned += 1
    except KeyboardInterrupt:
        print("\nAbgebrochen waehrend der Hintergrund-Lernphase.")
        cam.release()
        return
    print("Hintergrund gelernt. Tisch kann jetzt aufgebaut werden.\n")

    if not args.no_display:
        print("Tastenbelegung im Live-Fenster:")
        print("  1-6 = Tasche fuer die angesagte 8 waehlen (vor dem Stoss auf die 8)")
        print("  c   = Taschenansage zuruecksetzen")
        pocket_map = _build_pocket_map_image(table_config)
        cv2.imshow("Taschen-Uebersicht", pocket_map)
        print("  q   = Programm beenden\n")

    shot_active = False
    last_time = time.time()
    consecutive_failed_reads = 0
    expected_dt = 1.0 / args.fps
    last_lag_warning = 0.0

    print("Starte Überwachung... (Strg+C bzw. 'q' im Fenster zum Beenden)")
    try:
        while rules.phase != GamePhase.GAME_OVER:
            frame = cam.read()
            if frame is None:
                consecutive_failed_reads += 1
                if consecutive_failed_reads >= MAX_CONSECUTIVE_FAILED_READS:
                    print(f"\nFEHLER: {MAX_CONSECUTIVE_FAILED_READS} Kameraframes in "
                          f"Folge fehlgeschlagen - Kamera getrennt oder abgestuerzt? "
                          f"Programm wird beendet.")
                    break
                continue
            consecutive_failed_reads = 0
            now = time.time()
            dt = now - last_time
            last_time = now

            # Verarbeitung faellt hinter die Kamera-Framerate zurueck? Ein
            # grosses dt wirkt auf das Tracking wie eine ploetzlich viel
            # schnellere Kugel (siehe Testprotokoll: uebersprungene Frames
            # genau waehrend eines Bandenabprallers koennen zur Fragmentierung
            # fuehren) - hier max. alle 5s eine Warnung, keine Frame-fuer-
            # Frame-Flut.
            if dt > 3 * expected_dt and now - last_lag_warning > 5.0:
                print(f"HINWEIS: Verarbeitung haengt hinterher ({dt*1000:.0f}ms "
                      f"statt erwarteter {expected_dt*1000:.0f}ms pro Frame) - "
                      f"evtl. --out-w/--out-h verkleinern, falls Kugeln bei "
                      f"schnellen Stoessen falsch zugeordnet werden.")
                last_lag_warning = now

            topdown = calib.warp(frame)
            detections = detector.detect(topdown)
            tracker.update(detections, detector.classify_group, dt)

            moving = not tracker.all_stationary(px_to_mm=calib.scale_x)

            if moving and not shot_active:
                shot_active = True
                shot_detector.reset()

            if shot_active:
                shot_detector.process_frame(tracker)

            if shot_active and not moving:
                shot_active = False
                rules.evaluate_shot(shot_detector.events)
                rules.called_pocket_for_eight = None  # Ansage gilt nur fuer 1 Stoss

            if not args.no_display:
                display = topdown.copy()
                remaining = rules.current_player.remaining_balls
                status = (f"Phase={rules.phase.name}  Dran={rules.current_player.name}"
                          f"  Eigene Kugeln uebrig={remaining if rules.current_player.group else '-'}"
                          f"  Angesagte Tasche(8)="
                          f"{rules.called_pocket_for_eight if rules.called_pocket_for_eight is not None else '-'}")
                cv2.putText(display, status, (8, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                # Auffaellige Erinnerung: eigene Gruppe fertig, aber noch keine
                # Tasche fuer die 8 angesagt -> jetzt waere der richtige Moment.
                if (remaining == 0 and rules.current_player.group is not None and
                        rules.called_pocket_for_eight is None):
                    cv2.putText(display, "JETZT TASCHE FUER DIE 8 ANSAGEN (1-6)!",
                                (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Live-Ueberwachung 8-Ball", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    rules.called_pocket_for_eight = None
                elif ord('1') <= key <= ord('6'):
                    idx = key - ord('1')
                    if 0 <= idx < len(table_config.pocket_positions_mm):
                        rules.called_pocket_for_eight = idx
                        print(f"Angesagte Tasche fuer die 8: Nr. {idx + 1}")

    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if rules.winner:
        print(f"\nSpielende – Gewinner: {rules.winner}")


if __name__ == "__main__":
    main()
