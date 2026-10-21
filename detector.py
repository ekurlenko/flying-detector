#!/usr/bin/env python3
"""Обработчик видео: ищет летающие объекты моделью YOLO11n.

1. Берёт видео из общей папки по порядку; просмотренные запоминает и больше не смотрит.
2. Прогоняет кадры через yolo11n.pt, учитывая только летающие классы COCO.
3. Видео с находкой переносит в отдельную папку.
4. Кадры с находкой сохраняет отдельно, без рамок и подписей.
5. Одинаковые кадры с неподвижным объектом не дублирует: хватает одного.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Классы датасета COCO, на котором обучена стандартная yolo11n.pt.
# Класса «дрон» в COCO нет: дрон обычно распознаётся как airplane, bird или kite.
FLYING_CLASSES = {4: "airplane", 14: "bird", 33: "kite"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".ts"}


def parse_args():
    p = argparse.ArgumentParser(description="Поиск летающих объектов на видео (YOLO11n)")
    p.add_argument("--inbox", required=True, type=Path, help="общая папка с видео")
    p.add_argument("--found", required=True, type=Path, help="куда переносить видео с находками")
    p.add_argument("--frames", required=True, type=Path, help="куда сохранять кадры с находками")
    p.add_argument("--model", default="yolo11n.pt", help="путь к весам YOLO")
    p.add_argument("--state", type=Path, default=Path("processed.json"), help="файл со списком просмотренных видео")
    p.add_argument("--conf", type=float, default=0.35, help="минимальная уверенность детекции")
    p.add_argument("--stride", type=int, default=3, help="анализировать каждый N-й кадр")
    p.add_argument("--frame-diff", type=float, default=4.0,
                   help="порог отличия кадров (средняя разница яркости 0-255), ниже — кадры одинаковые")
    p.add_argument("--move-px", type=float, default=0.02,
                   help="порог смещения объекта (доля диагонали кадра), ниже — объект стоит на месте")
    p.add_argument("--min-age", type=float, default=10,
                   help="не трогать файлы моложе N секунд (их ещё может копировать шара)")
    p.add_argument("--watch", type=float, default=0,
                   help="если > 0, проверять папку заново каждые N секунд")
    p.add_argument("--device", default=None, help="cpu, 0 (GPU) и т.п.; по умолчанию выбирает ultralytics")
    return p.parse_args()


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_state(path: Path, state: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def video_key(path: Path, inbox: Path) -> str:
    st = path.stat()
    return f"{path.relative_to(inbox)}|{st.st_size}|{int(st.st_mtime)}"


def pending_videos(inbox: Path, state: dict, min_age: float):
    now = time.time()
    for path in sorted(inbox.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        if now - path.stat().st_mtime < min_age:
            continue
        if video_key(path, inbox) in state:
            continue
        yield path


def thumb(frame):
    """Маленькая серая копия кадра для сравнения: шум и мелкие колебания не мешают."""
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def box_centers(boxes, w, h):
    xyxy = boxes.xyxy.cpu().numpy()
    centers = np.column_stack(((xyxy[:, 0] + xyxy[:, 2]) / 2 / w, (xyxy[:, 1] + xyxy[:, 3]) / 2 / h))
    return centers


def max_shift(prev, cur):
    """Наибольшее смещение объекта относительно ближайшего объекта на прошлом сохранённом кадре."""
    if prev is None or len(prev) != len(cur):
        return float("inf")
    dists = np.linalg.norm(cur[:, None, :] - prev[None, :, :], axis=2)
    return float(dists.min(axis=1).max())


def process_video(model, path: Path, args) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"status": "error", "reason": "не удалось открыть"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames_dir = args.frames / path.stem
    last_thumb = last_centers = None
    saved, hits, idx = 0, 0, -1
    classes_seen = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % args.stride:
            continue

        result = model.predict(frame, classes=list(FLYING_CLASSES), conf=args.conf,
                               device=args.device, verbose=False)[0]
        if not len(result.boxes):
            continue
        hits += 1
        classes_seen.update(FLYING_CLASSES[int(c)] for c in result.boxes.cls.tolist())

        h, w = frame.shape[:2]
        cur_thumb = thumb(frame)
        cur_centers = box_centers(result.boxes, w, h)
        if last_thumb is not None:
            diff = float(cv2.absdiff(cur_thumb, last_thumb).mean())
            if diff < args.frame_diff and max_shift(last_centers, cur_centers) < args.move_px:
                continue  # тот же кадр, объект не сдвинулся

        frames_dir.mkdir(parents=True, exist_ok=True)
        # Сохраняем исходный frame, а не result.plot(): так на кадре нет рамок и подписей.
        cv2.imwrite(str(frames_dir / f"{path.stem}_{idx:06d}_{idx / fps:08.2f}s.jpg"), frame)
        saved += 1
        last_thumb, last_centers = cur_thumb, cur_centers

    cap.release()

    if hits:
        target = args.found / path.name
        if target.exists():
            target = args.found / f"{path.stem}_{int(time.time())}{path.suffix}"
        shutil.move(str(path), target)
        return {"status": "found", "classes": sorted(classes_seen), "frames": saved, "moved_to": str(target)}
    return {"status": "empty"}


def run_once(model, args) -> int:
    state = load_state(args.state)
    count = 0
    for path in pending_videos(args.inbox, state, args.min_age):
        key = video_key(path, args.inbox)
        print(f"→ {path.relative_to(args.inbox)} ... ", end="", flush=True)
        started = time.time()
        info = process_video(model, path, args)
        info["seconds"] = round(time.time() - started, 1)
        print(info)
        state[key] = info
        save_state(args.state, state)
        count += 1
    return count


def main():
    args = parse_args()
    if not args.inbox.is_dir():
        sys.exit(f"Нет папки {args.inbox}")
    args.found.mkdir(parents=True, exist_ok=True)
    args.frames.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    while True:
        n = run_once(model, args)
        if not args.watch:
            print(f"Готово, обработано видео: {n}")
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
