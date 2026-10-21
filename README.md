# flying-detector

Обработчик видео: ищет летающие объекты (airplane, bird, kite из COCO) моделью `yolo11n.pt`.

## Установка

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Веса `yolo11n.pt` ultralytics скачает сам при первом запуске, либо положите файл рядом со скриптом.

## Запуск

```bash
.venv/bin/python detector.py \
  --inbox "/mnt/обменник/Drone" \
  --found data/found_videos \
  --frames data/found_frames
```

- просмотренные видео записываются в `processed.json` и повторно не обрабатываются;
- видео с находкой переносится в `--found`;
- кадры с находкой сохраняются в `--frames/<имя видео>/` без рамок и подписей;
- одинаковые кадры с неподвижным объектом сохраняются один раз (пороги `--frame-diff`, `--move-px`);
- `--watch 60` проверяет папку каждую минуту, `--stride 3` анализирует каждый 3-й кадр, `--device 0` запускает на GPU.

Класса «дрон» в стандартной модели нет: дроны ловятся как airplane/bird/kite.
Для надёжного поиска дронов модель нужно дообучить на своих кадрах.
