# Влияние серийных пропусков наблюдений на SORT и оценка дополнительной ассоциации

## Запуск SORT на настроечном видео

```bash
uv run --locked sort-gap-study run-sort \
  --seqmap /path/to/evaluate_tracking.seqmap.training --sequence 0000 \
  --source gt --detections /path/to/label_02/0000.txt \
  --mask burst --loss-fraction 0.5 --series-length 3 --seed 0 \
  --output runs/sort-0000-gt-burst
```

Для запуска без искусственных пропусков можно опустить параметры маски.
Для Regionlet укажите `--source regionlet` и файл соответствующего видео из
`det_02`. Для каждого запуска требуется новый каталог `--output`.

- `predictions/data/0000.txt` содержит оценки подтверждённых треков в формате
  KITTI Tracking, включая прогнозы на кадрах без наблюдений и оценки на кадрах прогрева.
- `events.jsonl` содержит по одной записи на исходный кадр. `artificial_gap`
  обозначает кадр маски, а `natural_empty` — доступный кадр без детекций после
  фильтрации входа. `matched` и `created` содержат пары
  `[ID трека, индекс поданной детекции]` с нулевой индексацией детекций.
  `removed` содержит ID удалённых треков. В `estimates` поле `observed`
  указывает, была ли использована детекция на текущем кадре.
- `mask.json`, `protocol.toml` и `manifest.json` сохраняют маску, копию
  настроек и сведения о входах и диапазоне кадров.
