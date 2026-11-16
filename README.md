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
Параметр `--method sort-extra` включает второе сопоставление после искусственных
пропусков. По умолчанию используется базовый SORT (`--method sort`).

Для OC-SORT подготовьте локальную копию репозитория.

```bash
git clone --no-checkout https://github.com/noahcao/OC_SORT.git external/OC_SORT
git -C external/OC_SORT checkout --detach 8462e7e729a93ccd3bd995c0a79a890336cb3a0b
```

Добавьте `--method ocsort` к команде `run-sort`. Путь к другой локальной копии
можно передать через `--ocsort-checkout`. Локальная копия в `external/OC_SORT`
также нужна для тестов `tests/test_ocsort.py`.

- `predictions/data/0000.txt` содержит оценки подтверждённых треков в формате
  KITTI Tracking, включая прогнозы на кадрах без наблюдений и оценки на кадрах прогрева.
- `events.jsonl` содержит по одной записи на исходный кадр. `artificial_gap`
  обозначает кадр маски, а `natural_empty` — доступный кадр без детекций после
  фильтрации входа. `matched` и `created` содержат пары
  `[ID трека, индекс поданной детекции]` с нулевой индексацией детекций.
  `removed` содержит ID удалённых треков, а `recovery_candidates` —
  ID треков, допущенных к дополнительной ассоциации на первом доступном кадре
  после паузы, включая случай пустого входа. `recovery_matches` содержит пары
  `[ID трека, индекс поданной детекции]`, найденные на втором этапе.
  В `estimates` поле `observed`
  указывает, была ли использована детекция на текущем кадре.
- `mask.json`, `protocol.toml` и `manifest.json` сохраняют маску, копию
  настроек и сведения о входах и диапазоне кадров.

## Оценка SORT в TrackEval

Один раз подготовьте локальную копию TrackEval. Можно использовать существующий
Git-репозиторий и передать его путь через `--trackeval`.

```bash
git clone --no-checkout https://github.com/JonathonLuiten/TrackEval.git external/TrackEval
git -C external/TrackEval checkout --detach 12c8791b303e0a0b50f753af204249e622d0281a
```

```bash
uv run --locked sort-gap-study evaluate-sort \
  --run runs/sort-0000-gt-burst \
  --ground-truth /path/to/label_02/0000.txt
```

`--ground-truth` указывает на полную разметку KITTI для того же видео при любом
источнике детекций. Команда берёт TrackEval из указанной ревизии Git и сохраняет
`metrics.csv` в каталоге запуска. Участок оценки исключает первые десять
кадров и включает кадры маски. IDF1, HOTA, DetA и AssA записываются в процентах.
Для последовательностей без оцениваемых Car в CSV указывается
`status=no_ground_truth` и оставляется пустое поле RFP.

Локальная копия TrackEval также нужна для запуска тестов `tests/test_evaluation.py`.
