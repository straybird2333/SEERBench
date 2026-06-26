# Data Organization

The first code release does not include data. The benchmark data should be packaged separately for Baidu Netdisk.

## 1. Annotations

Target path:

```text
Benchmark_annotations/
```

Contents:

- One JSON file per benchmark task.
- Paths inside JSON should be relative, for example `data/egoexo4d_val/<scene>/<view>/<frame>.jpg`.
- Include a `statistics.json` summary if available.

## 2. Images

Target path:

```text
data/egoexo4d_val/
```

Recommended archive name:

```text
seerbench_images.tar.gz
```

Expected layout after extraction:

```text
data/egoexo4d_val/
  <scene_id>/
    aria*_214-1/
      000000.jpg
    cam01/
      000000.jpg
    cam02/
      000000.jpg
```

## 3. Optional Inspection Assets

Target path:

```text
Inspect_data/
```

These files are useful for manual inspection but are not required for evaluation.

## 4. Files To Exclude

Do not include these in public data packages:

- Model checkpoints and local model caches.
- Python virtual environments.
- `__pycache__`, logs, and runtime results.
- Raw full video dumps unless explicitly intended for release.
- Personal questionnaire submissions or other non-public annotation metadata.

## 5. Suggested Netdisk Release

Recommended packages:

```text
seerbench_annotations.tar.gz
seerbench_images.tar.gz
seerbench_inspection_optional.tar.gz
checksums.sha256
```

Keep `checksums.sha256` next to the archives so users can verify downloads.

The README has a reserved field:

```text
Baidu Netdisk: TODO: add link here
```
