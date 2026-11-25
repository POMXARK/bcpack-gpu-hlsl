# bcpack-gpu-hlsl v0.2.0 — GPU-декомпрессор Xbox BCPack (Dual-Chunk Wave32) и игровая 3D-сцена

Чистая (clean-room) реализация декодера **Xbox Series BCPack (BC1)** на вычислительных шейдерах **DirectX 12 HLSL Shader Model 6.6 (`WaveSize(32)`)**, достигающая **9 877.1 MB/s (9.88 ГБ/с — в 227.2 раза быстрее 1 ядра CPU и в 34.7 раза быстрее многопоточного CPU)** на RTX 5080, с побайтовой сверкой против [`Ahmet-Dedeler/bcpack-rs`](https://github.com/Ahmet-Dedeler/bcpack-rs).

---

## Что нового в v0.2.0

1. **Dual-Chunk Wave32 (75% загрузка варпа: 24/32 потока):**
   - Потоки `0..11` декодируют чанк A, потоки `16..27` — чанк B параллельно в одном варпе.
2. **Оптимизация памяти и ALU в `shaders/bcpack_decode.hlsl`:**
   - Таблицы моделей ужаты до **80 байт** (`bf = base | (freq << 16)`, `rfrsn = rf | (rs << 16) | (rn << 21)` — `thr[10]` убран).
   - На `HDR1` модель переносится в транспонированный `groupshared`-кэш (`lane_bf[10][24]`, `lane_rfrsn[10][24]`) — **ноль глобальных чтений** при декодировании символов.
   - Коалесцированный 64-байтный кэш потока `groupshared uint stream_cache[2][16]`.
   - Амортизированная проверка прогресса/зависаний раз в 64 такта.
   - Обойдён баг `DXC 1.8 -O3` с `WavePrefixCountBits` заменой на явный `WaveActiveBallot` + `countbits`.
3. **4K-тайловый стриминг + игровая 3D-сцена Coastal Highway (`media/demo_frame_v2.png`):**
   - **12 текстур 512×512 BC1**, каждая разбита на 4 тайла по 4 096 блоков (`32 KiB`).
   - Красный спорткар игрока + **10 машин трафика** на 4-полосном шоссе, изогнутые 3D-пальмы, фонарные столбы, набережная, пляж, анимированный океан, многоуровневые здания Art-Deco, закатное небо `sky.hlsl` и 4 динамических источника света.
   - Управление камерой клавишей **`C`**: **Chase Cam** / **Free Drive (`W/A/S/D` + `Q/E`)** / **Helicopter Orbit**.

---

## Замеры на NVIDIA GeForce RTX 5080 (Driver 621.24)

- `--verify`: **PASS (`0` расхождений)** на 5 синтетических потоках (21 824 блока), 12 монолитных текстурах (1.50 MiB BC1) и 48 тайлах 4K (1.50 MiB BC1).
- `--bench` (4K-тайловые чанки по 32 KiB BC1):
  - **256 тайлов (8.0 MiB):** CPU 1T `187.96 мс (42.6 MB/s)` | CPU MT `31.10 мс (257.3 MB/s)` | **GPU `5.321 мс (1 503.6 MB/s)`** (**`35.3x` / `5.8x`**)
  - **1024 тайла (32.0 MiB):** CPU 1T `744.06 мс (43.0 MB/s)` | CPU MT `111.53 мс (286.9 MB/s)` | **GPU `5.576 мс (5 739.3 MB/s)`** (**`133.5x` / `20.0x`**)
  - **2048 тайлов (64.0 MiB):** CPU 1T `1471.99 мс (43.5 MB/s)` | CPU MT `224.60 мс (284.9 MB/s)` | **GPU `6.480 мс (9 877.1 MB/s)`** (**`227.2x` / `34.7x`**)
- **Живой FPS в 3D-окне 1280×720** (полная распаковка на лету каждый кадр):
  - `bcpack-demo-cpu.exe` (1 ядро, 256 тайлов = 8.0 MiB/кадр): **5.26 FPS** (`189.00 мс`)
  - `bcpack-demo-cpu.exe --mt` (все ядра, 256 тайлов = 8.0 MiB/кадр): **31.14 FPS** (`29.71 мс`)
  - `bcpack-demo-gpu.exe` (GPU RTX 5080 v0.2, 256 тайлов = 8.0 MiB/кадр): **101.81 FPS** (`5.28 мс`)
  - `bcpack-demo-gpu.exe --chunks 1024` (GPU RTX 5080 v0.2, **32.0 MiB/кадр**!): **94.85 FPS** (`5.53 мс`)

---

## Сборка и запуск

```bat
git clone https://github.com/POMXARK/bcpack-gpu-hlsl.git
cd bcpack-gpu-hlsl
cargo build --release

target\release\bcpack-gpu-demo.exe --verify --bench
target\release\bcpack-demo-cpu.exe
target\release\bcpack-demo-gpu.exe
```

Клавиши в окне: `C` — камера (Chase / Free `WASD+QE` / Orbit), `UP`/`DOWN` — нагрузка (`48..1024` тайлов = `1.5..32 MiB/кадр`), `M` — `1 ядро ↔ все ядра` (CPU), `SPACE` — `GPU ↔ CPU` (в `bcpack-gpu-demo --demo`), `ESC` — выход.
