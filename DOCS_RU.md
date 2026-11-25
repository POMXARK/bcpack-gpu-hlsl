# bcpack-gpu-hlsl — GPU-декомпрессор Xbox BCPack на DirectX 12 HLSL SM 6.6

Чистая (clean-room) реализация декодера **Xbox Series BCPack (BC1)** на вычислительных шейдерах **DirectX 12 HLSL Shader Model 6.6 (`WaveSize(32)`)**, работающая со скоростью **3 383.6 MB/s** на RTX 5080 (в **77.3 раза быстрее** 1 ядра CPU и в **11.8 раза быстрее** многопоточного CPU), с побайтовой сверкой против [`Ahmet-Dedeler/bcpack-rs`](https://github.com/Ahmet-Dedeler/bcpack-rs).

---

## 1. Открытые проекты по BCPack и исследования GPU-декомпрессии

По состоянию на ноябрь 2026 года публичных GPU Compute-декодеров BCPack в открытом доступе нет — существующие проекты декодируют BCPack на CPU:
- [`Ahmet-Dedeler/bcpack-rs`](https://github.com/Ahmet-Dedeler/bcpack-rs) (`MIT OR Apache-2.0`) — спецификация `BCPACK_SPEC.md`, точный rANS (порт `microsoft/silicon-rans`), полный `BC1`, синтетический энкодер;
- [`ifBars/rage-texture`](https://github.com/ifBars/rage-texture) (`GPL-3.0`) — CPU-экстрактор/декодер `BC1/3/4/5/6H/7`;
- [`Tim0219800/bcpack-decoder`](https://github.com/Tim0219800/bcpack-decoder) — CPU-декодер `BC1`;
- [`xcl-dev/xcl`](https://github.com/xcl-dev/xcl) / `Prime1997/xcl-compat-fixes` — слой совместимости Xbox Series → Windows (RDNA2→HLSL, UMA, D3D12).

Ключевые работы по GPU-декомпрессии: **Microsoft DirectStorage 1.1+ (GDeflate) / 1.4 (Zstd)**, **NVIDIA nvCOMP + GPUDirect Storage**, **Weißenberger & Schmidt** (`weissenberger/multians`), **Recoil** ([arXiv:2306.12141](https://arxiv.org/abs/2306.12141)), **Meta DietGPU** (`facebookresearch/dietgpu`), **Fabian Giesen** (`rygorous/ryg_rans`), **Microsoft `silicon-rans`** (`mdu_minirans.v`, US11234023B2), **CODAG**, **Falcon**, **dtANS** ([arXiv:2603.01915](https://arxiv.org/abs/2603.01915)).

---

## 2. Ключевые проблемы и найденные решения

1. **Принцип Xbox MDU на видеокарте («на лету» без перепаковки 200+ ГБ архивов):**
   - Внутри одного чанка `BC1` 12 линий rANS работают синхронно на общем потоке байт (окно подачи до 8 байт/такт). Поэтому **1 Wave32-варп = 1 чанк** (`[WaveSize(32)] [numthreads(32, 1, 1)]`), где потоки `0..11` точно повторяют MDU-линии.
   - Сотни независимых чанков открытого мира запускаются одним `Dispatch(num_chunks, 1, 1)` на всех SM видеокарты, полностью скрывая последовательную задержку одного чанка (~17.5 мс для 16 384 блоков практически не меняется от 64 до 512 чанков).
2. **Обойдённый баг DXC 1.8 (`-T cs_6_6 -O3`):**
   - Связка `WavePrefixCountBits(want_byte)` + `WaveActiveCountBits(valid)` при агрессивной оптимизации `-O3` давала сбой переупорядочивания на `stream_pos` (при `-Od` всё проходило).
   - Замена на два явных `WaveActiveBallot` + `countbits(mask & ((1u << lane) - 1u))` полностью устранила проблему под `-O3` и ускорила код.
3. **Отсутствие промежуточных буферов в VRAM:**
   - Символы передаются между линиями напрямую через `WaveReadLaneAt(slot_sym, k)`.
   - Кольцевой буфер `groupshared uint ring_ep[256]` заранее заполняется 256-слотовым пресетным окном `g_Window`, что объединяет pre-start ссылки (`d > block_idx`) и back-reference (`i - d`) в одно чтение без ветвлений.
   - Распакованные блоки `BC1` копируются через `CopyTextureRegion` сразу в `DXGI_FORMAT_BC1_UNORM` текстуры.

---

## 3. Верификация и замеры (NVIDIA GeForce RTX 5080, Driver 621.24)

- `--verify`: **PASS (`0` расхождений)** на 5 синтетических чанках (21 824 блока) и 8 процедурных текстурах 512×512 `.bcpack` (1.00 MiB BC1).
- `--bench` (1 чанк = 128 KiB BC1):
  - **64 чанка (8.0 MiB):** CPU 1T `188.03 мс (42.5 MB/s)` | CPU MT `30.77 мс (260.0 MB/s)` | **GPU Kernel `17.550 мс (455.8 MB/s)`** (`10.7x` / `1.8x`)
  - **256 чанков (32.0 MiB):** CPU 1T `735.02 мс (43.5 MB/s)` | CPU MT `118.21 мс (270.7 MB/s)` | **GPU Kernel `17.780 мс (1799.8 MB/s)`** (`41.3x` / `6.6x`)
  - **512 чанков (64.0 MiB):** CPU 1T `1462.41 мс (43.8 MB/s)` | CPU MT `222.55 мс (287.6 MB/s)` | **GPU Kernel `18.915 мс (3383.6 MB/s)`** (`77.3x` / `11.8x`)
- **Живой FPS в 3D-окне 1280×720** (полная распаковка всех чанков на лету каждый кадр):
  - `bcpack-demo-cpu.exe` (1 ядро, 64 чанка = 8 MiB/кадр): **5.32 FPS** (`186.55 мс`)
  - `bcpack-demo-cpu.exe --mt` (все ядра, 64 чанка = 8 MiB/кадр): **29.87 FPS** (`31.68 мс`)
  - `bcpack-demo-gpu.exe` (GPU RTX 5080, 64 чанка = 8 MiB/кадр): **45.81 FPS** (`17.44 мс`)
  - `bcpack-demo-gpu.exe --chunks 256` (GPU RTX 5080, 256 чанков = 32 MiB/кадр): **44.36 FPS** (`17.53 мс`)

---

## 4. Сборка и запуск

```bat
git clone https://github.com/POMXARK/bcpack-gpu-hlsl.git
cd bcpack-gpu-hlsl
cargo build --release

target\release\bcpack-gpu-demo.exe --verify --bench
target\release\bcpack-demo-cpu.exe
target\release\bcpack-demo-gpu.exe
```

Управление в окне: `UP`/`DOWN` — нагрузка (`8..256` чанков/кадр), `M` — `1 ядро ↔ все ядра` (CPU), `SPACE` — `GPU ↔ CPU` (в `bcpack-gpu-demo --demo`), `ESC` — выход.
