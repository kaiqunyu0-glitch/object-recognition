# 在 Kaggle 上用 COCO 2017 訓練 YOLO11x

筆記本是 [`train_coco_yolo11x.ipynb`](train_coco_yolo11x.ipynb)。這條路徑和手機照片的預設流程是分開的。手機偵測的預設模型仍是 `yolo11m`。這裡才用最大的 **YOLO11x**。

不要在這個儲存庫裡登入 Kaggle，也不要在這裡真的下載 COCO 或開訓。筆記本是給 Kaggle 工作階段執行的。

## 預設是微調

筆記本最上面的參數格：

```python
MODE = "finetune"
```

`finetune` 會載入官方預訓練權重 `yolo11x.pt`（已在 COCO 上訓練過）。這是預設，也是建議的做法。

改成從零訓練只有一個開關：

```python
MODE = "scratch"
```

`scratch` 使用 `yolo11x.yaml`，`pretrained=False`，權重從隨機值開始。Ultralytics 這種大型偵測模型的完整 COCO 課表通常是數百輪。以 Kaggle **每週大約 30 小時**的 GPU 額度（官方說法是 30 小時，有時因供需再高一點）來看，從零訓練不是一個 12 小時工作階段能做完的事，而是很多週的額度。數字在下面的估計表。

## 估計，不是實測

下面的時間**沒有在這台機器或 Kaggle 上量過**。它是一組寫明的假設，方便你排工作階段。提早停止（patience、驗證損失平台、mAP 目標）可能讓實際輪數少於上限，所以表上的總時數是「跑滿上限」的估計，不是保證。

假設：

- YOLO11x 在 640 的前向計算量是 194.9 GFLOP（Ultralytics 模型表）。反向大約再 2 倍，一張圖約 585 GFLOP。
- 再乘 1.3，拿來粗略含 mosaic、讀圖，以及每輪 5,000 張驗證圖。COCO 2017 訓練集是 118,287 張。
- 一張 T4 的 FP16 Tensor Core 峰值約 65 TFLOPS，假設訓練只用到 20%（13 TFLOPS）。兩張 T4 假設是 1.7 倍而不是 2 倍（DDP 有通訊）。
- 一張 P100 沒有 Tensor Core，FP16 峰值約 18.7 TFLOPS，假設用到 30%（5.6 TFLOPS）。
- 四捨五入之後：T4 x2 **每輪 1.1 小時**，P100 **每輪 4.5 小時**。利用率若差一截，上下 40% 都算合理。
- 每個 12 小時工作階段留 1 小時做啟動、收尾驗證和存檔，所以可訓練時間按 **11 小時**算。第一輪還要下載 COCO，可能再吃掉 1 到 2 小時，表內沒有把下載算進訓練。
- 微調上限 100 輪，從零上限 300 輪。
- 每週額度按 **30 小時**除。

| 機器 | 模式 | 每輪（估計） | 輪數上限 | 總小時 | 12 小時工作階段 | 30 小時額度要幾週 |
| --- | --- | --- | --- | --- | --- | --- |
| T4 x2 | 微調 `yolo11x.pt`（預設） | 1.1 小時 | 100 | 110 | 10 | 約 3.7 週 |
| T4 x2 | 從零 `yolo11x.yaml` | 1.1 小時 | 300 | 330 | 30 | **11 週** |
| P100 | 微調 `yolo11x.pt`（預設） | 4.5 小時 | 100 | 450 | 41 | 15 週 |
| P100 | 從零 `yolo11x.yaml` | 4.5 小時 | 300 | 1350 | 123 | **45 週** |

從零訓練非常久。T4 x2 上 300 輪大約是 330 小時，也就是大約 11 週的額度、30 次工作階段。P100 上大約 1350 小時、45 週、123 次工作階段。預設請維持 `finetune`。

Kaggle 文件仍列出 P100（1 張 GPU、4 核 CPU、約 29 GB 記憶體）和 T4 x2（2 張 T4、4 核、約 29 GB）。有些 2026 的第二手文章說 P100 已下線。筆記本兩種都留著；以你帳號的 Accelerator 選單為準。

## 顯示卡、batch、AMP

兩張 T4 是兩張 16 GB 的卡，不是一張 32 GB。`device=0,1` 會走 Ultralytics 的 DDP。`batch` 是兩張卡加起來的張數，而且必須大於等於 GPU 數量。

| 設定 | T4 x2 | P100 |
| --- | --- | --- |
| `device` | `0,1` | `0` |
| `batch` | 8（每張卡 4 張圖） | 4 |
| `imgsz` | 640 | 640 |
| `workers` | 2 | 2 |
| `amp` | True | True |

16 GB 對 YOLO11x 很緊。8 和 4 是保守值。第一輪若 CUDA out of memory，把 batch 減半。不要在雙 GPU 上用 `batch=-1`：Ultralytics 的 AutoBatch 只支援單卡。

T4 有 Tensor Core，AMP 很重要。P100 的 FP16 沒有 Tensor Core，所以表上更慢。AMP 若出現 NaN，把 `AMP` 設成 `False` 並再把 batch 減半。

`workers=2` 是因為機器只有 4 個 CPU 核心、記憶體大約 29 GB。COCO 解圖很吃記憶體，先不要開到 8。

## 磁碟

這是假設，請在筆記本裡看 `df` 的輸出再確認。

- Kaggle 文件寫 `/kaggle/working` 最多存下大約 **20 GB**，而且這是工作階段結束後還會留下的輸出。
- `/kaggle/working` 以外的空間是暫存，工作階段結束就會被清掉。文件沒有保證一定有一個叫 `/kaggle/temp` 的獨立磁碟。筆記本預設把資料集放在 `/kaggle/temp/coco`，會建立這個目錄並檢查剩餘空間。若它和很小的系統碟是同一塊，檢查會拒絕下載。
- Ultralytics 的 `coco.yaml` 會下載 `train2017.zip`（約 19 GB）和 `val2017.zip`（約 1 GB），再加上標註。下載函式預設 **不刪 zip**（`delete=False`），所以 zip 和解開的圖會同時在，尖峰大約 **40 GB**。這放不進 20 GB 的輸出目錄。
- 剩餘空間少於約 45 GB、又沒有附上現成資料集時，筆記本會停下來，不開始下載。
- `last.pt` 和 `best.pt` 含 optimizer 狀態，YOLO11x 每個檔大約幾百 MB 到 1 GB，放得進 `/kaggle/working`。`save_period=10` 會多存 `epochN.pt`，檔案一樣大；輸出快滿 20 GB 時把 `SAVE_PERIOD` 改成 0。
- 沒有查證過某個公開的 Kaggle COCO 資料集代號。若你已經附加了一份 Ultralytics 版面的 COCO（`images/train2017` 或 `train2017`），把 `COCO_INPUT` 設成 `/kaggle/input/...`，筆記本就不會再下載。版面不對會直接報錯，不會猜代號。

## 怎麼跑

1. 在 Kaggle 開一個 GPU 筆記本（預設機器選 T4 x2）。開啟網路，COCO 和 `yolo11x.pt` 都要下載。
2. 把這個儲存庫放到筆記本看得到的地方，因為停止條件在 `src/detect/coco_early_stop.py`，測試也是測這支檔案。例如在有網路的格子裡：

```bash
git clone https://github.com/kaiqunyu0-glitch/object-recognition.git /kaggle/working/object-recognition
```

合併進預設分支之前，請 clone 含有這個檔案的那個版本。然後確認參數格的 `REPO_ROOT` 指向它。

3. 上傳 `train_coco_yolo11x.ipynb`，或在 clone 出來的目錄裡打開它。
4. 參數格維持 `MODE = "finetune"`，除非你確定要從零訓練。
5. 用 **Save & Run All** 跑完整個筆記本。互動工作階段閒置大約 20 分鐘會斷；Save & Run All 會自己跑完，仍受 12 小時限制。
6. 結束後確認 `/kaggle/working/last.pt` 和 `/kaggle/working/best.pt` 都在。權重目錄裡也有一份：`/kaggle/working/coco-yolo11x/train/weights/`。

## 12 小時後怎麼續訓

Kaggle 一個 GPU 工作階段最長約 12 小時。`TIME_HOURS = 11` 會在被系統殺掉之前停下來，留下檔案。Ultralytics 的 `time` 參數會在輪與輪之間、以及訓練迴圈裡檢查時鐘；回呼也會在每輪結束再檢查一次。

下一輪：

1. 把這個筆記本的輸出存成 Kaggle Dataset。Save Version 之後，在新筆記本的 Input 裡 Add Input，選這個筆記本的輸出，或先用輸出建一個 Dataset。
2. 檔案會出現在唯讀的 `/kaggle/input/<名稱>/`。
3. 把參數格的 `RESUME_LAST` 設成那個 `last.pt`，例如 `/kaggle/input/coco-yolo11x-ckpt/last.pt`。
4. 筆記本會把它複製到 `/kaggle/working`，因為下一份 checkpoint 必須寫在可寫的目錄。然後用 `resume=True` 接著訓練。
5. COCO 在暫存碟上，工作階段結束就沒了。下一輪要嘛再下載，要嘛重新附加資料集。`resume=True` 仍會採用你這次傳入的 `data`，所以 yaml 可以指向新的路徑。`patience`、`time`、`batch`、`device`、`workers`、`save_period` 也可以在續訓時改；其他超參數 Ultralytics 會沿用 checkpoint 裡的值。

續訓時回呼會重新掛上，所以 mAP 目標和損失平台仍然有效。Ultralytics 自己的 patience 狀態在 `last.pt` 裡。

## 結果可以就停

兩層都開著。

1. **Ultralytics patience**（預設 `PATIENCE = 15`，落在 10 到 20）。它看的是 fitness。Ultralytics 8.4 的偵測 fitness 就是驗證 **mAP50-95**（權重為 1）。較舊的文件曾用 `0.1 * mAP50 + 0.9 * mAP50-95`。patience 設成 0 會關掉這層。
2. **自訂回呼** `AcceptableResultStop.on_fit_epoch_end`，規則在 `src/detect/coco_early_stop.py`：
   - 驗證損失 `box + cls + dfl` 連續 `LOSS_PATIENCE` 輪（預設 15）都沒有下降至少 `LOSS_MIN_DELTA`（預設 0.01）。第一輪只建立基準，不算停滯。驗證沒跑到的輪不計入。
   - 驗證 mAP50-95 達到 `MAP_TARGET`（預設 0.55）。官方 YOLO11x 在 COCO val 大約是 0.547，所以 0.55 的意思是「接近公布的水準」。設成 0.50 的話，預訓練微調可能第一輪驗證就停。設成 `None` 就關掉這個目標。
   - 牆鐘超過 `TIME_HOURS`（預設 11）。

優先順序是時間、然後 mAP、然後損失平台。回呼在該輪的驗證和 `save_model` **之後**才把 `trainer.stop` 設成 True，並把 `epochs` 收到剛剛跑完的那一輪，所以這一輪的驗證和存檔會留下，接著 Ultralytics 還會對 `best.pt` 做最後一次驗證。`last.pt` 和 `best.pt` 寫在 `/kaggle/working/coco-yolo11x/train/weights/`，筆記本再複製到 `/kaggle/working/last.pt` 和 `/kaggle/working/best.pt`。

多 GPU：這個回呼只在 rank -1（單卡）和 rank 0 做決定。Ultralytics 8.4 會在 `on_fit_epoch_end` 之後把 `trainer.stop` 從 rank 0 廣播出去，而且啟動 DDP 時會把回呼 pickle 進每個 worker。若以後的版本不再廣播，或 pickle 失敗，就改 `GPU = "p100"` 或把 `DEVICE` 設成 `"0"` 只用一張卡，或者只靠 `patience`。

## English

The notebook [`train_coco_yolo11x.ipynb`](train_coco_yolo11x.ipynb) trains **YOLO11x** on COCO 2017 on Kaggle. The phone-photo default in this repo stays `yolo11m`. Do not log into Kaggle from this repo, and do not run the COCO training here. The notebook is meant to run inside a Kaggle session.

The default switch at the top of the notebook is `MODE = "finetune"`, which loads official pretrained `yolo11x.pt`. `MODE = "scratch"` trains `yolo11x.yaml` from random weights. Scratch training takes a very long time under Kaggle's weekly GPU quota of **about 30 hours**.

### Estimates, not measurements

These times were not measured on Kaggle or in this repo. Early stopping can finish sooner, so the totals are the cost of using the whole epoch cap.

Assumptions: YOLO11x is 194.9 GFLOP per 640 image forward (Ultralytics table). Backward is taken as 2× forward. A further 1.3× covers mosaic, loading, and the 5,000-image validation. COCO train2017 has 118,287 images. One T4 is assumed at 20% of its ~65 FP16-tensor TFLOPS. Two T4s are assumed at 1.7× one T4, not 2×. A P100 has no tensor cores and is assumed at 30% of its ~18.7 FP16 TFLOPS. Rounded point estimates: **1.1 h/epoch on T4 x2**, **4.5 h/epoch on P100**. A ±40% band is reasonable because utilization was not measured. Each 12-hour session counts as 11 training hours. The weekly quota is divided by 30. Fine-tune is capped at 100 epochs, scratch at 300. The first session also spends time downloading COCO; that is not included.

| Machine | Mode | Hours/epoch | Epoch cap | Total hours | 12h sessions | Weeks of a 30h quota |
| --- | --- | --- | --- | --- | --- | --- |
| T4 x2 | fine-tune `yolo11x.pt` (default) | 1.1 | 100 | 110 | 10 | about 3.7 |
| T4 x2 | from scratch `yolo11x.yaml` | 1.1 | 300 | 330 | 30 | **11** |
| P100 | fine-tune `yolo11x.pt` (default) | 4.5 | 100 | 450 | 41 | 15 |
| P100 | from scratch `yolo11x.yaml` | 4.5 | 300 | 1350 | 123 | **45** |

From-scratch training is the long one: about 330 hours on T4 x2 (11 weeks of quota, 30 sessions) and about 1,350 hours on a P100 (45 weeks, 123 sessions). Leave `MODE` on `finetune`.

Kaggle's notebook docs still list a P100 (1 GPU, 4 CPU cores, about 29 GB RAM) and T4 x2 (two T4s, 4 cores, about 29 GB RAM). Some 2026 secondary writeups say the P100 was retired. The notebook supports both. Use whichever accelerator the account actually offers. The two T4s are separate 16 GB cards. `device=0,1` is DDP. `batch` is the total across GPUs and must be at least the GPU count. Defaults are batch 8 on T4 x2 (4 per GPU) and batch 4 on a P100, `imgsz` 640, `workers` 2, AMP on. Halve the batch if the first epoch runs out of memory. Do not use `batch=-1` on two GPUs; Ultralytics AutoBatch is single-GPU only.

### Disk

`/kaggle/working` keeps about 20 GB of output after the session. Anything outside it is scratch and is wiped. The docs do not promise a mount named `/kaggle/temp`; the notebook creates `/kaggle/temp/coco` and checks free space. Ultralytics `coco.yaml` downloads the ~19 GB train zip and ~1 GB val zip and, in the current helper, does not delete the zip after unzip, so peak use is about 40 GB. That does not fit in the 20 GB output folder. The notebook refuses to download when free space is under about 45 GB. `last.pt` and `best.pt` are a few hundred MB to about 1 GB each and do fit in `/kaggle/working`. Set `SAVE_PERIOD = 0` if extra epoch checkpoints crowd that 20 GB. No public Kaggle COCO dataset slug was verified. Set `COCO_INPUT` only when you attached a dataset that already has the Ultralytics layout (`images/train2017`).

### Resume across the 12-hour limit

`TIME_HOURS = 11` stops the run with time left to save. Afterward, `last.pt` and `best.pt` are in `/kaggle/working` and under `coco-yolo11x/train/weights/`. Save the notebook output as a Kaggle Dataset (Save Version, then Add Input on the next notebook, or create a dataset from that output). The files show up read-only under `/kaggle/input/<name>/`. Set `RESUME_LAST` to that `last.pt`. The notebook copies it into `/kaggle/working` and continues with `resume=True`. COCO was on scratch, so download it again or attach it again. Passing `data` on resume points at the new copy. `patience`, `time`, `batch`, `device`, `workers`, and `save_period` can change on resume; Ultralytics keeps the other hyperparameters from the checkpoint.

### Stopping when the result is acceptable

`PATIENCE = 15` is Ultralytics early stopping on fitness. In Ultralytics 8.4 that fitness is validation mAP50-95. The callback `AcceptableResultStop.on_fit_epoch_end` in `src/detect/coco_early_stop.py` also stops when total validation loss (`box + cls + dfl`) improves by less than `LOSS_MIN_DELTA` (0.01) for `LOSS_PATIENCE` epochs (15), when mAP50-95 reaches `MAP_TARGET` (0.55, near the published YOLO11x number of about 0.547; set `None` to disable; 0.50 would stop a pretrained fine-tune almost immediately), or when the 11-hour budget is used. The callback sets `trainer.stop` and caps `epochs` only after that epoch's validation and checkpoint write, so `best.pt` and `last.pt` are kept.

On two GPUs the callback decides on rank 0. Ultralytics 8.4 broadcasts `trainer.stop` after `on_fit_epoch_end` and pickles callbacks into the DDP workers. If that broadcast or pickle fails on a newer build, use one GPU (`device=0`) or patience only.
