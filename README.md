# 用手機照片辨識日常物品

這個專案用遷移學習訓練一個影像分類模型。你拍好幾類日常物品，用資料夾名稱當標籤，然後執行一條指令就能開始訓練。

目標是在「沒有拿來訓練、也沒有拿來挑選模型」的測試照片上達到 99% 以上。程式會自己把照片分成訓練、驗證、測試，並檢查有沒有幾乎重複的照片漏到不同組。

99% 來自照片拍得夠多樣，不是把模型調到會背答案。訓練集上看起來很高的正確率不算數。

## 這個程式在做什麼

模型先在 ImageNet（一百多萬張、一千類的公開照片）上學過邊緣、顏色和形狀。我們留下那些能力，把最後一層換成你的類別：

1. 第一階段只訓練新的最後一層，原本的層先固定。
2. 第二階段用較小的學習率，微調整個網路。

有 NVIDIA 顯示卡會用 CUDA，Apple 晶片會用 MPS，否則用 CPU。不用自己改程式。

驗證集用來決定要留哪一輪的模型。測試集留到最後才看一次，那個數字才是成績。

## 安裝

需要 Python 3.10 或更新的版本。

```bash
python -m venv .venv
```

macOS / Linux：

```bash
source .venv/bin/activate
```

Windows：

```bash
.venv\Scripts\activate
```

Mac、Windows，或是已經有 NVIDIA 顯示卡、想用官方預設安裝方式：

```bash
pip install -r requirements.txt
```

Linux 而且沒有 NVIDIA 顯示卡時，先裝 CPU 版的 PyTorch，避免抓到很大的 CUDA 套件：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

有 NVIDIA 顯示卡時，先到 [pytorch.org](https://pytorch.org) 選對應 CUDA 版本的安裝指令，裝好 `torch` 和 `torchvision`，再執行 `pip install -r requirements.txt`。

每次開新的終端機都要再啟動一次 `.venv`。第一次訓練會下載預訓練權重（ConvNeXt-Tiny 大約 100MB），需要網路。下載完會留在本機。

iPhone 若是 HEIC，可以在照片 App 輸出成 JPEG，或另外安裝：

```bash
pip install pillow-heif
```

直著拍沒關係。程式會讀 EXIF 方向，把照片轉成你在相機裡看到的方向。

## 怎麼拍照、怎麼標註

想靠近 99%，先看這一段。類別大約 5 到 20 個日常物品，例如杯子、鑰匙、水瓶、鞋子、書、遙控器。

每一類要多少張：

| 情況 | 每一類大約幾張 |
| --- | --- |
| 類別長得差很多 | 80 到 120 張通常就夠 |
| 類別有一點像 | 150 到 200 張，而且要拍到能分辨的細節 |
| 少於 40 張 | 可以先跑通流程，不要期待 99% |

每一類至少 50 張才值得認真看測試分數。照片少的時候，測試集可能只有十幾張，100% 只代表那十幾張碰巧都對。

同一類裡要換著拍，不要用連拍湊數量：

- 至少 4 種角度：正面、側面、斜上方、再低一點
- 至少 3 種光線：窗邊、室內日光燈、晚上的燈
- 至少 3 種背景：桌子、地板、手裡拿著、戶外都可以
- 有的近、有的遠，物件不要永遠放在正中間
- 偶爾讓手指遮到一點點
- 可以分兩天拍。衣服、桌面、天氣變了，模型才不會背背景

標註方式：

- 一張照片只放一個主要物件
- 資料夾名稱就是答案。用英文短名、不要空格：`water_bottle`，不要 `water bottle`
- `Mug` 和 `mug` 在 Linux 上會被當成兩個類別。名稱請固定一種寫法
- 不確定的照片不要放進去
- 同一張照片不要複製到兩個資料夾，也不要複製到 train 和 test

兩件物品如果長得很像（兩支白水瓶、兩雙白鞋），模型會把它們搞混。要嘛拍到固定看得到的差異（標籤、鞋帶、瓶蓋），要嘛把它們合成同一類。

拍完不要自己挑「最好看的」當考試照片。讓程式切分，或事先把 train、val、test 分好，之後就不要再把同一組連拍拆進不同資料夾。

## 資料夾長什麼樣子

最簡單：一個資料夾，裡面每個類別一個子資料夾。程式會用固定種子做 70% / 15% / 15% 的分層切分（每一類都按這個比例進訓練、驗證、測試）。

```text
photos/
  mug/
    001.jpg
    002.png
  keys/
    001.jpg
  bottle/
    001.jpg
```

也可以自己分好。有 `train` 資料夾，而且裡面是類別資料夾時，程式不會重切。`val` 也可以叫 `valid` 或 `validation`。

```text
photos/
  train/
    mug/
    keys/
  val/
    mug/
    keys/
  test/
    mug/
    keys/
```

只有 train 和 test、沒有 val 時，程式會從 train 挪出一部分當驗證，test 保持不動。只有 train 和 val、沒有 test 時，訓練還是會跑，但不會假裝有一個測試分數。

照片直接放在類別資料夾裡，不要再多包一層。支援 jpg、jpeg、png、webp、bmp。HEIC 需要上面的 `pillow-heif`。

也可以用 CSV，欄位是 `path,label`，路徑可以相對 `--data`：

```csv
path,label
mug/001.jpg,mug
keys/001.jpg,keys
```

```bash
python train.py --data photos --csv labels.csv
```

照片建議放在專案外面，或放在 `data/`。`data/` 和 `runs/` 不會被 git 上傳。

## 指令

真實照片：

```bash
python train.py --data /path/to/photos
python evaluate.py
python predict.py image.jpg
```

`evaluate.py` 和 `predict.py` 會讀 `config.yaml` 裡的 `output_dir`（預設 `runs/exp/best.pt`）。如果訓練時改過 `--output-dir`，這裡要加上 `--checkpoint` 指向那個 `best.pt`。

其他常用參數：

```bash
python train.py --data /path/to/photos --model mobilenet_v3_small
python predict.py image.jpg another.jpg
python predict.py some_folder/
python predict.py image.jpg --threshold 0.7
python export_onnx.py
```

`predict.py` 會印出前幾名和信心。最高機率低於門檻（預設 60%）時，答案是 `unknown`，不會硬猜一個類別。前幾名仍會印出來，方便你看它差一點點猜了什麼。

還沒有自己的照片時，可以先用合成的形狀圖把流程跑一遍。這個設定用很小的模型，不會下載預訓練權重，正確率也不代表你之後的照片：

```bash
python scripts/make_synthetic_data.py --out data/synthetic
python train.py --data data/synthetic --config config.smoke.yaml
python evaluate.py --config config.smoke.yaml
python predict.py data/synthetic/circle/000.jpg --config config.smoke.yaml
```

想試公開資料集的一小部分（CIFAR-10，32×32，需要網路，大約 170MB）：

```bash
python scripts/download_sample.py --out data/cifar_sample
python train.py --data data/cifar_sample
```

CIFAR 的圖片很小，不要用它來判斷能不能到 99%。

檢查程式本身沒有壞掉：

```bash
python -m pytest tests -q
```

## 怎麼看結果

訓練時每一輪會印一行。詞對照：

| 畫面上的字 | 意思 |
| --- | --- |
| epoch | 一輪，訓練照片全部看過一遍 |
| train_acc | 訓練正確率。模型有沒有記住練習照片 |
| val_acc | 驗證正確率。用來決定要留哪一輪。不是期末成績 |
| loss | 損失。我們用正確率挑選模型，損失只是參考 |
| lr | 學習率。第二階段會比第一階段小 |
| best.pt | 驗證正確率最高的那一輪 |

訓練結束後會看到：

```text
Held-out test accuracy: 96.7%  (58/60)
```

這一行才是成績。括號裡是答對張數 / 測試張數。

同一段還會列出每一類的 precision（猜成這一類的時候有多準）和 recall（這一類的照片抓到多少），以及 support（這一類在測試集有幾張）。support 很小的時候，百分比會跳很兇，要再多拍。

檔案都在 `runs/exp/`（或你設定的 `output_dir`）：

| 檔案 | 內容 |
| --- | --- |
| `best.pt` | 要留下來的模型。`evaluate.py` 和 `predict.py` 用的是這個，不是最後一輪 |
| `test_metrics.json` | 測試正確率、每一類的數字、混淆矩陣的數字 |
| `confusion_matrix.png` | 混淆矩陣。列是真正的類別，欄是模型猜的類別 |
| `misclassified.csv` | 猜錯的照片。信心高的排在前面，先打開那些 |
| `duplicates.csv` | 幾乎一樣、卻被分到不同組的照片。沒有這個問題就不會產生警告 |
| `split_manifest.csv` | 每一張照片被分到哪一組。評估時會沿用這份清單 |
| `history.json` | 每一輪的訓練和驗證數字 |

混淆矩陣可以想成這樣。對角線越大越好：

```text
              猜成 mug    猜成 bottle
真正是 mug         28            2
真正是 bottle       1           29
```

mug 那一列有 2 張被看成 bottle，就到 `misclassified.csv` 把那兩張找出來。常常會發現背景太像、角度太少，或是兩類本來就很像。補拍那些情況，再訓練一次。

其他訊號：

- 訓練正確率接近 100%，測試正確率差一截：照片太像、太少，或是背景被背起來了。看重複照片警告，並補不同的背景和光線。
- 出現 near-duplicate 警告：連拍或複製的圖進了不同組，測試分數會灌水。刪掉重複的那張，再訓練。
- 你為了把測試分數變高而一直改設定、一直看測試：那個分數會慢慢失去意義。調學習率和訓練輪數時看驗證正確率，測試留到一段調整結束再看。
- 不要改 `seed` 來碰一個比較好看的測試分數。換種子等於換了一份考卷。
- `predict.py` 常常回答 `unknown`：門檻太高，或是這張照片和訓練時差很多。可以把 `predict.unknown_threshold` 降到 0.45 看看，也可以看印出來的百分比。
- 幾乎全部猜成同一類：先確認每個資料夾裡都有照片，而且那些檔案打得開。再把 `train.full_epochs` 加長。CPU 很慢時把 `model.name` 改成 `mobilenet_v3_small`。

`test_metrics.json` 裡的 `accuracy` 是 0 到 1 的小數。0.99 就是 99%。

## 設定檔

常用的開關在 `config.yaml`，檔案裡有中文註解。

| 設定 | 作用 |
| --- | --- |
| `model.name` | `convnext_tiny`（預設）、`efficientnet_v2_s`、`vit_b_16`、`mobilenet_v3_small` |
| `train.head_epochs` / `train.full_epochs` | 兩階段各最多練幾輪。驗證分數還在往上、卻已經停了，就把 `full_epochs` 加大 |
| `train.batch_size` | 顯示卡記憶體不夠就改成 8 或 4 |
| `train.head_lr` / `train.full_lr` | 第一階段、第二階段的學習率 |
| `predict.unknown_threshold` | 低於這個信心就回答 unknown |
| `eval.tta` | 測試時多看一次水平翻轉再平均。挑選模型時不會用 |
| `image_size` | 預設 224。ViT 請維持 224 |
| `seed` | 切分和訓練的隨機種子 |
| `duplicates.max_distance` | 越小越嚴格。警告把明顯不同的照片也算進去時，改成 3 |

命令列可以暫時蓋過一個值，不必改檔案：

```bash
python train.py --data /path/to/photos --set train.full_epochs=25
```

`config.smoke.yaml` 只給合成圖片和自動測試。不要拿它訓練真實照片。

## 訓練很慢或出錯

CPU 上每一輪可能要幾分鐘，有顯示卡會快很多。畫面有在印 epoch，就代表還在跑。

- `CUDA out of memory`：把 `train.batch_size` 調小。
- Mac 出現 MPS 錯誤：把 `device` 改成 `cpu`。
- 讀不到 HEIC：安裝 `pillow-heif`，或改存 JPEG。
- 程式說某一類少於 3 張：自動切分時每一類至少要 3 張，訓練、驗證、測試才都有照片。
- 混淆矩陣的中文資料夾名變成空心方塊：系統沒有中文字型。資料夾用英文名最省事。

## 輸出成 ONNX

平常辨識一張照片用 `predict.py` 就好。之後若要接到別的程式：

```bash
python export_onnx.py
```

會在模型旁邊寫出 `model.onnx` 和 `model.labels.json`。輸出是機率，順序跟 labels 檔案一樣，加總是 1。輸入是已經縮放、並用 ImageNet 平均值與標準差正規化過的張量，形狀是 `[batch, 3, 224, 224]`（邊長跟訓練時的 `image_size` 相同）。

## 程式怎麼排

給想讀程式的人。註解是英文短句，說明為什麼要有這一步。

| 檔案 | 內容 |
| --- | --- |
| `train.py` `evaluate.py` `predict.py` | 你執行的指令 |
| `src/data.py` | 找照片、切分、EXIF |
| `src/duplicates.py` | 近重複照片檢查 |
| `src/transforms.py` | 訓練時的隨機增強，以及測試時的固定裁切 |
| `src/model.py` | 預訓練骨幹，換成你的類別數 |
| `src/engine.py` | 兩階段訓練。測試集在選完模型之後才評分 |
| `src/reporting.py` | 正確率、混淆矩陣、猜錯清單 |
| `tests/test_smoke.py` | 用合成圖片把整條流程跑完 |

GitHub Actions（`.github/workflows/smoke.yml`）會在 CPU 上跑 `pytest`，不需要你的照片。

## English

Train a classifier on phone photos of everyday objects.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --data /path/to/photos
python evaluate.py
python predict.py image.jpg
```

Put images in `photos/<class_name>/*.jpg`. If `train/`, `val/`, and `test/` already exist, those folders are used as-is. Otherwise the tool makes a stratified 70/15/15 split with a fixed seed. iPhone EXIF orientation is applied on load. HEIC needs `pip install pillow-heif` or a JPEG export.

The number that matters is `Held-out test accuracy`. Those images were not used for training and were not used to pick the checkpoint. A near-duplicate warning means almost the same photo leaked across splits; delete the copies and train again. `predict.py` prints top-k probabilities and says `unknown` when confidence is below `predict.unknown_threshold` (default 0.60).

For about 99% on 5–20 visually distinct classes, aim for 80–150 varied photos per class (up to 200 when classes look alike): different angles, distances, lighting, and backgrounds. Burst shots of the same frame do not count as new information. Console text is English; the sections above explain each term in Traditional Chinese.

Try the pipeline before your photos are ready:

```bash
python scripts/make_synthetic_data.py --out data/synthetic
python train.py --data data/synthetic --config config.smoke.yaml
python -m pytest tests -q
```

`config.smoke.yaml` is only for that synthetic run. Real photos use `config.yaml` and a pretrained backbone (`convnext_tiny` by default). The device is chosen automatically: CUDA, then Apple MPS, then CPU. `python export_onnx.py` writes probabilities plus a label list.
