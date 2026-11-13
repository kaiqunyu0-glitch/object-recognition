# 用手機照片找出畫面裡的每一個物品

這個專案的主要功能是**偵測**：一張照片裡可以同時有很多不同的東西，模型要把它們都找出來，每個物件一個邊界框、一個類別、一個信心。

它用 Ultralytics YOLO（預設 YOLO11 nano，已在 COCO 上預訓練）在你自己的標註上微調。有 NVIDIA 顯示卡用 CUDA，Apple 晶片用 MPS，否則用 CPU。

同一份程式裡也留著「一張照片只回答一個類別」的分類流程。那個流程在最後一節。你現在要做的是偵測，請從下面的標註步驟開始。

## 偵測和分類差在哪

分類：整張照片只准有一個答案，例如「這張是杯子」。照片裡如果同時有杯子和鑰匙，分類模型沒有地方把兩個都說出來。

偵測：模型可以在同一張照片裡畫很多框。每一個框有三樣東西：

- 類別：這是什麼
- 邊界框：它在畫面的哪裡
- 信心：模型有多確定

所以標註方式也不同。分類是把照片放進以類別命名的資料夾。偵測是在照片上把每一個物件圈起來。

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

Mac、Windows，或是已經有 NVIDIA 顯示卡：

```bash
pip install -r requirements.txt
```

Linux 而且沒有 NVIDIA 顯示卡時，先裝 CPU 版的 PyTorch，避免抓到很大的 CUDA 套件：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

有 NVIDIA 顯示卡時，先到 [pytorch.org](https://pytorch.org) 選對應 CUDA 版本，裝好 `torch` 和 `torchvision`，再執行 `pip install -r requirements.txt`。

每次開新的終端機都要再啟動一次 `.venv`。第一次用 `yolo11n.pt` 訓練會下載 COCO 預訓練權重（nano 大約幾 MB 到幾十 MB），需要網路。

iPhone 的 HEIC 可以在照片 App 輸出成 JPEG，或另外安裝：

```bash
pip install pillow-heif
```

直著拍沒關係。匯出成 YOLO 之後，框是你在標註工具裡看到的方向。程式會把 EXIF 方向烙進像素，框的相對位置保持不變。

## 怎麼標註

建議用 Roboflow 的免費方案。它會帶你畫框，並匯出這個程式能直接讀的 YOLO 格式。Label Studio 是完全跑在你電腦上的替代方案，步驟在後面。

畫框時請固定一套規則，整份資料都用同一套：

- 框貼著物件外緣，不要留一大圈背景，也不要切掉一半
- 畫面裡每一個你想辨認的物件都要框，漏標會教模型「那裡不該有框」
- 被遮住時，從頭到尾只選一種：框整個物件，或只框看得到的部分
- 類別名稱用英文短名、不要空格：`water_bottle`，不要 `water bottle`
- 不要用連拍湊數量。幾乎一樣的照片若被分到訓練和測試，分數會灌水

### Roboflow（建議，逐步）

1. 打開 [https://roboflow.com](https://roboflow.com)，註冊免費帳號。
2. 建立專案。類型選 **Object Detection**。名稱用英文，例如 `desk-objects`。
3. 上傳照片（jpg 或 png）。
4. 進入標註。每個物件拖一個框，輸入或選擇類別，存檔。一張照片可以有很多框，也可以有很多類別。
5. 把標好的照片加進 Dataset。切分建議設成 train 70%、valid 15%、test 15%。Roboflow 的 valid 就是這個程式的 val。
6. 產生一個版本後，Export。格式選 **YOLO v8** 或 **YOLOv5 PyTorch txt**。這兩種都是 `class cx cy w h`。下載 zip。
7. 解壓縮。裡面會有 `data.yaml`，以及 `train/`、`valid/`、`test/`。
8. 訓練：

```bash
python detect_train.py --data /path/to/unzipped-folder
```

程式看到現成的 train / valid / test 就不會重切。

### Label Studio（免費、資料留在自己的電腦）

1. 安裝並打開：

```bash
pip install label-studio
label-studio
```

2. 建立專案。Labeling setup 選 **Object Detection with Bounding Boxes**。把標籤改成你的類別。
3. Import 你的照片，逐張畫框。
4. Export 選 **JSON**。若匯出選項裡直接有 YOLO，用 YOLO 也可以，然後把解壓後的資料夾交給 `detect_train.py`。
5. JSON 匯出不含照片檔。請把 `label_studio.json` 和照片放在一起，或把照片放在同層的 `images/` 裡。程式會用檔名對上框。
6. 訓練：

```bash
python detect_train.py --data /path/to/that-folder
```

只有一個池子的照片時，程式會用固定種子做 70% / 15% / 15% 的切分，並盡量讓比較少見的類別也出現在驗證集和測試集。

CVAT 也可以。匯出 **YOLO 1.1**，把解壓後的資料夾路徑傳給 `--data`。

## 要多少照片、多少框

偵測比分類吃資料。分類只要整張照片一個答案；偵測要在各種背景裡把每一個物件的位置也學對。

| 目標 | 大約需要 |
| --- | --- |
| 先把流程跑通 | 每類 30 到 50 張照片 |
| 認真做一個小場景（5 到 15 類、物件長得差很多） | 每類 150 到 300 張以上「有這個物件出現」的照片，而且每類幾百個框 |
| 兩類長得很像，或物件常常很小、常常被擋住 | 再往上加，而且要刻意拍那些難的情況 |

一張照片裡有 3 個杯子，算 3 個杯子實例，但背景仍是同一張。實例數要夠，場景也要夠。

每一類都要有這些變化，否則模型只記得你的桌子：

- 不同背景、不同房間、室內和窗邊
- 白天、日光燈、晚上
- 物件有時近、有時遠，有時貼在畫面邊緣
- 有時被手或其他東西遮住一部分
- 有時兩個物件重疊或靠在一起
- 不要只有一種角度

框的品質和數量一樣重要。框歪了、漏了一個、類別點錯了，模型都會學進去。

## 分數怎麼讀才誠實

偵測沒有一個叫 accuracy 的總分。不要把分類那種「99%」直接拿來當目標。

| 數字 | 它在問什麼 | 這個專案裡，怎樣算高 |
| --- | --- | --- |
| mAP50 | 框有沒有蓋到對的物件（重疊 IoU 至少 0.5） | 類別清楚、標註乾淨時，測試集 0.85 到 0.95 算很強 |
| mAP50-95 | 框還要貼得緊，從寬鬆到嚴格平均 | 真實手機照片 0.60 到 0.80 已經很好。0.90 以上很少見 |
| precision 精確率 | 畫出來的框裡，有多少真的是那個物件 | 高代表少亂框 |
| recall 召回率 | 真的在場的物件裡，有多少被找到 | 高代表少漏 |

精確率高、召回率低：模型很保守，沒把握就不畫。召回率高、精確率低：幾乎都找得到，但多了很多錯框。兩個都要看。

驗證集的 mAP 用來決定要留哪一輪的 `best.pt`。測試集的 mAP 才是成績。那些測試照片沒有拿來訓練，也沒有拿來選模型。

`errors.csv` 裡的「漏掉」和「多框」用的是信心門檻（預設 0.25），也就是 `detect_predict.py` 實際會畫出來的框。mAP 會把各種門檻都算過，所以兩個數字不必相等。

若出現 near-duplicate 警告，代表幾乎一樣的照片被分到不同組。刪掉重複的那張再訓練，否則測試分數會看起來比真正的能力高。

不要為了把測試 mAP 調高而一直改設定、一直看測試。調輪數和模型大小時看驗證 mAP，測試留到一段調整結束再看。也不要改 `seed` 來換一份比較好看的考卷。

## 資料長什麼樣子

YOLO 格式是這個程式的原生格式。`data.yaml` 加上照片和標籤：

```text
dataset/
  data.yaml
  images/train/001.jpg
  images/val/002.jpg
  images/test/003.jpg
  labels/train/001.txt
  labels/val/002.txt
  labels/test/003.txt
```

`001.txt` 裡每一行是一個物件，數字都介於 0 和 1，原點在框的中心：

```text
0 0.512000 0.430000 0.180000 0.220000
1 0.210000 0.760000 0.090000 0.110000
```

第一個數字是類別編號，後面是中心 x、中心 y、寬、高，都除以照片的寬和高。

只有 `images/` 和 `labels/`、還沒切分也可以。程式會自己切。COCO JSON、Label Studio JSON 會先轉成上面的格式再訓練。原始照片不會被改寫；轉好的複本在 `runs/detect/dataset/`。

## 指令

```bash
python detect_train.py --data /path/to/dataset
python detect_eval.py
python detect_predict.py image.jpg
```

`detect_eval.py` 和 `detect_predict.py` 讀 `config.detect.yaml` 的 `output_dir`（預設 `runs/detect`）。訓練時如果改過 `--output-dir`，這裡也要加上同一個 `--output-dir`。

其他用法：

```bash
python detect_predict.py photo_folder/
python detect_predict.py video.mp4
python detect_predict.py 0
python detect_train.py --data /path/to/dataset --model yolo11s.pt
```

`0` 是預設攝影機。影片或攝影機按 Ctrl+C 停止。每張照片會印出物件、信心和框，並把畫好框的圖存到 `runs/detect/predictions/`。

還沒有自己的標註時，可以用合成的形狀圖把流程跑一遍。這個設定用很小的模型，不下載權重，mAP 也不代表你之後的照片：

```bash
python scripts/make_synthetic_detection.py --out data/detect_synthetic
python detect_train.py --data data/detect_synthetic --config config.detect.smoke.yaml
python detect_eval.py --config config.detect.smoke.yaml
```

檢查程式本身：

```bash
python -m pytest tests -q
```

GitHub Actions 會在 CPU 上跑同一套測試，不需要你的照片。

## 訓練時會看到什麼

| 畫面上的字 | 意思 |
| --- | --- |
| epoch | 一輪，訓練照片全部看過一遍 |
| mAP50 / mAP50-95 | 驗證集上的偵測分數，用來決定要留哪一輪 |
| best.pt | 驗證分數最高的模型。評估和預測都用這個 |
| Near-duplicate check | 有沒有幾乎一樣的照片跨組 |

訓練結束後執行 `detect_eval.py`，成績在這幾行：

```text
mAP50:     0.910
mAP50-95:  0.720
precision: 0.880
recall:    0.850
```

檔案在 `runs/detect/`：

| 檔案 | 內容 |
| --- | --- |
| `train/weights/best.pt` | 要留下的模型 |
| `test_metrics.json` | 測試集的 mAP、精確率、召回率、每一類的數字 |
| `confusion_matrix.png` | 列是真正的類別，欄是模型猜的類別。最後一欄是漏掉，最後一列是多框 |
| `errors.csv` | 漏掉（miss）和多框（false_positive）的照片。先打開這些 |
| `duplicates.csv` | 幾乎一樣、卻在不同組的照片。沒有這個問題就不會警告 |
| `dataset/` | 轉正、切分之後的 YOLO 資料。評估沿用這份，不會重切 |

某一類的 recall 很低：多拍那種它找不到的角度和遮擋，並檢查是不是漏標。某一類的 precision 很低：看看 `errors.csv` 裡的 false_positive，常常是背景裡有長得很像的東西沒被標成別的類，或是框得太鬆。

CPU 上每一輪可能要幾分鐘。畫面有在走 epoch 就代表還在跑。記憶體不夠就把 `batch` 改成 8 或 4，或把 `imgsz` 改成 512。Mac 若出現 MPS 錯誤，把 `device` 改成 `cpu`。

## 設定檔

常用開關在 `config.detect.yaml`，檔案裡有中文註解。

| 設定 | 作用 |
| --- | --- |
| `model` | `yolo11n.pt`（預設，最快）、`yolo11s.pt`、`yolo11m.pt`，或 `yolov8n.pt` |
| `imgsz` | 輸入邊長，預設 640 |
| `epochs` / `patience` | 最多幾輪；驗證 mAP 多久沒進步就停 |
| `batch` | 一次看幾張。顯示卡記憶體不夠就調小 |
| `conf` | 預測以及漏掉/多框清單的信心門檻。不影響 mAP 的算法 |
| `match_iou` | 框和標準答案要重疊多少才算同一個物件 |
| `seed` | 切分和訓練的隨機種子 |

`config.detect.smoke.yaml` 只給合成圖片和自動測試。不要拿它訓練真實照片。

## 若一張照片只有一個物件：分類

如果每張照片只放一個物件、你只要一個類別名稱、不要框，再用分類。照片放成 `photos/<類別名>/*.jpg`，然後：

```bash
python train.py --data /path/to/photos
python evaluate.py
python predict.py image.jpg
```

分類的成績是測試正確率，不是 mAP。類別之間長得差很多、每類大約 80 到 150 張、角度和背景都有變過時，測試正確率才有機會到 99%。那個數字只算沒拿來訓練、也沒拿來選模型的照片。設定在 `config.yaml`。細節和偵測相同的部分：EXIF、近重複警告、CPU / CUDA / MPS，兩條流程都有。

分類不適合「一張照片很多東西」的題目。那種題目請用上面的偵測。

## English

The primary task is multi-object detection. One photo may contain many things; the model returns a box, a class, and a confidence for each.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python detect_train.py --data /path/to/dataset
python detect_eval.py
python detect_predict.py image.jpg
```

Label boxes in [Roboflow](https://roboflow.com) (Object Detection project, export YOLO v8 or YOLOv5 PyTorch) or in Label Studio (bounding-box template, export JSON next to the images). A folder that already has `data.yaml` plus `images/` and `labels/` works as-is. Existing train/val/test folders are kept. One unlabeled-by-split pool is split 70/15/15 with a fixed seed. iPhone EXIF orientation is baked into the pixels; normalized boxes stay as drawn on the upright view.

Detection is not one accuracy number. Read **mAP50** (found the object), **mAP50-95** (the box also fits tightly), **precision** (few extra boxes), and **recall** (few misses). On a small set of distinct objects with careful labels, test mAP50 around 0.85–0.95 is a strong result. Test mAP50-95 around 0.60–0.80 is already good on real phone photos. A plan that starts near that range is about 150–300+ images in which each class appears, and several hundred box instances per class, with different backgrounds, lighting, occlusion, and overlap. The test split is not used to train or to pick `best.pt`.

`detect_predict.py video.mp4` and `detect_predict.py 0` run on a video or the webcam. Single-label classification is still available with `train.py`, `evaluate.py`, and `predict.py` when each photo is only one object.
