# Smart Parking – Hệ thống bãi đỗ xe thông minh

Đồ án môn học: phát hiện ô đỗ trống/có xe từ camera cố định bằng Computer Vision & Deep Learning,
hiển thị realtime trên dashboard web.

**Phương pháp đề xuất:** cắt từng ô đỗ (vị trí vẽ một lần cho mỗi camera) → phân loại trống/có xe bằng
**MobileNetV3-Small** (học chuyển giao từ ImageNet) → làm mượt theo thời gian → backend + dashboard.
**Phương pháp so sánh:** HOG/LBP + SVM, CNN nhỏ train từ đầu, ResNet18, YOLOv8n và YOLO11n (train trên PKLot / pretrained COCO).

## Kết quả chính
| Phương pháp | PKLot test | PKLot bỏ 1 camera | CNR-EXT (bãi khác) | Video demo | FPS (CPU, 100 ô) |
|---|---|---|---|---|---|
| HOG + SVM | 98.02% | 71.88% | – | – | – |
| LBP + SVM | 99.47% | 83.91% | – | – | – |
| CNN nhỏ | 99.90% | 98.39% | 82.91% | 99.75% | 5 |
| **MobileNetV3-Small** | **99.92%** | **99.51%** | 94.23% | 99.16% | **11** |
| ResNet18 | 99.95% | – | 94.76% | 99.54% | 3 |
| YOLOv8n-PKLot | 96.20% | – | 45.45% | 99.60% | 24 |
| YOLO11n-PKLot | 97.62% | – | 46.11% | 99.87% | 27 |
| YOLOv8n-COCO 1280 | 82.24% | – | 92.04% | 49.66% | 6 |
| YOLO11n-COCO 1280 | 73.77% | – | 95.70% | 36.20% | 6 |

Fine-tune MobileNetV3 bằng **5 khung hình có nhãn mỗi camera** của CNR-EXT: 96.27% → **98.58%**
(đo trên 8 ngày test không trùng ngày fine-tune). Bảng đầy đủ: `results/summary_all.csv`.

Kết luận: YOLO train trên PKLot mạnh nhất trên camera quen nhưng sụp đổ ở bãi mới; YOLO pretrained COCO tổng quát
khi xe đủ lớn nhưng bỏ sót xe nhỏ; MobileNetV3 ổn định nhất và chỉ cần vài khung hình có nhãn để thích nghi bãi mới.

## Kiến trúc
```
[Video/Camera] → [CV Worker: cắt ô → MobileNetV3 (ONNX) → làm mượt 4/5 frame]
                         │ HTTP POST (chỉ khi ô đổi trạng thái) + ảnh overlay
                         ▼
                 [FastAPI + SQLite] ──WebSocket──► [Dashboard Web]
```

## Cấu trúc thư mục
| Thư mục | Nội dung |
|---|---|
| `notebooks/` | Notebook chạy trên Kaggle: tải dữ liệu, EDA, cắt crop, train phân loại, train YOLO |
| `src/vision/` | Cắt ô (`slot_cropper`), phân loại ONNX (`classifier`), YOLO ONNX (`yolo_detector`), làm mượt (`smoothing`), vòng lặp chính (`worker`), tool vẽ ô (`slot_annotator`), tự tìm ô (`slot_proposer`) |
| `src/backend/` | FastAPI + SQLite + WebSocket, dashboard và trang tải video HTML/JS thuần (chạy được không cần Internet) |
| `src/evaluation/` | Đánh giá cả hệ thống từ log worker + nhãn thật |
| `tools/` | Tạo video demo PKLot, đánh giá CNR-EXT, fine-tune cho bãi mới, gom bảng kết quả |
| `configs/` | File JSON polygon ô đỗ cho từng camera |
| `models/` | File ONNX + JSON tiền xử lý; `checkpoints/` chứa trọng số `.pt` (không đưa lên Git) |
| `results/` | Bảng số liệu và hình cho báo cáo (`classifier/`, `yolo/`, `yolo11/`, `cnrpark/`, `summary_all.csv`) |
| `data/` | Dữ liệu tải về và sinh ra trên máy (không đưa lên Git) |
| `tests/` | Unit test (pytest) |

## Cài đặt
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Chạy demo (mọi lệnh chạy trong thư mục `smart-parking/`)
```bash
# 1. Tạo video demo từ 48 ảnh PKLot thật (PUCPR, sáng 2012-09-20) + cấu hình 100 ô + nhãn thật
python tools/make_pklot_demo.py

# 2. Bật backend + dashboard → mở http://localhost:8000
uvicorn src.backend.main:app --port 8000

# 3. (cửa sổ khác) Chạy worker theo thời gian thực, đẩy dữ liệu lên dashboard
python -m src.vision.worker --video data/pklot_demo/video.mp4 --config configs/pklot_pucpr.json ^
    --model models/classifier.onnx --api http://localhost:8000 --log-csv data/pklot_demo/worker_log.csv

# 4. Đánh giá so với nhãn thật
python -m src.evaluation.evaluate --log data/pklot_demo/worker_log.csv --gt data/pklot_demo/ground_truth.csv
```
Tùy chọn worker: `--show` xem cửa sổ video, `--loop` phát lặp, `--no-pace` xử lý nhanh nhất có thể,
`--method yolo --model models/yolo/<tên>.onnx --cover-thr <ngưỡng trong yolo.json>` để chạy pipeline YOLO.
API docs: `http://localhost:8000/docs`.

## Triển khai public (Hugging Face Spaces)
`Dockerfile` chạy backend + dashboard ở cổng 7860, tự bật hai camera demo (PKLot, CNR-EXT camera 8) và giới hạn
chức năng tải video (≤ 50 MB, ≤ 2 phút, tối đa 2 video chạy cùng lúc, tự dừng sau 15 phút, giữ 5 video gần nhất).
Mỗi lần khởi động sinh một mã ngẫu nhiên cho worker; chỉ worker có mã mới gửi được dữ liệu lên dashboard.

Mỗi lần đẩy lên nhánh `main`, GitHub Action `.github/workflows/deploy-huggingface.yml` chạy test rồi cập nhật Space:
1. Tạo Space trên Hugging Face: *New Space* → SDK **Docker** → *Blank*.
2. Tạo token quyền **Write** (Settings → Access Tokens).
3. Trong repo GitHub: *Settings → Secrets and variables → Actions* → secret `HF_TOKEN` = token trên,
   variable `HF_SPACE` = `ten-tai-khoan/ten-space`.

Chạy thử image trên máy: `docker build -t smart-parking . && docker run -p 7860:7860 smart-parking` → http://localhost:7860.
Các giới hạn đặt bằng biến môi trường `SMART_PARKING_*` trong `Dockerfile`; khi chạy bằng `uvicorn` trên máy thì không giới hạn.

## Tải video lên trên web (tự tìm ô đỗ)
Mở `http://localhost:8000/upload` (nút **Tải video lên** trên dashboard):
1. Chọn video bãi xe (camera cố định, nhìn từ trên cao) → **Tải lên**.
2. **Tự tìm ô đỗ**: quét 24 khung hình trải đều video bằng YOLO-COCO (xe đứng yên ở nhiều khung → một ô) và
   YOLO-PKLot (ô đỗ kiểu PKLot); mất khoảng 20–40 giây. Sửa lại nếu cần: bấm vào ô rồi Delete để xoá, kéo chuột để vẽ thêm.
3. **Bắt đầu theo dõi** → hệ thống phát lặp video theo thời gian thực, ô trống viền xanh, có xe viền đỏ, mở dashboard.

Thử nghiệm tự tìm ô: video PKLot tìm thấy 100% trong 100 ô có nhãn; video CNR-EXT camera 8 (bãi chưa từng thấy) 91% trong 54 ô.
Ô không bao giờ có xe trong video và không giống ô PKLot sẽ bị thiếu – vẽ thêm bằng tay ở bước 2.

## Dùng cho camera / bãi xe mới (dòng lệnh)
```bash
# Vẽ polygon ô đỗ (chuột trái: thêm góc, u: xoá, s: lưu, q: lưu và thoát)
python -m src.vision.slot_annotator --video data/my_video.mp4 --out configs/camera_01.json --camera-id cam01

# Chạy worker với cấu hình vừa vẽ
python -m src.vision.worker --video data/my_video.mp4 --config configs/camera_01.json --model models/classifier.onnx --show
```
Nếu độ chính xác ở bãi mới chưa tốt: gán nhãn vài khung hình mỗi camera rồi fine-tune (xem `tools/finetune_cnrpark.py`).

## Đánh giá và thí nghiệm trên máy
| Script | Việc làm |
|---|---|
| `tools/make_pklot_demo.py` | Ghép ảnh PKLot liên tiếp thành video time-lapse + cấu hình ô + nhãn thật từng frame |
| `tools/eval_cnrpark.py` | Đánh giá mô hình (phân loại hoặc YOLO) trên CNR-EXT, tách theo camera/thời tiết |
| `tools/finetune_cnrpark.py` | Fine-tune MobileNetV3 bằng K khung hình/camera của CNR-EXT, đo trước/sau, xuất ONNX |
| `tools/make_summary.py` | Gom mọi kết quả thành `results/summary_all.csv` |
| `tools/make_cnr_demo.py` | Tạo video time-lapse "bãi xe mới" từ ảnh CNR-EXT + cấu hình ô + nhãn thật |
| `tools/export_onnx.py` | Xuất trọng số `.pt` (CNN nhỏ, MobileNetV3, ResNet18) sang ONNX + JSON tiền xử lý |

```bash
python tools/eval_cnrpark.py --model models/classifier.onnx
python tools/eval_cnrpark.py --method yolo --model models/yolo/yolo11n_coco_1280.onnx --cover-thr 0.5
python tools/finetune_cnrpark.py --weights models/checkpoints/mobilenet_v3.pt --shots 5 20 50
python tools/make_summary.py
python -m pytest
```
Dữ liệu CNR-EXT (đặt trong `data/cnrpark/`): `CNRPark+EXT.csv`, `CNR-EXT_FULL_IMAGE_1000x750.tar`,
`CNR-EXT-Patches-150x150.zip` – tải từ http://cnrpark.it (các link trỏ tới GitHub Releases của repo
`fabiocarrara/deep-parking`). Xuất ONNX sau fine-tune cần gói `onnx`.

## Dataset
- **[dronefreak/PKLot](https://huggingface.co/datasets/dronefreak/PKLot)** (CC BY 4.0): 12.416 ảnh 1280×720, 3 camera
  (PUCPR, UFPR04, UFPR05), 693.755 bbox ô đỗ (`0 = vacant`, `1 = occupied`), train/valid/test chia sẵn theo (camera, ngày).
  P. R. L. de Almeida et al., *PKLot – A robust dataset for parking lot classification*,
  Expert Systems with Applications 42(11), 2015. DOI: 10.1016/j.eswa.2015.02.009
- **CNRPark+EXT** (ODbL v1.0, http://cnrpark.it): phần CNR-EXT gồm 144.965 ô có nhãn, 9 camera, nắng/mây/mưa,
  ảnh toàn cảnh 1000×750 + toạ độ ô. G. Amato et al., *Deep learning for decentralized parking lot occupancy detection*,
  Expert Systems with Applications 72, 2017.

Video demo trong `data/pklot_demo/` và `data/cnr_demo/` được ghép từ ảnh của hai bộ dữ liệu trên, giữ nguyên giấy phép
và ghi nguồn như trên. Mô hình YOLO huấn luyện/xuất bằng Ultralytics theo giấy phép AGPL-3.0.

## Quy trình trên Kaggle
| Notebook | Input | GPU / Internet | Kết quả |
|---|---|---|---|
| `00_download_pklot` | – (secret `HF_TOKEN`) | Không / Có | PKLot 12.416 ảnh + label YOLO |
| `01_eda_pklot` | output của 00 | Không / Không | Biểu đồ phân bố dữ liệu |
| `02_make_crops` | output của 00 | Không / Không | 460.652 crop 160×160 (Kaggle tự nén thành `_output_.zip`) |
| `03_train_classifier` | output của 02 | T4 / Có | `classifier.onnx`, `results.csv`, hình, `*.pt` |
| `04_train_yolo` | output của 00 | T4 / Có | YOLOv8n: `onnx/*.onnx`, `results_yolo.csv` |
| `04b_train_yolo11` | output của 00 | T4 / Có | YOLO11n: như trên |

Ghi chú:
- Có thể gắn thẳng output của một notebook làm input cho notebook khác (*Add Input → Your Work*), không cần tạo Dataset.
- Tải ẩn danh từ Hugging Face trên Kaggle hay bị **HTTP 429** → thêm token Read vào *Add-ons → Secrets* tên `HF_TOKEN`.
- Notebook 03/04 lưu kết quả từng phần; nếu bị dừng giữa chừng, gắn output lần chạy trước làm input để không train lại.
- Tải output: tab *Output* → **⋮ → Download all**. File `.pt` tải lẻ có thể bị đổi đuôi thành `.zip` – đổi lại `.pt`, không giải nén.
