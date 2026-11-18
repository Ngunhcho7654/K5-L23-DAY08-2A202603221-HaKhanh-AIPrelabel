# Báo cáo Day 08 — 2A202603221 - Hà Khánh

> Báo cáo tổng hợp quy trình AI-assisted labeling: thử nghiệm model, chọn ngưỡng, duyệt lô có ảnh bẫy, tạo mask SAM, calibration guideline v1.1 và QC mẫu. Mọi số liệu được trích trực tiếp từ `submission/qc/*`, `submission/model_log.csv` và `submission/time_log.csv`.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | `YOLO26n` (`yolo26n.pt`) | `0.3` (mặc định) | Chạy nhanh trên CPU (`0.40s–1.16s/ảnh`), nhận diện tốt 6 lớp COCO gần camera (`r01`, `r02`). Quét ngưỡng `threshold_sweep.py` trên 4 ảnh gold cho thấy `bicycle` (`0`), `motorbike` (`4`), `bus` (`0`), `truck` (`1`) có `< 5` vật gold nên không đặt ngưỡng riêng; `person` (max recall `67%`, 9 vật) và `car` (max recall `83%`, 6 vật) không đạt mục tiêu `90%` ở ngưỡng `<= 0.5`, nên giữ mặc định `0.3`. |
| three_wheeler | `YOLOE-26s-seg` (`yoloe-26s-seg.pt`, prompt: `tuk-tuk`, `cyclo`, `cargo tricycle`) | `0.3` (mặc định) | `YOLO26n` là detector đóng nên hoàn toàn không có lớp `three_wheeler`. `YOLOE` (`r03`) nhận diện được tuk-tuk (`img_002`, `img_011`) và xích lô (`img_032`), được ưu tiên thắng box `YOLO26` khi `IoU >= 0.6`. Do trên 4 ảnh gold chỉ có `6` vật `three_wheeler` và max recall ở ngưỡng `<= 0.5` là `83%` (`< 90%`), script đề xuất giữ ngưỡng mặc định `0.3`. |

- Lệnh `merge_prelabels.py` đã dùng:
  `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM bạn chọn** (`SAM 2.1 tiny` — `sam2.1_t.pt`) và lý do: Máy chạy CPU (`torch 2.15.0+cpu`, `16 GB RAM`, GPU GTX 960M 2GB VRAM không gánh bản lớn và chưa xin quyền gated `facebook/sam3`). `sam2.1_t.pt` (~78 MB) nhẹ hơn nhiều so với `sam2.1_b.pt` hay `SAM 3`, chạy ổn định trên CPU (`~6.56s/ảnh` khi ghép box -> mask ở `r04`), cho đường biên sắc nét trên vật tách biệt (`img_004.jpg` `person 0.91`) dù thỉnh thoảng bị lem khi box detector rộng hoặc khung xe có khe hở (`img_013.jpg`).
- Lỗi nào **không sửa được bằng đổi ngưỡng**? Sửa thế nào? Lỗi ngoài từ điển đóng của `YOLO26n`: ở `img_002.jpg` (`r01`) tuk-tuk xanh bị gọi nhầm thành `truck 0.37`, ở `img_011.jpg` mép tuk-tuk bị gọi thành `truck 0.28`, ở `img_032.jpg` xích lô chỉ bắt lẻ bánh sau thành `bicycle` hoặc bỏ sót hoàn toàn. Khi tăng ngưỡng lên `0.50` (`r02`), xe biến mất chứ không biến thành `three_wheeler`. Phải sửa bằng model open-vocabulary (`YOLOE` ở `r03`) rồi gộp ưu tiên (`IoU >= 0.6`) để thay thế box `truck`/`car` trùng lắp.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng**:
  - Điểm cao mà sai: `#97 truck` đạt `conf 0.66` nhưng thực chất là box ma nằm trên mặt đường trống (`img_014.jpg`); `#112 three_wheeler` đạt `conf 0.63` (`YOLOE`) nhưng là chiếc quang gánh hàng rong (`img_014.jpg`); `img_032.jpg` (`r01`) `YOLO26n` nhận nhầm hình vẽ trên tường thành `person`.
  - Điểm thấp mà đúng: `img_011.jpg` phần tuk-tuk ở rìa trái chỉ đạt `three_wheeler 0.28` (`r03`), `img_032.jpg` chiếc xích lô bên phải đạt `three_wheeler 0.27` (`r03`), `img_016.jpg` xe máy từ xa đạt `motorbike 0.41` (`r01`).
- Giới hạn: Chọn ngưỡng trên chính 4 ảnh gold (`26` đối tượng được chấm trên gold, `4` lớp có `< 5` vật nên phải giữ mặc định `0.3`) vừa có cỡ mẫu quá nhỏ vừa là đánh giá *in-sample* (chọn và chấm trên cùng dữ liệu), nên precision/recall của máy trên gold chỉ mang tính tham khảo lạc quan.

## 2. AI error profile (từ `submission/qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 308 | 308/321 = 96.0% | 321 box máy |
| edited | 3 | 3/321 = 0.9% | 321 box máy |
| deleted | 10 | 10/321 = 3.1% | 321 box máy |
| added | 16 | 16/327 = 4.9% | 327 đối tượng sau duyệt |

*(Ghi chú: 22 đối tượng nằm quanh 12 vị trí bẫy đã được loại khỏi bảng trên để đo chất lượng máy thuần; trong các box `deleted + added`, có `3` trường hợp là đổi lớp nên số vật máy sót thật xấp xỉ `16 - 3 = 13` vật).*

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? **`three_wheeler`** (`accepted=16`, `edited=2`, `deleted=6`, `added=4`) và **`motorbike`** (`added=5`) / **`bus`** (`added=2`, `accepted=2`). `YOLOE` hay bắt nhầm vật không phải xe như quang gánh (`img_014.jpg` #111, #112) hoặc xe bán tải (`img_002.jpg`), đồng thời sinh box chồng lấn hoặc hơi lỏng cần sửa/xoá; trong khi `YOLO26n` hay bỏ sót xe máy đậu sát nhau hoặc bị che (`img_006.jpg`, `img_035.jpg`).
- So với gold: Precision của máy là **`57.6%` (`19/33`, KTC 95%: `41–73%`)**, Recall của máy là **`73.1%` (`19/26`, KTC 95%: `54–86%`)** (`TP=19`, `FP=14`, `FN=7`; mẫu số: `26` đối tượng gold, `33` box máy trên `4` ảnh gold).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: **`4/12 = 33.3%`** (`KTC 95%: 14–61%`, thấp hơn ngưỡng đạt `80%`). Chi tiết theo loại: `extra: 2/3`, `missing: 0/3`, `wrong_class: 1/3`, `loose: 1/3`. Danh sách 8 bẫy bị lọt: `img_018.jpg (missing)`, `img_024.jpg (missing)`, `img_035.jpg (missing)`, `img_025.jpg (wrong_class)`, `img_028.jpg (wrong_class)`, `img_005.jpg (loose)`, `img_022.jpg (loose)`, `img_014.jpg (extra)`. Nguyên nhân trực tiếp là tôi đã bỏ qua **Bước 0 (ẩn hết nhãn máy để tự quét ảnh trước)** và duyệt quá nhanh, dẫn đến mắt chỉ nhìn quanh các box có sẵn nên bỏ lọt `100%` (`3/3`) bẫy `missing` (đặc biệt trên ảnh đông xe máy như `img_018`, `img_024` và xe máy trắng bị khuất một phần ở `img_035`), bị thuyết phục bởi box `conf` cao (`#97 truck 0.66` trên mặt đường ở `img_014`), và bỏ qua box lệch vừa (`loose` ở `img_005`, `img_022`).
- **`accepted` `96.0%` (`308/321`) nhưng bắt bẫy chỉ `33.3%` (`4/12`)** là minh chứng rõ rệt cho **thiên kiến tự động hóa (automation bias)** và **neo nhãn (anchoring)**: vì nhìn lướt thấy phần lớn box máy khá hợp lý, người duyệt rơi vào trạng thái bấm chấp nhận theo quán tính, chỉ xoá những box quá vô lý mà không thực hiện quét độc lập từng vật và từng cạnh box theo checklist 4 bước.

## 4. Máy có đáng dùng không?

- Thời gian (từ `submission/time_log.csv`): gán tay **`3.75 phút/ảnh`** (`15 phút / 4 ảnh gold`) vs duyệt có pre-label **`0.83 phút/ảnh`** (`20 phút / 24 ảnh batch`), nhanh hơn khoảng `4.5×` (tiết kiệm `~77.8%` thời gian mỗi ảnh). Thiên lệch: khi bước vào duyệt `batch`, tôi đã quen quy tắc và bối cảnh đường phố từ 4 ảnh gold trước đó; đồng thời tốc độ `0.83 phút/ảnh` có được phần nào do duyệt lướt, bỏ qua bước ẩn nhãn dẫn tới sót bẫy.
- Kết luận: **Có đáng dùng cho `person`, `motorbike`, `car`** vì giảm rất nhiều công vẽ tay trên ảnh đông xe, nhưng **không được duyệt nhanh kiểu tin tưởng mặc định** — đặc biệt với `three_wheeler` (`YOLOE` còn nhiễu và hay chồng box) và các vật nhỏ/bị che (`added = 16/327 = 4.9%`, `Recall` trên gold chỉ `73.1%`).
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. Bắt buộc bấm ẩn tất cả box máy (`con mắt` trong CVAT) ở **100% ảnh** và nhẩm đếm tổng số người/xe trong `10–15` giây trước khi bật lại box (kiểm tra bằng việc không lọt bẫy `missing`: mục tiêu `3/3`).
  2. Chia lô `24` ảnh thành `3` chặng mỗi chặng `8` ảnh, nghỉ bắt buộc `3–5 phút` giữa các chặng và giữ nhịp duyệt tối thiểu `>= 1.5 phút/ảnh` (kiểm tra được trực tiếp bằng dòng `break` và `minutes` trong `time_log.csv`).
  3. Quét kiểm tra riêng mọi box thuộc các lớp dễ nhầm (`three_wheeler`, `truck`, `bicycle`) bất kể `conf` cao hay thấp (kể cả `conf >= 0.6`), xác nhận có đủ bánh xe/khung xe và chỉ giữ `1` box cho mỗi vật trước khi bấm lưu (kiểm tra được qua tỷ lệ bắt bẫy đạt `>= 10/12 = 83.3%`).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| `img_013.jpg` | `SAM 2.1 tiny` (`sam2.1_t.pt`) | `4` (`2` dương, `2` âm) | Đã loại người lái (`neg 720,400`), nhưng mask vẫn lem qua khung cửa hở sang thực khách ngồi bàn ăn phía sau dù có `neg 460,340` | `1 polygon`, `41.0% ảnh` (`286,611 px`); rò rỉ mask qua khoảng trống khung xe hở | [tuktuk_img013.jpg](submission/masks/tuktuk_img013.jpg) |
| `img_004.jpg` | `SAM 2.1 tiny` (`sam2.1_t.pt`) | `4` (`2` dương, `2` âm) | Không gồm người đứng cạnh (`neg 465,300`); kính chắn gió liền khung nằm trong mask xe | `1 polygon`, `13.2% ảnh` (`92,294 px`); ranh giới tách tốt khỏi xe xanh bên cạnh và nền gạch | [tuktuk_img004.jpg](submission/masks/tuktuk_img004.jpg) |
| `img_032.jpg` | `SAM 2.1 tiny` (`sam2.1_t.pt`) | `4` (`2` dương, `2` âm) | Không gồm người đạp xích lô đứng bên phải (`neg 560,320`) | `1 polygon`, `7.8% ảnh` (`46,115 px`); bám sát đệm hoa và cả hai bánh trước | [cyclo_img032.jpg](submission/masks/cyclo_img032.jpg) |

- Đối chiếu guideline mục 4.2: Quy tắc 4.2.1 (bám đường biên phần nhìn thấy) và 4.2.4 (hai vật chạm nhau không chồng lấn) là căn cứ để đặt click âm tách người lái (`img_013`, `img_032`) và người đứng cạnh (`img_004`). Tuy nhiên, mục 4.2.2 (lỗ thủng `< 5%`) và 4.2.3 (kính xe) ban đầu còn lúng túng với phần khung sườn hở của xe tuk-tuk nhìn xuyên ra người phía sau (`img_013`), nên đã được chốt rõ trong `guideline_v1.1_calibrated.md`.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: Chưa dùng trực tiếp trên giao diện CVAT vì bản CVAT Docker mặc định chưa triển khai serverless Nuclio (AI Tools chưa có model gắn sẵn); dùng script `sam2_points.py` offline giúp tái lập chính xác toạ độ click dương/âm, nhưng chậm hơn về mặt tương tác so với click trực tiếp trên trình duyệt khi phải thử lại toạ độ (`img_004`, `img_032` ban đầu bị trượt toạ độ xuống nền đường trước khi căn lại).

## 6. Calibration (Part 6)

- Số quyết định: **`6`** (`CAL-01` đến `CAL-06` trong `submission/qc/calibration_decisions.csv`). Quan trọng nhất là `CAL-01` (chuẩn hoá gán `three_wheeler` thay vì `truck`/`car` của `YOLO26` ở `img_002`, `img_011`), `CAL-05` (loại quang gánh bán hàng rong bị `YOLOE` gán nhầm và xoá box trùng lặp ở `img_014`, `img_003`, `img_035`), và `CAL-06` (từ chối đặt ngưỡng `conf` tự chấp nhận box sau khi gặp box ma `conf 0.66`).
- Quy tắc mới đã vào [guideline_v1.1_calibrated.md](submission/guidelines/guideline_v1.1_calibrated.md): chốt toàn bộ `7` câu hỏi mở ở mục 6 của `v1.0` (xe đạp điện, mật độ `ignore_region` gần camera, hàng nhô ra `> 20%`, ngưỡng che `20%`, kính chắn gió & khoảng hở mask SAM, gộp box chồng lấn, chính sách `conf`) và bổ sung `5` ca khó kèm ảnh minh hoạ.
- Nguyên nhân gốc: Do **cả 3 yếu tố**: (1) *guideline v1.0 còn mơ hồ* ở các ca ranh giới (`CAL-02`, `CAL-03`, `CAL-04`), (2) *lỗi hệ thống của máy* với đặc thù Việt Nam/Thái Lan (`CAL-01`, `CAL-05`), và (3) *thao tác của người duyệt* bỏ bước quét độc lập (`CAL-06`).

## 7. QC và quyết định (Part 6)

- Mẫu: **`5/24` ảnh** (`0` ảnh `can_review` + `5` ảnh ngẫu nhiên với `seed = 42`: `img_003.jpg`, `img_006.jpg`, `img_014.jpg`, `img_031.jpg`, `img_035.jpg`), tổng cộng **`71` đối tượng**. Người kiểm: tự kiểm (`HaKhanh`) — **hạn chế**: tự chấm bài của chính mình dễ bị lặp lại cùng điểm mù nhận thức như lúc duyệt.
- Lỗi theo **đối tượng**: **`8/71 = 11.3%`** · theo **ảnh**: **`5/5 = 100.0%`** (vượt xa ngưỡng Release `<= 2%`).
- Top lỗi (từ `submission/qc/qc_review.csv` và `submission/qc/qc_report_auto.md`):
  1. `thua` (`5/8` lỗi): `img_014.jpg` (box ma `#97 truck` trên mặt đường trống và 2 quang gánh `#111`, `#112` gán thành `three_wheeler`), `img_003.jpg` (`#1` trùng với `#18` trên cùng xe ba gác), `img_035.jpg` (`#337` trùng với `#336` trên cùng xích lô).
  2. `bo_sot` (`2/8` lỗi): `img_006.jpg` (bỏ sót các xe máy đỗ sát cạnh `#50 motorbike` trong ngõ), `img_035.jpg` (bỏ sót xe máy trắng cạnh `#340 person` và người lái trên `#326 motorbike`).
  3. `sai_vi_tri_kich_thuoc` (`1/8` lỗi): `img_031.jpg` (`#292 person` cắt box ngang hông dù vẫn nhìn rõ phần chân bên dưới).
- **Giới hạn mẫu nhỏ**: Với `5/5` ảnh mẫu có lỗi, cận trên khoảng tin cậy 95% (Wilson) trong `qc_report_auto.md` là **`100%`**. Ngay cả trong trường hợp lý tưởng là `0/5` lỗi, cận trên 95% theo Wilson/quy tắc 3 vẫn lên tới ~`43%–60%`, nên cỡ mẫu 5 ảnh tuyệt đối không thể chứng minh được tiêu chuẩn `<= 2%` mà chỉ đủ để phát hiện lỗi hệ thống rõ ràng.
- Script kiểm được / không kiểm được: Các script (`validate_labels.py`, `review_stats.py`, `qc_report.py`) kiểm được tính hợp lệ JSON, box ngoài khung, vật `< 20 px` (cảnh báo 2 box `car` `14 px` và `17 px` ở `img_023.jpg`, `img_027.jpg`), độ lệch IoU giữa pre-label và final, và khớp 12 bẫy; **không** kiểm được box có ôm khít biên vật (`#292` cắt ngang hông), vật có đúng thực tế hay không (quang gánh `#111/#112`), hay vật bị cả máy và người bỏ sót (`img_006.jpg`).

**Quyết định: `Rework`, ngày `2026-11-18`.** Lý do (đi theo cây `Hold -> Rework -> Release`): Không `Hold` vì các câu hỏi mở của guideline đã được chốt sang `v1.1` và không có ảnh hỏng dữ liệu; buộc phải **`Rework`** vì thoả cả 3 điều kiện kích hoạt của `Rework`: tỷ lệ ảnh lỗi `5/5 = 100.0% > 2%`, có lỗi hệ thống lặp lại (`thua` chiếm `5/8` lỗi), và tỷ lệ bắt bẫy chỉ đạt `4/12 = 33.3% < 80%` khiến lô nhãn sau duyệt chưa đáng tin.
Việc cần làm tiếp cho vòng Rework: Dùng `guideline_v1.1_calibrated.md` cùng 3 quy tắc chống thiên kiến (bắt buộc ẩn nhãn máy quét từng ảnh), rà lại toàn bộ `24` ảnh `batch` trong CVAT (xoá box thừa/trùng ở `img_003`, `img_014`, `img_035`, bỏ box `< 20 px` đơn lẻ ở `img_023`, `img_027`, vẽ bù xe bỏ sót ở `img_006`, `img_035`, kéo đủ chân `#292`), export ra `submission/labels/qc_fixed.json`, chạy lại `build_release.py` và lấy mẫu QC vòng 2 (`qc_sample.py --force`).

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài**: Có dùng trợ lý AI (Antigravity) để hướng dẫn thao tác CVAT, giải đáp các điểm quy tắc trong guideline (ví dụ box `person` chồng lên `motorbike`), hỗ trợ chạy các lệnh Python có sẵn trong `scripts/`, sửa 2 bug nhỏ ở script thiết lập, kiểm tra cấu trúc file và rà soát đối chiếu số liệu từ file `viz/`, ảnh QC và `qc_report_auto.md`. Mọi kết quả nhãn trên CVAT (`gold_manual.json`, `batch_reviewed.json`) đều do tự tay tôi vẽ/duyệt và không sửa tay kết quả thô của các script thống kê.
- **Gold có bị nhiễm không**: Không — 4 ảnh gold (`img_001`, `img_012`, `img_020`, `img_036`) được tạo Task riêng và gán tay hoàn toàn trước khi chạy bất kỳ model AI nào (`15 phút`, `30 đối tượng`).
- **Máy / GPU / CVAT dùng; vướng mắc cài đặt**: Windows 11, RAM 16 GB, CPU (`torch 2.15.0+cpu`, `ultralytics 8.4.215`, dùng bộ model nhẹ `yolo26n.pt`, `yoloe-26s-seg.pt`, `sam2.1_t.pt`), CVAT chạy bằng Docker Desktop local tại `http://localhost:8080`. Vướng mắc gặp phải: (1) ban đầu chạy ngoài `.venv` bị thiếu `numpy` do quên activate venv sau khi restart máy; (2) khi tạo Task `batch-review` ban đầu tạo ngoài Project nên bị lỗi `Label 'person' is not registered for this task`, đã tạo lại với bộ nhãn JSON đầy đủ.
- **Dùng cache `data/cache/` thay vì tự chạy?** Không — tự chạy trực tiếp tất cả model trên máy (`t_y26`, `t_y26_c05`, `t_yoloe`, `t_sam`, `y26_all`, `yoloe_all`, `sam2_points`).
- **Lỗi / bug script đã gặp**:
  1. `scripts/make_cvat_labels.py` (dòng 25): CVAT báo lỗi `Attribute: "conf": attribute values must be a non-empty array` khi dán `cvat/labels_generated.json` vào tab Raw vì thuộc tính `text`/`checkbox` để `"values": []`. Đã sửa thành `[str(a.get("default", ""))]` và sinh lại JSON.
  2. `scripts/check_submission.py` (dòng 48): Regex kiểm tra MSHV gốc chỉ nhận số (`\d{5,}`), không khớp MSHV có chữ cái (`2A202603221`). Đã sửa thành `([A-Za-z0-9]{5,})`.
- **Một điều bạn làm khác lần sau**: Khi mở Task review có pre-label, nhất quyết bật nút ẩn toàn bộ nhãn máy ngay ở ảnh đầu tiên và tự đếm vật tối thiểu 10 giây trước khi bật lại để tránh bị neo nhãn và không để lọt bẫy `missing`.
