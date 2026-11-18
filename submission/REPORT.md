# Báo cáo Day 08 — 2A202603221 - Hà Khánh

> Ghi chép lại quá trình tự làm bài từ bước gán tay tập gold, thử model, chạy ngưỡng, duyệt 24 ảnh batch có cài bẫy cho tới lúc soi mask SAM, chốt guideline v1.1 và chạy QC. Toàn bộ số liệu đều trích nguyên từ `submission/qc/*`, `submission/model_log.csv` và `submission/time_log.csv`.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | `YOLO26n` (`yolo26n.pt`) | `0.3` (mặc định) | Chạy nhẹ trên CPU (`0.40s–1.16s/ảnh`), bắt tốt các lớp COCO quen thuộc ở gần (`r01`, `r02`). Khi chạy `threshold_sweep.py` trên 4 ảnh gold thì `bicycle` (`0`), `motorbike` (`4`), `bus` (`0`), `truck` (`1`) đều có dưới 5 vật nên script không cho đặt ngưỡng riêng; còn `person` (tối đa `67%` recall trên 9 vật) và `car` (tối đa `83%` trên 6 vật) cũng không chạm mức `90%` ở ngưỡng `<= 0.5`, nên mình giữ mặc định `0.3`. |
| three_wheeler | `YOLOE-26s-seg` (`yoloe-26s-seg.pt`, prompt: `tuk-tuk`, `cyclo`, `cargo tricycle`) | `0.3` (mặc định) | `YOLO26n` là model closed-set nên không hề có nhãn `three_wheeler`. `YOLOE` (`r03`) nhận ra được tuk-tuk (`img_002`, `img_011`) và xích lô (`img_032`), nên khi gộp mình cho `YOLOE` thắng box `YOLO26` nếu đè nhau `IoU >= 0.6`. Trên 4 ảnh gold có 6 chiếc `three_wheeler`, recall cao nhất ở `<= 0.5` là `83%` (`< 90%`) nên cũng giữ ngưỡng mặc định `0.3`. |

- Lệnh `merge_prelabels.py` đã dùng:
  `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM mình chọn** (`SAM 2.1 tiny` — `sam2.1_t.pt`) và lý do: Máy mình chạy CPU (`16 GB RAM`, GPU GTX 960M 2GB VRAM hơi yếu nên không kéo bản nặng và cũng chưa xin quyền HuggingFace cho `SAM 3`). Bản `sam2.1_t.pt` (~78 MB) nhẹ, chạy ổn trên CPU (`6.56s/ảnh` ở `r04`), ra mask khá gọn cho các đối tượng đứng tách biệt (như `person 0.91` ở `img_004.jpg`), dù gặp khe hở hoặc box đầu vào bị dư thì vẫn dễ bị lem nền (`img_013.jpg`).
- Lỗi nào **không sửa được bằng đổi ngưỡng**? Cách xử lý: Đó là lỗi phân loại nhầm do từ điển cố định của `YOLO26n`: ở `img_002.jpg` (`r01`) tuk-tuk xanh bị gọi thành `truck 0.37`, ở `img_011.jpg` mép tuk-tuk cũng thành `truck 0.28`, còn ở `img_032.jpg` thì xích lô bị bỏ sót hoặc chỉ bắt lẻ bánh xe thành `bicycle`. Khi mình kéo ngưỡng lên `0.50` (`r02`), mấy box này chỉ biến mất chứ không tự đổi sang `three_wheeler` được. Phải kết hợp thêm `YOLOE` (`r03`) rồi gộp ưu tiên (`IoU >= 0.6`) để đè lên box `truck`/`car` trùng chỗ.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng**:
  - Điểm cao mà sai: ở `img_014.jpg` có box `#97 truck` đạt `conf 0.66` nhưng thực ra nằm trống trơn trên mặt đường; cũng ở ảnh này `#112 three_wheeler` đạt `conf 0.63` lại là cái quang gánh hàng rong; còn ở `img_032.jpg` (`r01`) thì `YOLO26n` nhận nhầm người vẽ trên tranh tường thành `person`.
  - Điểm thấp mà đúng: ở `img_011.jpg` tuk-tuk bị cắt mép trái chỉ đạt `0.28` (`r03`), chiếc xích lô bên phải ở `img_032.jpg` chỉ đạt `0.27` (`r03`), hay xe máy phía xa ở `img_016.jpg` đạt `0.41` (`r01`) nhưng đều là vật thật.
- Giới hạn cần lưu ý: Mình quét ngưỡng trên đúng 4 ảnh gold (`26` đối tượng được tính khi chấm), nên vừa bị cỡ mẫu nhỏ (4 lớp có `< 5` vật phải dùng ngưỡng mặc định `0.3`), vừa bị tình trạng chọn và chấm trên cùng một tập dữ liệu — thành ra precision/recall của máy trên gold nhìn sẽ lạc quan hơn thực tế.

## 2. AI error profile (từ `submission/qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 308 | 308/321 = 96.0% | 321 box máy |
| edited | 3 | 3/321 = 0.9% | 321 box máy |
| deleted | 10 | 10/321 = 3.1% | 321 box máy |
| added | 16 | 16/327 = 4.9% | 327 đối tượng sau duyệt |

*(22 đối tượng quanh 12 vị trí bẫy được script tách riêng khỏi bảng này để không làm méo số đo của máy; ngoài ra trong `deleted + added` có `3` ca là đổi nhãn tại chỗ, nên số vật máy bỏ sót thực tế tầm `16 - 3 = 13` vật).*

- Lớp nào máy tệ nhất? Nhìn theo từng lớp thì `three_wheeler` là mệt nhất (`accepted=16`, `edited=2`, `deleted=6`, `added=4`): `YOLOE` hay bắt nhầm quang gánh (`img_014.jpg` #111, #112), xe bán tải (`img_002.jpg`), hoặc sinh hai box đè lên nhau (`img_003.jpg` #1/#18, `img_035.jpg` #336/#337). Bên cạnh đó `motorbike` (`added=5`) và `bus` (`added=2`, `accepted=2`) cũng hay bị `YOLO26n` bỏ sót khi xe đậu sát nhau hoặc bị che (`img_006.jpg`, `img_035.jpg`).
- So với tập gold: Máy đạt Precision **`57.6%` (`19/33`, KTC 95%: `41–73%`)** và Recall **`73.1%` (`19/26`, KTC 95%: `54–86%`)** (`TP=19`, `FP=14`, `FN=7` trên 4 ảnh gold).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: Mình chỉ bắt được **`4/12 = 33.3%`** (`KTC 95%: 14–61%`, dưới xa ngưỡng `80%`). Theo từng loại thì `extra: 2/3`, `missing: 0/3`, `wrong_class: 1/3`, `loose: 1/3`. Tám bẫy bị lọt gồm: `img_018.jpg (missing)`, `img_024.jpg (missing)`, `img_035.jpg (missing)`, `img_025.jpg (wrong_class)`, `img_028.jpg (wrong_class)`, `img_005.jpg (loose)`, `img_022.jpg (loose)` và `img_014.jpg (extra)`. Lý do thật là khi vào CVAT mình đã bỏ qua **Bước 0 (ẩn nhãn máy để tự nhìn ảnh trước)** và lướt khá vội, nên mắt cứ đi theo các box sẵn có: hễ không có box thì bỏ sót luôn (`0/3` bẫy `missing`, đặc biệt ở ảnh đông xe như `img_018`, `img_024` và xe máy trắng bị khuất một phần ở `img_035`), thấy box nào điểm cao là tin (`#97 truck 0.66` trên mặt đường ở `img_014`), còn box hơi xô lệch (`loose` ở `img_005`, `img_022`) thì ngại kéo lại.
- Tỷ lệ `accepted` lên tới **`96.0%` (`308/321`)** nhưng bắt bẫy chỉ **`33.3%` (`4/12`)** cho thấy đúng trạng thái chủ quan của mình lúc duyệt: vì nhãn máy nhìn sơ qua trông có vẻ ổn rồi nên mình bấm giữ nguyên quá nhiều, gần như chỉ sửa những chỗ quá lộ liễu chứ chưa kiểm tra độc lập từng đối tượng.

## 4. Máy có đáng dùng không?

- Thời gian (từ `submission/time_log.csv`): Tự vẽ tay 4 ảnh gold hết `15 phút` (**`3.75 phút/ảnh`**), còn duyệt 24 ảnh có pre-label hết `20 phút` (**`0.83 phút/ảnh`**), tức nhanh hơn tầm `4.5×` (tiết kiệm ~`77.8%` thời gian mỗi ảnh). Tuy nhiên mức chênh này có phần bị kéo lên do hai yếu tố: lúc sang tập batch mình đã quen tay và thuộc quy tắc từ 4 ảnh gold trước đó, cộng thêm việc duyệt nhanh nên bị lọt bẫy nhiều.
- Kết luận: **Vẫn đáng dùng cho `person`, `motorbike`, `car`** vì đỡ rất nhiều công kẻ box ở những ảnh đường phố đông người/xe, nhưng **chỉ hiệu quả khi chịu khó rà kỹ bước kiểm soát** — nhất là với `three_wheeler` (hay nhiễu và trùng box) và các xe bị che một phần.
- **Ba quy tắc giảm thiên kiến của mình** (đo/kiểm tra được):
  1. Ở **100% ảnh**, phải bấm nút con mắt ẩn toàn bộ box máy và tự đếm nhẩm tổng số người/xe trong khoảng `10–15 giây` trước khi bật lại (kiểm chứng trực tiếp qua việc hết lọt bẫy `missing`: mục tiêu `3/3`).
  2. Chia 24 ảnh thành `3` lượt (mỗi lượt `8` ảnh), nghỉ bắt buộc `3–5 phút` giữa mỗi lượt và giữ nhịp duyệt tối thiểu `>= 1.5 phút/ảnh` thay vì chạy một mạch (kiểm tra được qua dòng `break` và `minutes` trong `time_log.csv`).
  3. Rà riêng một lượt cho các lớp hay bị nhầm (`three_wheeler`, `truck`, `bicycle`) bất kể `conf` cao hay thấp (kể cả `>= 0.6`), kiểm tra xem có bánh xe/khung xe thật không và chỉ giữ `1` box cho mỗi xe trước khi lưu (mục tiêu bắt bẫy đạt `>= 10/12 = 83.3%`).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| `img_013.jpg` | `SAM 2.1 tiny` (`sam2.1_t.pt`) | `4` (`2` dương, `2` âm) | Đã loại người lái (`neg 720,400`), nhưng qua khe hở sườn xe thì mask vẫn ăn vào người ngồi ăn phía sau dù có `neg 460,340` | `1 polygon`, `41.0% ảnh` (`286,611 px`); bị rò mask qua khung cửa hở | [tuktuk_img013.jpg](submission/masks/tuktuk_img013.jpg) |
| `img_004.jpg` | `SAM 2.1 tiny` (`sam2.1_t.pt`) | `4` (`2` dương, `2` âm) | Không dính người đứng cạnh (`neg 465,300`); phần kính chắn gió phía trước nằm trọn trong mask xe | `1 polygon`, `13.2% ảnh` (`92,294 px`); tách sạch khỏi xe xanh bên cạnh và nền gạch | [tuktuk_img004.jpg](submission/masks/tuktuk_img004.jpg) |
| `img_032.jpg` | `SAM 2.1 tiny` (`sam2.1_t.pt`) | `4` (`2` dương, `2` âm) | Không dính bác đạp xích lô đứng sát bên phải (`neg 560,320`) | `1 polygon`, `7.8% ảnh` (`46,115 px`); ôm khá khít đệm hoa và cả 2 bánh trước | [cyclo_img032.jpg](submission/masks/cyclo_img032.jpg) |

- Đối chiếu guideline mục 4.2: Quy tắc 4.2.1 và 4.2.4 giúp mình đặt click âm khá tự tin để tách người lái và người đứng cạnh ra khỏi xe (`img_013`, `img_004`, `img_032`). Tuy nhiên với xe có khung sườn hở nhìn xuyên qua người khác như `img_013.jpg`, quy tắc 4.2.2 (lỗ `< 5%`) và 4.2.3 (kính xe) ở v1.0 chưa nói rõ, nên mình đã bổ sung cụ thể trong `guideline_v1.1_calibrated.md`.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: Bản CVAT Docker trên máy mình chưa cài serverless Nuclio nên mục AI Tools chưa có SAM để click trực tiếp. Dùng `sam2_points.py` offline giúp lưu lại được đúng toạ độ và dễ tái lập, nhưng bất tiện hơn khi phải thử lại toạ độ bằng dòng lệnh (như lúc đầu `img_004` và `img_032` click hơi thấp bị ăn xuống nền đường trước khi căn lại).

## 6. Calibration (Part 6)

- Số quyết định: **`6` ca** (`CAL-01` đến `CAL-06` trong `submission/qc/calibration_decisions.csv`). Đáng chú ý nhất là `CAL-01` (không dùng ngưỡng để sửa lỗi `YOLO26` gọi tuk-tuk là `truck`/`car` ở `img_002`, `img_011`), `CAL-05` (xoá box `YOLOE` bắt nhầm quang gánh và box trùng ở `img_014`, `img_003`, `img_035`), và `CAL-06` (không đặt ngưỡng `conf` auto-accept).
- Quy tắc mới đã đưa vào [guideline_v1.1_calibrated.md](submission/guidelines/guideline_v1.1_calibrated.md): chốt toàn bộ 7 câu hỏi mở của mục 6 bản `v1.0` và thêm 5 ca khó thực tế có dẫn mã ảnh cụ thể.
- Nguyên nhân gốc: Xuất phát từ **cả 3 phía**: guideline v1.0 cố ý chừa chỗ mở ở các ca ranh giới (`CAL-02`, `CAL-03`, `CAL-04`), mô hình pre-label chưa quen vật thể địa phương như quang gánh/xích lô (`CAL-01`, `CAL-05`), và bản thân mình lúc duyệt lướt bị neo nhãn (`CAL-06`).

## 7. QC và quyết định (Part 6)

- Mẫu: **`5/24` ảnh** (`0` ảnh `can_review` + `5` ảnh ngẫu nhiên `seed = 42`: `img_003.jpg`, `img_006.jpg`, `img_014.jpg`, `img_031.jpg`, `img_035.jpg`), gồm **`71` đối tượng**. Người kiểm là chính mình (`HaKhanh`), có hạn chế là tự kiểm lại bài mình vừa duyệt nên dễ dính cùng điểm mù.
- Lỗi theo **đối tượng**: **`8/71 = 11.3%`** · theo **ảnh**: **`5/5 = 100.0%`** (vượt ngưỡng Release `<= 2%`).
- Top lỗi (từ `submission/qc/qc_review.csv` và `submission/qc/qc_report_auto.md`):
  1. `thua` (`5/8` lỗi): `img_014.jpg` (box ma `#97 truck 0.66` trên mặt đường và 2 quang gánh `#111`, `#112` gán thành `three_wheeler`), `img_003.jpg` (`#1` trùng với `#18` trên cùng xe ba gác), `img_035.jpg` (`#337` trùng với `#336` trên cùng chiếc xích lô).
  2. `bo_sot` (`2/8` lỗi): `img_006.jpg` (bỏ sót mấy xe máy đậu sát cạnh `#50 motorbike` trong ngõ), `img_035.jpg` (sót xe máy trắng cạnh `#340 person` và người ngồi trên `#326 motorbike`).
  3. `sai_vi_tri_kich_thuoc` (`1/8` lỗi): `img_031.jpg` (`#292 person` bị cắt ngang hông dù bên dưới vẫn thấy chân).
- **Giới hạn mẫu nhỏ**: Với `5/5` ảnh mẫu có lỗi, cận trên khoảng tin cậy 95% (Wilson) trong `qc_report_auto.md` là **`100%`**. Kể cả giả sử mẫu 5 ảnh không ra lỗi nào (`0/5`), cận trên 95% vẫn lên tới ~`43%–60%`, nên 5 ảnh mẫu không bao giờ đủ để chứng minh đạt ngưỡng `<= 2%` mà chủ yếu dùng để phát hiện lỗi hệ thống.
- Script kiểm được / không kiểm được: Script kiểm tra được cấu trúc COCO, tên lớp, box tràn viền, vật `< 20 px` (cảnh báo 2 box `car` `14 px` và `17 px` ở `img_023.jpg`, `img_027.jpg`), IoU và đáp án bẫy; nhưng **không** tự phát hiện được box cắt ngang hông (`#292`), vật bị nhận nhầm như quang gánh (`#111/#112`), hay xe bị cả mình lẫn máy cùng bỏ sót (`img_006.jpg`).

**Quyết định: `Rework`, ngày `2026-11-18`.** Lý do theo cây `Hold -> Rework -> Release`: Không `Hold` vì các điểm mơ hồ của guideline đã được chốt sang `v1.1` và không có ảnh hỏng; rơi thẳng vào **`Rework`** vì đủ cả 3 tiêu chí cần sửa lại: tỷ lệ ảnh lỗi `5/5 = 100.0% > 2%`, có lỗi hệ thống lặp lại (`thua` chiếm `5/8` lỗi), và bắt bẫy chỉ đạt `4/12 = 33.3% (< 80%)`.
Việc cần làm tiếp cho vòng Rework: Dùng bản `guideline_v1.1_calibrated.md` cùng 3 quy tắc chống thiên kiến (ẩn hết box máy trước khi xem ảnh) để duyệt lại toàn bộ `24` ảnh trong CVAT — xoá các box trùng/thừa (`img_003`, `img_014`, `img_035`), xoá hoặc gộp ignore cho 2 box `< 20 px` (`img_023`, `img_027`), vẽ bù vật sót (`img_006`, `img_035`), kéo hết chân `#292` (`img_031`), rồi xuất `submission/labels/qc_fixed.json` để chạy lại `build_release.py` và kiểm QC vòng 2.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài**: Mình có dùng trợ lý AI (Antigravity) để giải thích quy tắc guideline (như cách vẽ `person` chồng lên `motorbike`), hướng dẫn thao tác CVAT, chạy các script trong `scripts/`, sửa 2 lỗi nhỏ ở script cài đặt và tổng hợp lại số liệu đối chiếu từ `viz/`, ảnh QC và `qc_report_auto.md`. Việc vẽ tay 4 ảnh gold và duyệt 24 ảnh batch trên CVAT là do mình tự làm hoàn toàn, không chỉnh sửa tay các file kết quả thô của script.
- **Gold có bị nhiễm không**: Không — mình tạo Task `gold-manual` và gán tay xong 4 ảnh (`15 phút`, `30` đối tượng) rồi mới bắt đầu chạy các model pre-label ở Part 2.
- **Máy / GPU / CVAT dùng; vướng mắc cài đặt**: Windows 11, RAM 16 GB, GPU GTX 960M 2 GB VRAM nhưng chạy model trên CPU (`torch 2.15.0+cpu`, `ultralytics 8.4.215`, bộ nhẹ `yolo26n.pt`, `yoloe-26s-seg.pt`, `sam2.1_t.pt`); CVAT chạy qua Docker Desktop local ở `http://localhost:8080`. Lúc làm gặp 2 trục trặc nhỏ: quên activate `.venv` sau khi restart máy nên thiếu `numpy`, và khi tạo Task `batch-review` ban đầu tạo ngoài Project nên bị báo `Label 'person' is not registered for this task` (sau đó dán lại nhãn JSON đầy đủ).
- **Dùng cache `data/cache/` thay vì tự chạy?** Không — mình chạy trực tiếp toàn bộ model trên máy (`t_y26`, `t_y26_c05`, `t_yoloe`, `t_sam`, `y26_all`, `yoloe_all`, `sam2_points`).
- **Lỗi / bug script đã gặp**:
  1. `scripts/make_cvat_labels.py` (dòng 25): Khi dán `cvat/labels_generated.json` vào CVAT, CVAT báo lỗi `Attribute: "conf": attribute values must be a non-empty array` vì thuộc tính `text`/`checkbox` để `"values": []`. Mình đã sửa thành `[str(a.get("default", ""))]` và chạy lại script.
  2. `scripts/check_submission.py` (dòng 48): Regex kiểm tra MSHV gốc (`\d{5,}`) chỉ nhận chữ số nên không khớp mã có chữ cái `2A202603221`. Mình đã nới thành `([A-Za-z0-9]{5,})`.
- **Một điều làm khác lần sau**: Bất kỳ khi nào mở Task review nhãn máy, việc đầu tiên ở mỗi ảnh là bấm phím ẩn toàn bộ box, tự đếm người/xe khoảng 10 giây rồi mới bật lại — tuyệt đối không vừa nhìn ảnh vừa nhìn box có sẵn ngay từ giây đầu.
