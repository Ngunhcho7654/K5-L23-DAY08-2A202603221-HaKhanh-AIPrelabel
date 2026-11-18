# Báo cáo Day 08 — 2A202603221 - Hà Khánh

> Ghi chép lại toàn bộ quá trình làm bài từ gán tay tập gold, thử 3 model, quét ngưỡng, duyệt 24 ảnh batch có ảnh bẫy (gồm cả vòng 1 bị thiên kiến tự động hóa và vòng 2 Rework lại), chạy mask SAM, calibration guideline v1.1 và QC cuối. Mọi số liệu đều trích nguyên từ `submission/qc/*`, `submission/model_log.csv` và `submission/time_log.csv`.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | `YOLO26n` (`yolo26n.pt`) | `0.3` (mặc định) | Chạy nhẹ trên CPU (`0.40s–1.16s/ảnh`), bắt tốt các lớp COCO quen thuộc ở gần (`r01`, `r02`). Khi chạy `threshold_sweep.py` trên 4 ảnh gold thì `bicycle` (`0`), `motorbike` (`4`), `bus` (`0`), `truck` (`1`) đều có dưới 5 vật nên script không cho đặt ngưỡng riêng; còn `person` (tối đa `67%` recall trên 9 vật) và `car` (tối đa `83%` trên 6 vật) cũng không chạm mức `90%` ở ngưỡng `<= 0.5`, nên mình giữ mặc định `0.3`. |
| three_wheeler | `YOLOE-26s-seg` (`yoloe-26s-seg.pt`, prompt: `tuk-tuk`, `cyclo`, `cargo tricycle`) | `0.3` (mặc định) | `YOLO26n` là model closed-set nên không hề có nhãn `three_wheeler`. `YOLOE` (`r03`) nhận ra được tuk-tuk (`img_002`, `img_011`) và xích lô (`img_032`), nên khi gộp mình cho `YOLOE` thắng box `YOLO26` nếu đè nhau `IoU >= 0.6`. Trên 4 ảnh gold có 6 chiếc `three_wheeler`, recall cao nhất ở `<= 0.5` là `83%` (`< 90%`) nên cũng giữ ngưỡng mặc định `0.3`. |

- Lệnh `merge_prelabels.py` đã dùng:
  `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM mình chọn** (`SAM 2.1 tiny` — `sam2.1_t.pt`) và lý do: Máy mình chạy CPU (`16 GB RAM`, GPU GTX 960M 2GB VRAM hơi yếu nên không kéo bản nặng và cũng chưa xin quyền HuggingFace cho `SAM 3`). Bản `sam2.1_t.pt` (~78 MB) nhẹ, chạy ổn trên CPU (`6.56s/ảnh` ở `r04`), ra mask khá gọn cho các đối tượng đứng tách biệt (như `person 0.91` ở `img_004.jpg`), dù gặp khe hở hoặc box đầu vào bị dư thì vẫn dễ bị lem nền (`img_013.jpg`).
- Lỗi nào **không sửa được bằng đổi ngưỡng**? Cách xử lý: Đó là lỗi phân loại nhầm do từ điển cố định của `YOLO26n`: ở `img_002.jpg` (`r01`) tuk-tuk xanh bị gọi thành `truck 0.37`, ở `img_011.jpg` mép tuk-tuk cũng thành `truck 0.28`, còn ở `img_032.jpg` xích lô bị bỏ sót hoặc chỉ bắt lẻ bánh xe thành `bicycle`. Khi mình kéo ngưỡng lên `0.50` (`r02`), mấy box này chỉ biến mất chứ không tự đổi sang `three_wheeler` được. Phải kết hợp thêm `YOLOE` (`r03`) rồi gộp ưu tiên (`IoU >= 0.6`) để đè lên box `truck`/`car` trùng chỗ.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng**:
  - Điểm cao mà sai: ở `img_014.jpg` trước khi sửa có box ma `truck` đạt `conf 0.66` nhưng nằm trống trơn trên mặt đường; cũng ở ảnh này `YOLOE` bắt nhầm quang gánh hàng rong thành `three_wheeler` tới `conf 0.63`; còn ở `img_032.jpg` (`r01`) thì `YOLO26n` nhận nhầm người vẽ trên tranh tường thành `person`.
  - Điểm thấp mà đúng: ở `img_011.jpg` tuk-tuk bị cắt mép trái chỉ đạt `0.28` (`r03`), chiếc xích lô bên phải ở `img_032.jpg` chỉ đạt `0.27` (`r03`), hay xe máy phía xa ở `img_016.jpg` đạt `0.41` (`r01`) nhưng đều là vật thật.
- Giới hạn cần lưu ý: Mình quét ngưỡng trên đúng 4 ảnh gold (`26` đối tượng được tính khi chấm), nên vừa bị cỡ mẫu nhỏ (4 lớp có `< 5` vật phải dùng ngưỡng mặc định `0.3`), vừa bị tình trạng chọn và chấm trên cùng một tập dữ liệu — thành ra precision/recall của máy trên gold nhìn sẽ lạc quan hơn thực tế.

## 2. AI error profile (từ `submission/qc/review_stats.md` sau Rework)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 296 | 296/321 = 92.2% | 321 box máy |
| edited | 7 | 7/321 = 2.2% | 321 box máy |
| deleted | 18 | 18/321 = 5.6% | 321 box máy |
| added | 23 | 23/326 = 7.1% | 326 đối tượng sau duyệt |

*(24 đối tượng quanh 12 vị trí bẫy được script tách riêng khỏi bảng này để không làm méo số đo của máy; ngoài ra trong `deleted + added` có `3` ca là đổi nhãn tại chỗ, nên số vật máy bỏ sót thực tế tầm `23 - 3 = 20` vật).*

- Lớp nào máy tệ nhất? Nhìn theo từng lớp thì **`three_wheeler`** là vất vả nhất (`accepted=11`, `edited=3`, `deleted=10`, `added=4`): gần một nửa box máy vẽ ra bị xoá (`10` box) vì `YOLOE` hay bắt nhầm quang gánh (`img_014.jpg`), xe bán tải (`img_002.jpg`), hoặc sinh hai box đè lên nhau (`img_003.jpg`, `img_035.jpg`). Bên cạnh đó **`car`** (`deleted=3`, `added=7`) và **`motorbike`** (`added=5`) cũng bị `YOLO26n` bỏ sót khá nhiều khi xe đậu sát nhau, bị che hoặc ở ngã tư đông ban đêm (`img_024.jpg`, `img_035.jpg`).
- So với tập gold: Máy đạt Precision **`57.6%` (`19/33`, KTC 95%: `41–73%`)** và Recall **`73.1%` (`19/26`, KTC 95%: `54–86%`)** (`TP=19`, `FP=14`, `FN=7` trên 4 ảnh gold).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy:
  - **Lượt 1 (duyệt vội `20 phút`)**: Mình chỉ bắt được **`4/12 = 33.3%`** (`extra: 2/3, missing: 0/3, wrong_class: 1/3, loose: 1/3`), bỏ lọt 8 bẫy gồm `img_018.jpg (missing)`, `img_024.jpg (missing)`, `img_035.jpg (missing)`, `img_025.jpg (wrong_class)`, `img_028.jpg (wrong_class)`, `img_005.jpg (loose)`, `img_022.jpg (loose)` và `img_014.jpg (extra)`. Lý do thật là khi đó mình bỏ qua **Bước 0 (ẩn nhãn máy để tự quét ảnh trước)** và tin vào box sẵn có: hễ máy không vẽ là bỏ sót (`0/3` bẫy `missing`), thấy box điểm cao (`truck 0.66` ở `img_014`) là giữ luôn, và ngại kéo lại những box chỉ lệch nhẹ trong ảnh đông xe (`img_005`, `img_022`).
  - **Lượt 2 (Rework thêm `15 phút` theo checklist)**: Sau khi tự đối chiếu vòng 1 và áp dụng lại đúng checklist 4 bước trên CVAT, mình đã rà và xử lý hết các ca lọt để đưa tỷ lệ bắt bẫy cuối cùng lên **`12/12 = 100.0%`** (`KTC 95%: 76–100%`; `extra: 3/3`, `missing: 3/3`, `wrong_class: 3/3`, `loose: 3/3`).
- Bài học về thiên kiến tự động hóa: Con số lượt 1 (`accepted 96.0%` nhưng bắt bẫy chỉ `33.3%`) so với lượt 2 (`accepted 92.2%`, bắt bẫy `100.0%`) cho thấy rất rõ cái bẫy của sự tiện lợi: nhãn máy càng "trông hợp lý" thì người duyệt càng lười kiểm, dẫn tới tỷ lệ giữ nguyên cao giả tạo trong khi hàng loạt lỗi bỏ sót và sai lớp bị lọt thẳng vào bộ dữ liệu.

## 4. Máy có đáng dùng không?

- Thời gian (từ `submission/time_log.csv`): Tự vẽ tay 4 ảnh gold hết `15 phút` (**`3.75 phút/ảnh`**), còn duyệt 24 ảnh có pre-label hết tổng cộng `35 phút` (**`1.46 phút/ảnh`**, gồm `20 phút` lượt 1 bị nhanh ẩu + `15 phút` Rework kỹ lại), vẫn nhanh hơn gán tay khoảng `2.6×` (tiết kiệm ~`61.1%` thời gian mỗi ảnh). Thiên lệch cần nói: lúc sang tập batch mình đã quen tay và thuộc luật từ 4 ảnh gold trước đó.
- Kết luận: **Đáng dùng cho `person`, `motorbike`, `car`** vì tiết kiệm hơn nửa thời gian vẽ trên các ảnh phố đông người/xe, nhưng **không được bỏ qua bước kiểm soát** — nhất là với `three_wheeler` (tỉ lệ xoá `10` box vì nhiễu và trùng) và lỗi bỏ sót vật ở xa/bị khuất (`added = 23/326 = 7.1%`).
- **Ba quy tắc giảm thiên kiến của mình** (đo/kiểm tra được):
  1. Ở **100% ảnh**, phải bấm nút con mắt ẩn toàn bộ box máy và tự đếm nhẩm tổng số người/xe trong khoảng `10–15 giây` trước khi bật lại (kiểm chứng trực tiếp qua việc bắt đủ `3/3` bẫy `missing`).
  2. Chia 24 ảnh thành `3` lượt (mỗi lượt `8` ảnh), nghỉ bắt buộc `3–5 phút` giữa mỗi lượt và giữ nhịp duyệt tối thiểu `>= 1.5 phút/ảnh` thay vì chạy nhanh một mạch (kiểm tra được qua `minutes` trong `time_log.csv`).
  3. Rà riêng một lượt cho các lớp hay bị nhầm (`three_wheeler`, `truck`, `bicycle`) bất kể `conf` cao hay thấp (kể cả `>= 0.6`), kiểm tra xem có bánh xe/khung xe thật không và chỉ giữ `1` box cho mỗi xe trước khi lưu (mục tiêu bắt bẫy `>= 10/12 = 83.3%`).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| `img_013.jpg` | `SAM 2.1 tiny` (`sam2.1_t.pt`) | `4` (`2` dương, `2` âm) | Đã loại người lái (`neg 720,400`), nhưng qua khe hở sườn xe thì mask vẫn ăn vào người ngồi ăn phía sau dù có `neg 460,340` | `1 polygon`, `41.0% ảnh` (`286,611 px`); bị rò mask qua khung cửa hở | [tuktuk_img013.jpg](submission/masks/tuktuk_img013.jpg) |
| `img_004.jpg` | `SAM 2.1 tiny` (`sam2.1_t.pt`) | `4` (`2` dương, `2` âm) | Không dính người đứng cạnh (`neg 465,300`); phần kính chắn gió phía trước nằm trọn trong mask xe | `1 polygon`, `13.2% ảnh` (`92,294 px`); tách sạch khỏi xe xanh bên cạnh và nền gạch | [tuktuk_img004.jpg](submission/masks/tuktuk_img004.jpg) |
| `img_032.jpg` | `SAM 2.1 tiny` (`sam2.1_t.pt`) | `4` (`2` dương, `2` âm) | Không dính bác đạp xích lô đứng sát bên phải (`neg 560,320`) | `1 polygon`, `7.8% ảnh` (`46,115 px`); ôm khá khít đệm hoa và cả 2 bánh trước | [cyclo_img032.jpg](submission/masks/cyclo_img032.jpg) |

- Đối chiếu guideline mục 4.2: Quy tắc 4.2.1 và 4.2.4 giúp mình đặt click âm khá tự tin để tách người lái và người đứng cạnh ra khỏi xe (`img_013`, `img_004`, `img_032`). Tuy nhiên với xe có khung sườn hở nhìn xuyên qua người khác như `img_013.jpg`, quy tắc 4.2.2 (lỗ `< 5%`) và 4.2.3 (kính xe) ở v1.0 chưa nói rõ, nên mình đã bổ sung cụ thể trong `guideline_v1.1_calibrated.md`.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: Bản CVAT Docker trên máy mình chưa cài serverless Nuclio nên mục AI Tools chưa có SAM để click trực tiếp. Dùng `sam2_points.py` offline giúp lưu lại được đúng toạ độ và dễ tái lập, nhưng bất tiện hơn khi phải thử lại toạ độ bằng dòng lệnh (như lúc đầu `img_004` và `img_032` click hơi thấp bị ăn xuống nền đường trước khi căn lại).

## 6. Calibration (Part 6)

- Số quyết định: **`6` ca** (`CAL-01` đến `CAL-06` trong `submission/qc/calibration_decisions.csv`). Quan trọng nhất là `CAL-01` (không dùng ngưỡng để sửa lỗi `YOLO26` gọi tuk-tuk là `truck`/`car` ở `img_002`, `img_011`), `CAL-05` (xoá box `YOLOE` bắt nhầm quang gánh và box trùng ở `img_014`, `img_003`, `img_035`), và `CAL-06` (không đặt ngưỡng `conf` auto-accept sau bài học bị lọt bẫy ở vòng 1).
- Quy tắc mới đã đưa vào [guideline_v1.1_calibrated.md](submission/guidelines/guideline_v1.1_calibrated.md): chốt toàn bộ 7 câu hỏi mở của mục 6 bản `v1.0` và thêm 5 ca khó thực tế có dẫn mã ảnh cụ thể.
- Nguyên nhân gốc: Xuất phát từ **cả 3 phía**: guideline v1.0 cố ý chừa chỗ mở ở các ca ranh giới (`CAL-02`, `CAL-03`, `CAL-04`), mô hình pre-label chưa quen vật thể địa phương như quang gánh/xích lô (`CAL-01`, `CAL-05`), và bản thân mình lúc duyệt lượt 1 bị neo nhãn (`CAL-06`).

## 7. QC và quyết định (Part 6)

- Mẫu: **`5/24` ảnh** (`0` ảnh `can_review` + `5` ảnh ngẫu nhiên `seed = 42`: `img_003.jpg`, `img_006.jpg`, `img_014.jpg`, `img_031.jpg`, `img_035.jpg`), gồm **`67` đối tượng**. Người kiểm là chính mình (`HaKhanh`), có hạn chế là tự kiểm lại bài mình vừa duyệt nên vẫn dễ có điểm mù chủ quan.
- Kết quả QC theo 2 vòng:
  - **Vòng 1 (trước Rework)**: phát hiện `8/71 = 11.3%` lỗi theo đối tượng và `5/5 = 100.0%` ảnh có lỗi (gồm `thua`: 5, `bo_sot`: 2, `sai_vi_tri_kich_thuoc`: 1) cùng với bắt bẫy `4/12 = 33.3%` → ra quyết định **`Rework`** và sửa lại toàn bộ trên CVAT.
  - **Vòng 2 (bản hiện tại trong `submission/qc/qc_report_auto.md`)**: Lỗi theo **đối tượng**: **`0/67 = 0.0%`** · theo **ảnh**: **`0/5 = 0.0%`** (đạt ngưỡng `<= 2%`), không còn lỗi hệ thống lặp lại và bắt bẫy đạt **`12/12 = 100.0%`**.
- Top lỗi (vòng 2): Không ghi nhận lỗi nào trên 5 ảnh mẫu (các lỗi vòng 1 như box trùng `#1/#18` ở `img_003.jpg`, thiếu `ignore_region` ở `img_006.jpg`, box ma `truck` và quang gánh ở `img_014.jpg`, box cắt hông ở `img_031.jpg`, sót xe máy trắng ở `img_035.jpg`, cùng 2 box `< 20 px` ở `img_023.jpg`, `img_027.jpg` đã được xử lý sạch).
- **Giới hạn mẫu nhỏ**: Dù kết quả vòng 2 là `0/5` ảnh lỗi, `qc_report_auto.md` chỉ ra cận trên khoảng tin cậy 95% (Wilson) của tỉ lệ ảnh lỗi thật vẫn lên tới **`43%`** (và theo quy tắc 3 là `~3/5 = 60%`). Vì vậy với mẫu chỉ 5 ảnh, con số `0%` không thể chứng minh chắc chắn toàn lô đạt dưới ngưỡng `2%` — muốn khẳng định chặt chẽ hơn cần cỡ mẫu lớn hơn và người kiểm độc lập.
- Script kiểm được / không kiểm được: Script kiểm tra được cấu trúc COCO, tên lớp, box tràn viền, vật `< 20 px` (`validate_labels.py --all` sạch cả cảnh báo), IoU và đáp án bẫy; nhưng **không** tự kiểm được độ ôm sát biên từng box, vật thật bị cả người lẫn máy cùng sót ngoài tập gold, hay việc người duyệt có làm đúng bước 0 ẩn nhãn hay không.

**Quyết định: `Release` (sau 1 vòng Rework), ngày `2026-11-18`.** Lý do (đi theo cây `Hold -> Rework -> Release`): Không `Hold` vì các câu hỏi mở của guideline đã được chốt sang `v1.1` và không có ảnh hỏng; không còn rơi vào `Rework` như vòng 1 vì sau khi sửa lại toàn bộ lô, ảnh lỗi mẫu QC đạt `0/5 = 0.0% <= 2%`, không còn lỗi hệ thống lặp lại, và tỷ lệ bắt bẫy đạt `12/12 = 100.0% >= 80%`.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài**: Mình có dùng trợ lý AI (Antigravity) để giải thích quy tắc guideline (như cách vẽ `person` chồng lên `motorbike`), hướng dẫn thao tác CVAT, chạy các script trong `scripts/`, sửa 2 lỗi nhỏ ở script cài đặt, chỉ vị trí các frame còn sót ở vòng Rework và tổng hợp đối chiếu số liệu từ `viz/`, ảnh QC và `qc_report_auto.md`. Việc vẽ tay 4 ảnh gold và chỉnh sửa toàn bộ nhãn trên CVAT là do mình tự làm trực tiếp, không sửa tay các file thống kê đầu ra của script.
- **Gold có bị nhiễm không**: Không — mình tạo Task `gold-manual` và gán tay xong 4 ảnh (`15 phút`, `30` đối tượng) rồi mới bắt đầu chạy các model pre-label ở Part 2.
- **Máy / GPU / CVAT dùng; vướng mắc cài đặt**: Windows 11, RAM 16 GB, GPU GTX 960M 2 GB VRAM nhưng chạy model trên CPU (`torch 2.15.0+cpu`, `ultralytics 8.4.215`, bộ nhẹ `yolo26n.pt`, `yoloe-26s-seg.pt`, `sam2.1_t.pt`); CVAT chạy qua Docker Desktop local ở `http://localhost:8080`. Lúc làm gặp 2 trục trặc nhỏ: quên activate `.venv` sau khi restart máy nên thiếu `numpy`, và khi tạo Task `batch-review` ban đầu tạo ngoài Project nên bị báo `Label 'person' is not registered for this task` (sau đó dán lại nhãn JSON đầy đủ).
- **Dùng cache `data/cache/` thay vì tự chạy?** Không — mình chạy trực tiếp toàn bộ model trên máy (`t_y26`, `t_y26_c05`, `t_yoloe`, `t_sam`, `y26_all`, `yoloe_all`, `sam2_points`).
- **Lỗi / bug script đã gặp**:
  1. `scripts/make_cvat_labels.py` (dòng 25): Khi dán `cvat/labels_generated.json` vào CVAT, CVAT báo lỗi `Attribute: "conf": attribute values must be a non-empty array` vì thuộc tính `text`/`checkbox` để `"values": []`. Mình đã sửa thành `[str(a.get("default", ""))]` và chạy lại script.
  2. `scripts/check_submission.py` (dòng 48): Regex kiểm tra MSHV gốc (`\d{5,}`) chỉ nhận chữ số nên không khớp mã có chữ cái `2A202603221`. Mình đã nới thành `([A-Za-z0-9]{5,})`.
- **Một điều làm khác lần sau**: Bất kỳ khi nào mở Task review nhãn máy, việc đầu tiên ở mỗi ảnh là bấm phím ẩn toàn bộ box, tự đếm người/xe khoảng 10 giây rồi mới bật lại ngay từ vòng 1 — đỡ mất thêm một lượt Rework.
