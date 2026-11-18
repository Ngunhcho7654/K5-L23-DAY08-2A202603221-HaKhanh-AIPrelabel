# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603221 - Hà Khánh

> Tài liệu bổ sung cho `guideline_v1.0_label_contract.md` sau khi chạy 3 model trên tập `pilot`, duyệt lô 24 ảnh `batch`, chấm bẫy và kiểm QC mẫu.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | Nếu có bàn đạp rõ và khung nhỏ kiểu xe đạp → `bicycle`; không có bàn đạp, yên liền thân vỏ nhựa kiểu xe ga → `motorbike`; không rõ → tick `can_review`. *(Chưa gặp ca tranh chấp trong 4 ảnh gold, giữ làm quy tắc dự phòng).* | `img_015.jpg` | `CAL-03` |
| 2 | Khối xe máy đỗ dày đặc gần camera: từng xe hay `ignore_region`? | Chỉ dùng `ignore_region` khi có `>= 5` xe dính sát chồng chéo không tách được biên hoặc phần lớn vật cao `< 20 px`; xe rõ và tách biệt như ở ngõ hẹp phải gán từng xe riêng. | `img_001.jpg`, `img_006.jpg` | `CAL-02` |
| 3 | Hàng nhô ra (mẹt hoa quả, tre, rổ): box xe có gồm hàng không? | Box xe chỉ ôm khung xe và giỏ gắn cố định; không kéo box bao phần hàng rời/mẹt hàng nhô ra hơn `20%` chiều dài xe. Box `person` đi cạnh xe vẫn phải kéo đủ từ đầu tới chân nhìn thấy. | `img_031.jpg`, `img_015.jpg` | `CAL-03` |
| 4 | Người ngồi trong xe ba bánh có mui che: ngưỡng nhìn thấy `>= 20%` có hợp lý? | Giữ ngưỡng `>= 20%` diện tích cơ thể (hoặc thấy rõ từ đầu đến nửa thân trên không bị khung che khuất hoàn toàn); nếu bị che sát ngưỡng gây phân vân giữa 2 người chấm → tick `can_review`. | `img_013.jpg`, `img_003.jpg` | `CAL-04` |
| 5 | Kính chắn gió / cửa sổ xe có thuộc mask xe không? | Kính chắn gió liền khung xe **thuộc** mask xe; khoảng hở khung sườn không có kính nhìn xuyên ra nền hoặc người ngồi ngoài (`>= 5%` diện tích vật) phải **loại** bằng click âm (`--neg`) để không lem mask sang người khác. | `img_004.jpg`, `img_013.jpg` | `CAL-04` |
| 6 | Box máy chồng nhau quá nhiều (trùng cùng một vật): gộp hay giữ? | Không giữ nhiều box cho cùng một vật: giữ 1 box ôm đủ toàn bộ vật (ví dụ cả mui và bánh của xe ba bánh), xoá các box con cắt nửa xe hoặc trùng lặp. | `img_003.jpg`, `img_035.jpg` | `CAL-05` |
| 7 | Có đặt mức `conf` tối thiểu để chấp nhận box máy không cần sửa không? | **Không.** `conf` cao không bảo đảm đúng (box ma `#97 truck` đạt `conf 0.66`, box quang gánh `#112` đạt `0.63`). Dùng `conf` để xếp thứ tự soi kỹ trước, không bỏ qua bước quét ảnh độc lập. | `img_014.jpg`, `img_002.jpg` | `CAL-06` |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, và QC mẫu)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_002.jpg`, `img_011.jpg` | Tuk-tuk có mui và phần thân bị cắt ở mép ảnh | `YOLO26n` gán nhầm thành `truck` (`0.37` / `0.28`) do không có lớp `three_wheeler` trong COCO | Luôn đổi sang `three_wheeler`; ưu tiên `YOLOE` thắng `YOLO26` khi `IoU >= 0.6` | Mục 2 (`three_wheeler`) & Mục 5 (bảng ca khó) |
| `img_014.jpg` | Quang gánh của người bán hàng rong bên lề đường | `YOLOE` gán nhầm 2 quang gánh thành `three_wheeler` (`0.33` và `0.63`) | Xoá bỏ ngay vì quang gánh không có bánh xe, không thuộc lớp phương tiện nào | Mục 2 (vật không thuộc lớp nào) & Mục 5 |
| `img_003.jpg`, `img_035.jpg` | Một xe ba gác / xích lô có 2 box chồng nhau (1 box ôm mui+bánh, 1 box chỉ ôm thân dưới) | `YOLOE` sinh nhiều box lồng nhau từ các prompt đồng nghĩa (`tuk-tuk`, `cyclo`, `cargo tricycle`) | Chỉ giữ 1 box đầy đủ nhất ôm cả mui và bánh (`#18`), xoá box cắt cụt (`#1`) | Mục 4.1.1 (mỗi vật đúng 1 box ôm sát) |
| `img_014.jpg`, `img_035.jpg` | Box "ma" ở mặt đường (`#97 truck 0.66`) và bỏ sót xe máy trắng sát lề trái | Người duyệt bỏ qua bước ẩn nhãn máy trước nên bị thiên kiến neo nhãn (lọt `8/12` bẫy) | Bắt buộc ẩn box máy (`con mắt`) 10 giây ở 100% ảnh trước khi kiểm, sau đó đối chiếu đủ 4 bước | Mục 4.1 & Quy trình review (Bước 0 bắt buộc) |
| `img_023.jpg`, `img_027.jpg` | Ô tô ở xa đường chân trời có chiều cao `14 px` và `17 px` (`< 20 px`) | `YOLO26n` gắn box nhỏ và người duyệt giữ nguyên khiến `validate_labels.py` cảnh báo | Vật đơn lẻ ở quá xa `< 20 px` thì không vẽ box lẻ (chỉ vẽ `ignore_region` nếu thành cụm `>= 5` vật) | Mục 4.1.6 (vật quá nhỏ `< 20 px`) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên viết sẵn | — |
| v1.1 | Chốt 7 câu hỏi mở (xe đạp điện, ngưỡng `ignore_region` gần camera, hàng nhô ra `> 20%`, ngưỡng che `20%`, kính xe và lỗ hở mask SAM, gộp box trùng, không auto-accept theo `conf`) và bổ sung 5 ca khó từ lỗi `YOLO26`/`YOLOE`, bẫy lọt và QC mẫu | `CAL-01`, `CAL-02`, `CAL-03`, `CAL-04`, `CAL-05`, `CAL-06` (`img_001`, `img_002`, `img_003`, `img_004`, `img_006`, `img_011`, `img_013`, `img_014`, `img_015`, `img_023`, `img_027`, `img_031`, `img_035`) |
