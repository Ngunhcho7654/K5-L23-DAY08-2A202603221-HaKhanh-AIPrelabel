# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603221 - Hà Khánh

> Phần bổ sung riêng so với `guideline_v1.0_label_contract.md`, rút ra sau khi chạy 3 model trên tập `pilot`, duyệt lô 24 ảnh `batch` (cả vòng 1 lẫn vòng Rework), chấm ảnh bẫy và tự kiểm QC mẫu.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | Nếu có bàn đạp rõ và khung thanh kiểu xe đạp → `bicycle`; không bàn đạp, vỏ nhựa yên liền kiểu xe tay ga → `motorbike`; ca nào khó phân biệt → tick `can_review`. *(Chưa gặp ca tranh chấp trong 4 ảnh gold, giữ làm quy ước chung).* | `img_015.jpg` | `CAL-03` |
| 2 | Khối xe máy đỗ dày đặc gần camera: từng xe hay `ignore_region`? | Nếu `>= 5` xe đậu xếp hàng chồng lấn vào nhau trong góc khuất/tối không tách nổi ranh giới từng xe (như dãy xe trong ngõ sau `#47` ở `img_006.jpg` hay vỉa hè `img_001.jpg`) thì dùng 1 `ignore_region`; các xe tách rõ phía trước (`#47`, `#61`) vẫn vẽ riêng từng chiếc. | `img_001.jpg`, `img_006.jpg` | `CAL-02` |
| 3 | Hàng nhô ra (mẹt hoa quả, tre, rổ): box xe có gồm hàng không? | Box xe chỉ ôm thân xe và giỏ gắn liền xe; không kéo rộng ra trùm cả mẹt hàng rời nhô quá `20%` chiều dài xe (`#292 bicycle`). Riêng người dắt xe vẫn phải vẽ box `person` kéo xuống hết phần chân nhìn thấy (`#294 person`). | `img_031.jpg`, `img_015.jpg` | `CAL-03` |
| 4 | Người ngồi trong xe ba bánh có mui che: ngưỡng nhìn thấy `>= 20%` có hợp lý? | Vẫn giữ ngưỡng `>= 20%` diện tích cơ thể (hoặc thấy rõ từ đầu tới nửa thân trên không bị thành xe che hết); ca nào mập mờ sát ngưỡng dễ gây lệch giữa hai người chấm → tick `can_review`. | `img_013.jpg`, `img_003.jpg` | `CAL-04` |
| 5 | Kính chắn gió / cửa sổ xe có thuộc mask xe không? | Kính chắn gió liền khung vẫn tính **thuộc** mask xe; nhưng các khoảng hở khung sườn không kính nhìn xuyên sang nền hoặc người ngồi ngoài (`>= 5%` diện tích vật) cần chấm click âm (`--neg`) loại bớt để mask không bị lem. | `img_004.jpg`, `img_013.jpg` | `CAL-04` |
| 6 | Box máy chồng nhau quá nhiều (trùng cùng một vật): gộp hay giữ? | Mỗi vật chỉ giữ đúng `1` box ôm đủ cả thân lẫn bánh/mui xe; xoá hết các box con chỉ ôm một nửa hoặc bị model sinh trùng. | `img_003.jpg`, `img_035.jpg` | `CAL-05` |
| 7 | Có đặt mức `conf` tối thiểu để chấp nhận box máy không cần sửa không? | **Không đặt.** Điểm tin cậy cao không bảo đảm nhãn đúng (`#97 truck` ma trên mặt đường ở vòng 1 đạt tới `0.66`, quang gánh đạt `0.63`). Chỉ dùng `conf` để xếp thứ tự soi kỹ hơn, vẫn phải nhìn kiểm hết toàn bộ box. | `img_014.jpg`, `img_002.jpg` | `CAL-06` |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, và quá trình Rework)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_002.jpg`, `img_011.jpg` | Tuk-tuk có mui và tuk-tuk bị cắt nửa thân ở mép trái ảnh | `YOLO26n` gắn nhầm thành `truck` (`0.37` / `0.28`) vì không có nhãn `three_wheeler` | Luôn đổi về `three_wheeler`; ưu tiên lấy nhãn từ `YOLOE` khi đè nhau `IoU >= 0.6` | Mục 2 (`three_wheeler`) & Mục 5 (bảng ca khó) |
| `img_014.jpg` | Quang gánh của cô bán hàng rong đi bộ bên lề | `YOLOE` nhận nhầm cả hai bên thúng thành `three_wheeler` (`0.33` và `0.63`) | Xoá bỏ hoàn toàn vì quang gánh không có bánh xe, không nằm trong 8 lớp | Mục 2 (vật không thuộc lớp nào) & Mục 5 |
| `img_003.jpg`, `img_035.jpg` | Một xe ba gác / xích lô có 2 box lồng vào nhau (1 box to ôm cả mui, 1 box nhỏ chỉ ôm gầm/bánh) | `YOLOE` sinh trùng box từ các prompt gần nghĩa (`tuk-tuk`, `cyclo`, `cargo tricycle`) | Chỉ giữ 1 box đầy đủ nhất ôm trọn cả mui và bánh (`#1` ở `img_003`, `#326` ở `img_035`), xoá box cụt | Mục 4.1.1 (mỗi vật chỉ 1 box ôm sát) |
| `img_014.jpg`, `img_035.jpg` | Box ma trên mặt đường (`truck 0.66`) và bỏ sót chiếc xe máy trắng sát lề | Vòng 1 nhìn thẳng vào box sẵn nên dính neo nhãn, bỏ lọt `8/12` bẫy → phải Rework | Bắt buộc ẩn toàn bộ box máy (`con mắt`) 10 giây trước ở mọi ảnh rồi mới đối chiếu 4 bước | Mục 4.1 & Quy trình duyệt (Bước 0 bắt buộc) |
| `img_023.jpg`, `img_027.jpg` | Ô tô tít đằng xa chỉ cao `14 px` và `17 px` (`< 20 px`) | `YOLO26n` vẽ box nhỏ khiến `validate_labels.py` cảnh báo | Xe lẻ đứng tách biệt nhưng cao `< 20 px` thì xoá luôn khi Rework, chỉ dùng `ignore_region` khi thành cụm `>= 5` xe | Mục 4.1.6 (vật quá nhỏ `< 20 px`) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt đủ 7 câu hỏi mở (xe đạp điện, ngưỡng `ignore_region` gần camera, hàng nhô ra `> 20%`, ngưỡng che `20%`, kính chắn gió & khoảng hở mask SAM, bỏ box trùng, không auto-accept theo `conf`) và thêm 5 ca khó từ lỗi `YOLO26`/`YOLOE`, bài học Rework và QC mẫu | `CAL-01`, `CAL-02`, `CAL-03`, `CAL-04`, `CAL-05`, `CAL-06` (`img_001`, `img_002`, `img_003`, `img_004`, `img_006`, `img_011`, `img_013`, `img_014`, `img_015`, `img_023`, `img_027`, `img_031`, `img_035`) |
