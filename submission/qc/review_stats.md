# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 321; đối tượng sau duyệt: 327; ngưỡng accepted: IoU >= 0.9
- Đã **loại 22 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 308 | 308/321 = 96.0% | 321 box máy |
| edited (sửa) | 3 | 3/321 = 0.9% | 321 box máy |
| deleted (xoá) | 10 | 10/321 = 3.1% | 321 box máy |
| added (vẽ thêm) | 16 | 16/327 = 4.9% | 327 đối tượng sau duyệt |

Trong số deleted + added, **3 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 13.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 164 | 0 | 2 | 3 |
| bicycle | 4 | 0 | 1 | 1 |
| motorbike | 81 | 1 | 0 | 5 |
| car | 40 | 0 | 0 | 0 |
| bus | 2 | 0 | 0 | 2 |
| truck | 1 | 0 | 1 | 1 |
| three_wheeler | 16 | 2 | 6 | 4 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 26 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 19 | 14 | 7 | 57.6% (KTC 95%: 41–73%) | 73.1% (KTC 95%: 54–86%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 4/12 = 33.3% (KTC 95%: 14–61%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - extra: 2/3
  - missing: 0/3
  - wrong_class: 1/3
  - loose: 1/3
  - lọt: img_018.jpg (missing), img_025.jpg (wrong_class), img_005.jpg (loose), img_014.jpg (extra), img_035.jpg (missing), img_022.jpg (loose), img_028.jpg (wrong_class), img_024.jpg (missing)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
