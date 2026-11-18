# Thông tin học viên

- Họ và tên: Hà Khánh
- MSHV: 2A202603221
- Lớp (L2 / L3 / khác): K5-L23
- Tài khoản GitHub: Ngunhcho7654
- Repo bài nộp (URL): https://github.com/Ngunhcho7654/K5-L23-DAY08-2A202603221-HaKhanh-AIPrelabel
- Commit chốt bài (hash): nhánh main (sau commit 73edd60)
- Máy: Windows 11, RAM 16 GB, CPU (NVIDIA GTX 960M 2 GB VRAM, chạy torch CPU)
- CVAT (Docker): CVAT Docker chạy local tại http://localhost:8080

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 308 / 3 / 10 (96.0% / 0.9% / 3.1%, mẫu số: 321 box máy); added: 16 (4.9%, mẫu số: 327 đối tượng sau duyệt)
- So với gold: precision / recall của **máy**: 57.6% (19/33) / 73.1% (19/26) (mẫu số: 26 đối tượng gold, 33 box máy trên 4 ảnh gold)
- Bắt bẫy: 4/12 = 33.3%
- Thời gian: gán tay 3.75 phút/ảnh (15 phút / 4 ảnh); duyệt có pre-label 0.83 phút/ảnh (20 phút / 24 ảnh)
- Lỗi QC: 100.0% theo ảnh (5/5), 11.3% theo đối tượng (8/71)
- **Quyết định:** Rework — lý do một dòng: Bắt bẫy chỉ đạt 4/12 = 33.3% (< 80%), ảnh mẫu QC dính lỗi 5/5 = 100.0% (> 2%) và lặp lại lỗi box thừa/trùng ('thua' 5/8 lỗi).
