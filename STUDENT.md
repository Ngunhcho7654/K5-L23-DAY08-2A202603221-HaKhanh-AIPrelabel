# Thông tin học viên

- Họ và tên: Hà Khánh
- MSHV: 2A202603221
- Lớp (L2 / L3 / khác): K5-L23
- Tài khoản GitHub: Ngunhcho7654
- Repo bài nộp (URL): https://github.com/Ngunhcho7654/K5-L23-DAY08-2A202603221-HaKhanh-AIPrelabel
- Commit chốt bài (hash): nhánh main (cập nhật hoàn tất sau vòng Rework)
- Máy: Windows 11, RAM 16 GB, CPU (NVIDIA GTX 960M 2 GB VRAM, chạy torch CPU)
- CVAT (Docker): CVAT Docker chạy local tại http://localhost:8080

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 296 / 7 / 18 (92.2% / 2.2% / 5.6%, mẫu số: 321 box máy); added: 23 (7.1%, mẫu số: 326 đối tượng sau duyệt)
- So với gold: precision / recall của **máy**: 57.6% (19/33) / 73.1% (19/26) (mẫu số: 26 đối tượng gold, 33 box máy trên 4 ảnh gold)
- Bắt bẫy: 12/12 = 100.0% (vòng 1 đạt 4/12 = 33.3% -> Rework -> vòng 2 đạt 12/12)
- Thời gian: gán tay 3.75 phút/ảnh (15 phút / 4 ảnh); duyệt có pre-label 1.46 phút/ảnh (35 phút / 24 ảnh gồm cả vòng Rework)
- Lỗi QC: 0.0% theo ảnh (0/5), 0.0% theo đối tượng (0/67)
- **Quyết định:** Release — lý do một dòng: Sau 1 vòng Rework theo guideline v1.1, bắt bẫy đạt 12/12 = 100.0% (>= 80%), lỗi QC mẫu đạt 0/5 = 0.0% (<= 2%) và hết lỗi hệ thống lặp lại (dù lưu ý cận trên 95% Wilson là 43% do mẫu nhỏ 5 ảnh).
