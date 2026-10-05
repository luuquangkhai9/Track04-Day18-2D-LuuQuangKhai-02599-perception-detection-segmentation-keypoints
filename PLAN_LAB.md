# Kế hoạch hoàn thành Lab Ngày 18

- Họ tên: **Lưu Quang Khải**
- Mã sinh viên: **2A202602599**
- Bài: 2D Perception — Detection, Segmentation, Keypoints.
- Notebook: `lab_2d_perception_student.ipynb`.
- Mục tiêu: hoàn thành toàn bộ yêu cầu 100 điểm lõi; làm bonus sau khi phần bắt buộc đã đạt.

## Cách phối hợp

Khải chạy notebook trên Colab có T4 GPU, lưu bản trong Drive và gửi output hoặc lỗi sau mỗi mốc. Mình hỗ trợ điền TODO, giải thích thuật toán, sửa lỗi và cùng Khải viết Q1–Q12 dựa trên kết quả thực tế. Không điền số đo, mAP hoặc nhận xét ảnh khi chưa có output.

Sau mỗi lần sửa code, chạy lại chính ô đó rồi chạy tiếp các ô phụ thuộc. Chỉ qua mốc tiếp theo khi các kiểm tra bắt buộc đã đạt. Nếu dùng `lifeline=True` để đi tiếp, ghi lại hàm cần quay về hoàn thiện vì hàm đó chưa được tính điểm.

Thời gian dưới đây theo lộ trình 120 phút của lab; tải dữ liệu, lỗi môi trường và chạy lại toàn bộ có thể cần thêm thời gian.

## Mốc 0 — Chuẩn bị và setup (~10 phút)

- [ ] Khải mở notebook qua nút Colab trong README, lưu bản riêng trong Drive.
- [ ] Thêm ô Markdown đầu notebook ghi họ tên và mã sinh viên như trên.
- [x] Mở notebook trên Colab và chọn T4 GPU (Khải đã xác nhận).
- [ ] Chạy lần lượt 5 ô Phần 0; giữ phiên bản thư viện do notebook cài.
- [ ] Xác nhận `device = cuda`, thông báo sẵn sàng, 9 dấu kiểm và hai ảnh mẫu.

**Khải gửi:** output kiểm tra GPU và thông báo setup cuối, hoặc toàn bộ traceback nếu lỗi.

**Mình hỗ trợ:** kiểm tra môi trường; xử lý lỗi cài đặt hoặc tải weights/dataset trước khi đi tiếp.

## Mốc 1 — Detection (~35 phút)

- [ ] Chạy 1A, quan sát shape output Faster R-CNN và YOLO26.
- [ ] Cùng viết Q1 về số dự đoán và quy ước tọa độ box.
- [ ] Hoàn thiện `box_iou`, `nms`, `batched_nms`; chạy từng hàm kiểm tra và `gate("1B")`.
- [ ] Chạy 1C: so sánh box trước/sau NMS và head one-to-one.
- [ ] Có bảng latency đủ 4 cấu hình: hai head × confidence 0.25/0.001.
- [ ] Cùng viết Q2 theo bảng latency thực tế và Q3 về đánh đổi của NMS.

**Khải gửi:** kết quả kiểm tra 3 hàm, số box và bảng latency đầy đủ.

**Điều kiện xong:** cả 3 hàm đạt; bảng 4 dòng có output; Q1–Q3 đã điền.

## Mốc 2 — Segmentation (~30 phút)

- [ ] Chạy 2A, quan sát semantic và instance segmentation; viết Q4.
- [ ] Hoàn thiện `mask_iou`, `polygon_to_mask`, `mask_to_yolo_seg` và qua các kiểm tra.
- [ ] Xem bảng ghép cặp mask, viết Q5.
- [ ] Chạy 2C: prompt SAM bằng điểm và box, tạo `autolabel/bus.txt`.
- [ ] Kiểm tra định dạng nhãn, tọa độ trong [0, 1] và hình polygon vẽ lại; viết Q6.

**Khải gửi:** các dòng kiểm tra, kết quả hai prompt SAM, hình polygon và thông báo file nhãn hợp lệ.

**Điều kiện xong:** 3 hàm đạt, file nhãn hợp lệ, Q4–Q6 đã điền.

## Mốc 3 — Keypoints (~25 phút)

- [ ] Chạy 3A; đổi `KP_THR = -100`, chạy lại ô vẽ và quan sát; viết Q7.
- [ ] Hoàn thiện `oks`, qua kiểm tra và chạy thí nghiệm; viết Q8 theo đồ thị thực tế.
- [ ] Hoàn thiện `joint_angle`, qua kiểm tra và chạy luật phát hiện ngã.
- [ ] Viết Q9 về báo nhầm, bỏ sót, ảnh xoay và xử lý qua nhiều khung hình.

**Khải gửi:** kiểm tra OKS/góc khớp, đồ thị OKS, số người và góc thân trên ảnh gốc/ảnh xoay.

**Điều kiện xong:** 2 hàm đạt, các thí nghiệm có output, Q7–Q9 đã điền.

## Mốc 4 — Fine-tune và phân tích lỗi (~20 phút theo lab)

- [ ] Chạy ô giải phóng bộ nhớ và in tên 12 keypoint tiger-pose.
- [ ] Kiểm tra thống kê hướng quay train/val.
- [ ] Sửa `FLIP_IDX` theo từng cặp tên trái/phải; qua `check_flip_idx()` và `gate("4A")`.
- [ ] Chạy train trên GPU: **40 epoch, imgsz 640**. Nếu hết bộ nhớ, giảm batch 16 xuống 8.
- [ ] Viết Q10 trong khi chờ train.
- [ ] Chạy đánh giá để có bảng Box/Pose mAP.
- [ ] Chạy phân tích lỗi để có 6 ảnh val tệ nhất và biểu đồ sai số keypoint.
- [ ] Cùng viết Q11: nêu hai kiểu lỗi, dẫn ảnh cụ thể và cách khắc phục từng kiểu.
- [ ] Viết Q12 về sigma cho dataset custom.

**Khải gửi:** kết quả FLIP_IDX, bảng mAP, 6 ảnh val tệ nhất và biểu đồ sai số.

**Điều kiện xong:** train đủ cấu hình yêu cầu, có bảng đánh giá và ảnh phân tích lỗi; Q10–Q12 đã điền. Mức mAP tham khảo trong README không được thay cho số đo thực tế.

## Mốc 5 — Kiểm tra tái lập và chuẩn bị nộp

- [ ] Rà soát mọi TODO bắt buộc, Q1–Q12 và các hàm từng dùng phao.
- [ ] Lưu notebook vào Drive trước khi khởi động lại phiên.
- [ ] Chọn `Restart session and run all`; chờ chạy hết không lỗi, bao gồm train lại.
- [ ] Chạy `final_report()`; mọi mục bắt buộc phải ✅.
- [ ] Tải notebook còn nguyên output và `submission.zip`, giải nén.
- [ ] Kiểm tra có `submission/ket_qua.json` và `submission/autolabel/bus.txt`.
- [ ] Đưa notebook và thư mục submission lên repo GitHub public của Khải:
  `<username>/Track04-Day18-2D-perception-detection-segmentation-keypoints`.
- [ ] Nộp URL repo vào LMS Ngày 18, **không mở PR**; giữ repo public đến khi có điểm.

**Khải gửi:** checklist cuối và thông tin repo khi đến bước nộp. Username GitHub sẽ xác định ở mốc này.

## Bonus — Chỉ sau khi phần bắt buộc hoàn tất

1. Mục 1D: hoàn thiện `average_precision`, kiểm tra và vẽ đường PR (+5 điểm).
2. Mục 4C: train thêm model flip_idx đồng nhất, so hai model trên val gốc và val lật gương, giải thích metric che lỗi (+10 điểm).
3. Chọn bài bổ sung theo rubric: auto-label + fine-tune segmentation hoặc export ONNX + đo latency, lưu báo cáo trong submission (+5 điểm).

## Trạng thái ban đầu

Đã đọc README, rubric và notebook. Khải đã mở Colab và chọn T4 GPU. Notebook trong workspace đã có thông tin sinh viên, toàn bộ hàm TODO (gồm AP bonus), FLIP_IDX theo tên và hình Q7 ở cả hai ngưỡng. Đã điền bản trả lời lý thuyết Q1, Q3, Q5, Q9, Q10, Q12; Khải cần đọc lại. Q2, Q4, Q6, Q7, Q8, Q11 vẫn giữ placeholder để hoàn thiện theo output thực tế. Chưa có kết quả suy luận model hoặc train trên Colab. Bước tiếp theo: kiểm tra các hàm, rồi Khải upload notebook mới và Run all trên T4 GPU.

### Kiểm tra code đã hoàn thành

Chạy `verify_lab.py` trong môi trường CPU local với Ultralytics 8.4.171: notebook hợp lệ theo nbformat, mọi ô Python qua kiểm tra cú pháp, không còn TODO code chưa điền. Cả 10 mục code qua 38 phép thử gốc và tất cả gate 1B/2B/3B/3C/4A đều đạt, không dùng phao.

| Mục | Số phép thử đạt |
|---|---:|
| box_iou | 5 |
| nms | 6 |
| batched_nms | 4 |
| average_precision | 4 |
| mask_iou | 4 |
| polygon_to_mask | 2 |
| mask_to_yolo_seg | 4 |
| oks | 3 |
| joint_angle | 2 |
| FLIP_IDX | 4 |

FLIP_IDX xác nhận từ tên keypoint trong YAML: `[0, 1, 2, 3, 7, 6, 5, 4, 10, 11, 8, 9]`.

Ở thời điểm kiểm tra code local, chưa chạy suy luận model, bảng latency, SAM auto-label hoặc train GPU. Output kiểm tra local không được ghi vào notebook để tránh nhầm với kết quả Colab.

### Cập nhật sau khi Khải chạy Colab

- Đã đọc output notebook được Khải cập nhật vào workspace; không có output lỗi.
- Train hoàn thành 40 epoch, imgsz 640 trên Tesla T4 trong khoảng 4,4 phút.
- Box mAP50–95: 0,930; Pose mAP50: 0,995; Pose mAP50–95: 0,457.
- Có bảng latency đủ 4 cấu hình; NMS tự viết khớp thư viện.
- Auto-label 5 object hợp lệ; polygon round-trip IoU 0,967–0,983.
- OKS trung bình val 0,745; bỏ sót 0/53; nghi đảo trái/phải 4/53 (chỉ là chỉ báo).
- Đã quan sát các hình SAM, keypoint, đồ thị OKS, 6 ảnh lỗi và biểu đồ sai số để hoàn thiện Q2/Q4/Q6/Q7/Q8/Q11. Bản notebook local đã có đủ 12 câu trả lời.
- Đã bỏ trường metadata không hợp lệ khỏi 2 output stream; giữ nguyên log và ảnh. Định dạng notebook và 38 phép thử code đều đạt lại.
- Xóa riêng output checklist cuối cũ (ghi 6/12 câu), vì nó chưa phản ánh câu trả lời đã cập nhật; không tự tạo output Colab mới.

Bước còn lại: cập nhật câu trả lời vào runtime Colab bằng `cap_nhat_cau_tra_loi_colab.py`, chạy final_report để tạo ZIP mới. Trước khi nộp, Restart & Run All bản notebook đã cập nhật, tải notebook/ZIP còn nguyên output và đưa lên GitHub public. Chưa xác nhận submission ZIP mới hoặc URL repo GitHub.
