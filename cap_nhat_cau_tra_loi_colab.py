# Dán toàn bộ nội dung file này vào một ô code mới ở CUỐI notebook Colab đã chạy.
# Giữ phiên runtime hiện tại: final_report dùng các biến và mask của lần chạy đó.
# Sau khi chạy, tải lại notebook còn nguyên output và submission.zip mới.

Q2 = """
Trong bảng của tôi, postprocess giảm từ 1,260 xuống 0,450 ms ở conf 0,25 (giảm 0,810 ms, khoảng 64,3%) và từ 1,320 xuống 0,410 ms ở conf 0,001 (giảm 0,910 ms, khoảng 68,9%); preprocess gần như không đổi, còn inference chênh lệch nhỏ và không cùng chiều ở hai mức conf. Lợi ích rõ nhất trong lần đo này là postprocess ở conf 0,001, nơi số box đầu ra tăng từ 5 lên khoảng 203–204; khi ngưỡng thấp, nhiều ứng viên cần được xử lý trước khi trả kết quả. Cảnh đông thường làm tăng số ứng viên và chi phí so sánh/lọc NMS, còn CPU/NPU có thể xử lý bước NMS và chuyển dữ liệu kém thuận lợi hơn phần mạng đã tối ưu. Tuy nhiên bảng này chỉ đo trên GPU T4, nên lợi ích CPU/NPU cần benchmark riêng; head one-to-one ở conf rất thấp vẫn có thể trả dự đoán dư hoặc sai.
"""

Q4 = """
Trong lần chạy của tôi, semantic tạo 8 vùng liên thông person dù ảnh có 4 người, trong khi Mask R-CNN trả đúng 4 instance person. Semantic có thể đếm thiếu khi hai người chạm nhau thành một vùng, hoặc đếm thừa khi một người bị chia thành nhiều mảnh và có các pixel nhiễu rời rạc; ở đây các vùng nhỏ chỉ 3–17 pixel cũng được tính. Panoptic segmentation gán class cho pixel và phân biệt ID từng instance thuộc nhóm things, nên không phải suy số người từ các vùng liên thông của class. Trong box xe buýt, 63% pixel bị gán train; đây là lỗi phân loại có thể liên quan sự giống nhau về hình dạng/phần kính và khác biệt miền dữ liệu giữa ảnh bus.jpg với Cityscapes, chứ không thể quy chắc cho một nguyên nhân chỉ từ ảnh này.
"""

Q6 = """
Điểm A cho mask gần toàn bộ người mặc áo sáng ở bên trái, diện tích 47.441 pixel và IoU 0,99 với mask prompt bằng box; điểm B cho một mảnh nhỏ trên vùng ống quần/chân, diện tích 2.714 pixel và IoU làm tròn 0,00. Một điểm không chỉ rõ ta muốn toàn bộ người hay một bộ phận, trong khi box cung cấp phạm vi đối tượng nên thường ít mơ hồ hơn, dù vẫn phụ thuộc detector và cần kiểm tra nhãn. Tôi sẽ dùng SAM trực tiếp khi cần phân đoạn linh hoạt theo prompt và đủ tài nguyên/độ trễ cho phép; với camera chạy liên tục cho các class cố định, tôi sẽ dùng SAM tạo nhãn, sửa các ca sai rồi train YOLO26-seg nhỏ. Lần này SAM tạo 5 mask từ 5 box trong 0,18 giây và polygon lưu lại có IoU 0,967–0,983 với mask gốc, cho thấy pipeline tạo nhãn đạt kiểm tra nhưng vẫn có sai khác do chuyển polygon.
"""

Q7 = """
Ở hình Keypoint R-CNN logit > -100, các điểm đầu gối và mắt cá vẫn được vẽ gần mép dưới hoặc ở những vị trí không hợp lý trong vùng ảnh, dù chân thật của hai người nằm ngoài khung; logit của các điểm chân người 0 chỉ khoảng -3,648 đến -0,969. Cần phân biệt bảng đang in x/y của YOLO, không phải tọa độ R-CNN: YOLO cũng trả đầu gối y=715/718 và mắt cá y=712/700 trên ảnh cao 720 pixel nhưng confidence chỉ 0,001–0,002. Argmax của heatmap vẫn chọn một vị trí cực đại khi không có tín hiệu tốt về keypoint, nên tọa độ được sinh ra không chứng minh điểm đó nhìn thấy và mỗi điểm cần độ tin cậy để lọc trước khi tính góc hoặc suy luận tư thế.
"""

Q8 = """
Trên đồ thị của tôi, đường người 40×80 cắt OKS 0,5 ở khoảng 7–8 pixel sai số Gauss sigma; tại 8 pixel OKS đã dưới 0,5. Đường người 250×500 vẫn khoảng 0,65 ở 30 pixel, nên đồ thị đang vẽ chỉ cho phép kết luận ngưỡng của người lớn trên 30 pixel, chưa xác định chính xác điểm cắt. Sigma của mắt nhỏ hơn hông vì vị trí mắt thường được gán nhãn chính xác hơn và ít biến thiên giữa người gán nhãn; với sai lệch cố định 8 pixel trên người 100×200, độ giống mắt chỉ 0,53 còn hông 0,97. Vì thế camera trên cao nhìn người cao khoảng 80 pixel sẽ rất nhạy với sai số vài pixel: cần bảo đảm độ phân giải/crop phù hợp, kiểm tra độ tin cậy và đánh giá riêng nhóm người nhỏ; trục đồ thị là sigma của nhiễu Gauss, không phải mọi keypoint đều lệch đúng số pixel đó.
"""

Q11 = """
Kiểu lỗi thứ nhất là dự đoán bàn chân trước co về giữa hai vị trí thật: ở Frame_54.jpg và Frame_51.jpg (đều OKS khoảng 0,63), hai dấu X bàn chân trước nằm gần nhau trong khi hai điểm nhãn tròn cách xa theo pha bước đi; right_front_paw cũng có sai số trung bình lớn nhất, khoảng 0,26 lần căn diện tích object. Tôi sẽ bổ sung và cân bằng ảnh các pha bước chân có hai chân trước tách xa, đồng thời thử crop hổ/độ phân giải cao hơn để kiểm tra xem chi tiết bàn chân có được cải thiện không. Kiểu lỗi thứ hai là định vị chân sau không đúng khi các chân chồng hoặc che nhau: ở Frame_31.jpg (OKS 0,63) và Frame_67.jpg (OKS 0,64), các dấu X bàn chân sau lệch rõ so với nhãn; left_hind_paw có sai số trung bình khoảng 0,23, cao hơn nhiều so với head khoảng 0,03. Tôi sẽ rà soát tính nhất quán nhãn trái/phải trong các ca che khuất, bổ sung góc nhìn hai phía và mức che khuất đa dạng; thống kê 4/53 ảnh cải thiện OKS khi hoán đổi trái/phải chỉ là dấu hiệu nghi nhầm danh tính, chưa đủ kết luận tất cả các ảnh trên đều bị đảo nhãn.
"""

# Notebook local đã cập nhật sáu câu vào đúng ô gốc; file này giúp cập nhật
# runtime Colab đang mở và tạo lại ZIP mà không phải train lại chỉ để đổi câu trả lời.
# Để kiểm tra tái lập trước khi nộp, chạy Run all bản notebook local đã cập nhật.
final_report(download=True)
