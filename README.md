# LAB 09

## AI-assisted code

### What was generated

AI hỗ trợ tạo mã và notebook cho baseline Logistic Regression, tokenization và label alignment, fine-tuning Transformer, tính metric và pipeline Information Extraction (IE). Các bảng số liệu, biểu đồ và checkpoint được tạo khi chạy mã thí nghiệm.

### What was modified

Trong quá trình thực hiện, mã hỗ trợ được điều chỉnh để dùng cùng tập test CoNLL-2003 cho hai mô hình, áp dụng nhãn `-100` khi huấn luyện và đánh giá NER, và chấm IE theo loại thực thể và vị trí ký tự. 

### What was verified

Đã kiểm tra cú pháp hai notebook và output đã lưu, đối chiếu điểm entity-level của hai mô hình trên cùng 3.453 câu test và kiểm tra kết quả IE trên 30 câu lấy gold từ CoNLL-2003. Việc chạy lại từ riêng thư mục nộp hiện cần bổ sung các phụ thuộc của mã và notebook.
