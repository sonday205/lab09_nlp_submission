# Prediction

1. Pretrained Transformer so với baseline: Transformer pretrained cỡ đầy đủ sẽ có F1 cao hơn classifier dựa trên đặc trưng từ nhờ biểu diễn theo ngữ cảnh và khả năng chia sẻ thông tin giữa các vị trí. Tuy nhiên, checkpoint rất nhỏ hoặc ngân sách fine-tuning thấp có thể cho kết quả thấp hơn baseline; đó là giới hạn của cấu hình thí nghiệm, không bác bỏ lợi ích chung của pretraining.

2. Entity nhiều token: F1 exact-span giảm khi số token tăng. Mỗi vị trí và ranh giới đều phải đúng; entity dài tạo thêm cơ hội sai nhãn hoặc tách nhầm.

3. Entity hiếm: surface form ít gặp trong train thường có recall thấp hơn. Baseline dễ phụ thuộc vào từ đã thấy; Transformer có thể dùng hình thái và ngữ cảnh để tổng quát hóa nhưng tên riêng hiếm hoặc đa nghĩa vẫn khó.
