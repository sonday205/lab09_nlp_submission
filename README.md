# LAB 09

## Cài đặt

Môi trường đã kiểm tra dùng Python 3.13. Lần chạy đầu cần Internet để tải CoNLL-2003 và checkpoint BERT pretrained từ Hugging Face. Script fine-tuning tự chọn CUDA nếu PyTorch nhận GPU, hoặc dùng CPU.

```bash
git clone https://github.com/sonday205/lab09_nlp_submission.git
cd lab09_nlp_submission
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Chạy thí nghiệm

```bash
python baseline_ner.py
python train_ner.py
python information_extraction.py --text 'Barack Obama visited Paris.'
python information_extraction.py
```

Baseline ghi `baseline_metrics.json` và `baseline_predictions.json`. Script fine-tuning chạy 10 epoch, chọn checkpoint theo F1 trên tập validation, rồi ghi kết quả test vào `transformer_metrics.json` và `transformer_predictions.json`. Khi chạy IE không có `--text`, script đánh giá 30 câu test với gold từ CoNLL-2003 và ghi `ie_metrics.json` cùng `ie_examples.json`. Thư mục `checkpoint/` được tạo tại máy khi fine-tuning.

Có thể mở hai notebook bằng kernel của `.venv` trong VS Code. Để chạy từ terminal và lưu notebook kết quả riêng, dùng các lệnh sau sau khi fine-tuning. Notebook fine-tuning cũng tự huấn luyện nếu chưa có checkpoint.

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output ner_finetuning.executed.ipynb ner_finetuning.ipynb
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output tokenizer_alignment.executed.ipynb tokenizer_alignment.ipynb
```

`results.csv`, `entity_metrics.csv` và các biểu đồ là kết quả của lần chạy đã nộp. Chỉ số tính lại là kết quả của checkpoint đang dùng.

## AI-assisted code

### What was generated

AI hỗ trợ tạo mã và notebook cho baseline Logistic Regression, tokenization và label alignment, fine-tuning Transformer, tính metric và pipeline Information Extraction (IE). Các bảng số liệu, biểu đồ và checkpoint được tạo khi chạy mã thí nghiệm.

### What was modified

Trong quá trình thực hiện, mã hỗ trợ được điều chỉnh để dùng cùng tập test CoNLL-2003 cho hai mô hình, áp dụng nhãn `-100` khi huấn luyện và đánh giá NER, và chấm IE theo loại thực thể và vị trí ký tự. Gói nộp bổ sung hai mô-đun dùng chung, danh sách thư viện và hướng dẫn chạy. Notebook fine-tuning dùng trực tiếp biểu đồ đã có trong gói nộp.

### What was verified

Đã kiểm tra cú pháp hai notebook và output đã lưu, đối chiếu điểm entity-level của hai mô hình trên cùng 3.453 câu test và kiểm tra kết quả IE trên 30 câu lấy gold từ CoNLL-2003.

Sau khi bổ sung dependencies, đã chạy baseline, fine-tuning thử 1 epoch, IE và cả hai notebook từ bản sao riêng của gói nộp với dữ liệu và mô hình pretrained trong cache Hugging Face.
