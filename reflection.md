# reflection

cùng một pretrained transformer có thể giải cả sequence classification lẫn token classification vì phần backbone không quyết định trực tiếp dạng nhãn đầu ra. sau khi được pretrain, nó nhận chuỗi token và tạo một vector ẩn cho từng vị trí. mỗi vector phụ thuộc vào các từ xung quanh thông qua self-attention, nên cùng một từ có thể mang biểu diễn khác nhau trong hai câu. prediction head lấy các vector này và biến chúng thành logits cho nhiệm vụ cụ thể.

với sequence classification, ta cần một nhãn cho toàn câu. ở các model kiểu bert, vector tại vị trí [cls] là đầu vào thường dùng cho một lớp tuyến tính có số chiều bằng số lớp. ví dụ, phân loại cảm xúc hai lớp tạo hai logits từ vector [cls], rồi chọn lớp có điểm cao hơn. [cls] không tự động chứa mọi thông tin một cách hoàn hảo; fine-tuning khiến backbone và head cùng điều chỉnh để thông tin hữu ích cho nhãn câu tập trung vào vị trí này. một số kiến trúc có thể dùng pooling thay cho [cls] mà vẫn theo nguyên tắc một vector đại diện cho chuỗi.

với token classification, đầu ra cần một nhãn ở mỗi vị trí. ta áp dụng cùng một lớp tuyến tính lên từng token representation: hidden states có shape (b, t, d) trở thành logits (b, t, c). đối với conll-2003, c=9 gồm o và các nhãn bio của per, org, loc, misc. nhờ ngữ cảnh, washington trong một câu có thể mang nghĩa địa điểm, trong câu khác có thể là tên người hoặc tổ chức. tuy nhiên, head độc lập trên từng vị trí không tự bảo đảm một chuỗi bio hợp lệ; dữ liệu huấn luyện và bước giải mã vẫn quan trọng.

ner gán nhãn ở mức word, còn tokenizer của transformer tạo subword. vì vậy, trước khi tính loss phải nối lại nhãn với các vị trí token. bài này dùng nhãn ở subword đầu tiên và đặt -100 cho các mảnh còn lại, special tokens và padding. các vị trí -100 vẫn có thể tham gia attention nhưng không được tính vào loss. khi suy luận và đánh giá, ta lấy đúng một prediction cho mỗi word. sau đó gom nhãn bio thành entity spans và chấm đúng khi cả loại lẫn ranh giới khớp gold. cách đánh giá này phản ánh mục tiêu trích xuất thông tin tốt hơn token accuracy: dự đoán đúng new nhưng bỏ york vẫn làm sai thực thể new york.

như vậy, backbone pretrained cung cấp biểu diễn có ngữ cảnh dùng chung; prediction head, định dạng labels và hàm loss quyết định đầu ra là một nhãn cho câu hay một chuỗi nhãn cho token. fine-tuning cập nhật cả head lẫn backbone để thích nghi với nhiệm vụ.

## individual learning check

1. sequence vs token classification: loại đầu dự đoán một nhãn cho cả chuỗi; loại sau dự đoán một nhãn ở mỗi vị trí.
2. bio: mã hóa loại và ranh giới của entity bằng b-x, i-x, o.
3. b-per vs i-per: b-per bắt đầu tên người; i-per tiếp tục tên người đang mở.
4. vì sao cần alignment? gold thường gắn theo word nhưng tokenizer tạo nhiều subword cho một word.
5. ignore trong loss: vị trí -100 không góp vào loss; token vẫn được xử lý bởi transformer.
6. vì sao token accuracy chưa đủ? o thường nhiều và một entity sai ranh giới có thể chứa vài token dự đoán đúng.
7. entity f1 vs token f1: entity f1 đếm span và loại khớp hoàn toàn; token f1 chấm từng vị trí riêng.
8. vì sao boundary khó? tên có thể dài, chứa giới từ và đứng sát các từ hoặc entity khác.
9. contextual representation: cùng surface form được biểu diễn khác nhau theo từ xung quanh, giúp giải mơ hồ.
10. ner trong ie: gom bio thành tên người, tổ chức, địa điểm rồi xuất json hoặc cấu trúc khác.
