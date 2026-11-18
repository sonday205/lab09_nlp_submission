# phân tích lỗi

phần này dùng dự đoán của transformer trên 3.453 câu test. một thực thể được tính đúng khi khớp cả ranh giới và loại với nhãn gốc.

## false positive (769 trường hợp)

| text | gold | prediction | error type | explanation |
|---|---|---|---|---|
| soccer - japan get lucky win , china in surprise defeat . | china (per) | china (loc) | false positive | mô hình tìm đúng chữ china nhưng gán loc, trong khi nhãn gốc là per. china xuất hiện 91 lần dưới dạng thực thể trong tập train. |
| japan began the defence of their asian cup title with a lucky 2-1 win against syria in a group c championship match on friday . | — | c (misc) | false positive | mô hình gán misc cho chữ c trong group c. nhãn gốc bỏ qua chữ này và tập train cũng chưa có c dưới dạng thực thể. |
| the former soviet republic was playing in an asian cup finals tie for the first time . | — | republic (loc) | false positive | mô hình gán loc cho chữ republic dù nhãn gốc bỏ qua từ này. republic xuất hiện 5 lần dưới dạng thực thể trong tập train. |
| japan , co-hosts of the world cup in 2002 and ranked 20th in the world by fifa , are favourites to regain their title here . | — | co-hosts (misc) | false positive | mô hình gán misc cho co-hosts. từ này không được đánh dấu trong câu và cũng chưa xuất hiện dưới dạng thực thể trong tập train. |
| bitar saved well again from miura in the 37th minute , parrying away his header from a corner . | — | parrying (per) | false positive | mô hình gán per cho parrying. nhãn gốc không đánh dấu từ này và tập train cũng chưa có parrying dưới dạng thực thể. |

## false negative (680 trường hợp)

| text | gold | prediction | error type | explanation |
|---|---|---|---|---|
| soccer - japan get lucky win , china in surprise defeat . | china (per) | china (loc) | false negative | mô hình gán china là loc nên bỏ mất nhãn per của cùng đoạn chữ. tên này có 91 lần xuất hiện dưới dạng thực thể trong tập train. |
| defender hassan abbas rose to intercept a long ball into the area in the 84th minute but only managed to divert it into the top corner of bitar 's goal . | bitar (per) | — | false negative | mô hình bỏ sót tên bitar. tên này chưa xuất hiện dưới dạng thực thể trong tập train. |
| bitar pulled off fine saves whenever they did . | bitar (per) | — | false negative | bitar tiếp tục bị bỏ sót ở câu này. tập train không có ví dụ nào đánh dấu bitar là thực thể. |
| japan , co-hosts of the world cup in 2002 and ranked 20th in the world by fifa , are favourites to regain their title here . | fifa (org) | fifa (misc) | false negative | mô hình nhận ra fifa nhưng gán misc nên bỏ mất nhãn org. fifa xuất hiện một lần dưới dạng thực thể trong tập train. |
| rugby union - cuttitta back for italy after a year . | cuttitta (per) | — | false negative | mô hình bỏ sót tên cuttitta. tên này chưa xuất hiện dưới dạng thực thể trong tập train. |

## boundary error (558 trường hợp)

| text | gold | prediction | error type | explanation |
|---|---|---|---|---|
| despite winning the asian games title two years ago , uzbekistan are in the finals as outsiders . | asian games (misc) | asian (misc) | boundary error | mô hình chỉ lấy asian và bỏ mất games. asian xuất hiện 3 lần dưới dạng thực thể trong tập train. |
| takuya takagi scored the winner in the 88th minute , rising to head a hiroshige yanagimoto cross towards the syrian goal which goalkeeper salem bitar appeared to have covered but then allowed to slip into the net . | salem bitar (per) | salem (per) | boundary error | mô hình lấy phần tên salem và bỏ sót họ bitar. tập train không có salem dưới dạng thực thể. |
| cuttitta announced his retirement after the 1995 world cup , where he took issue with being dropped from the italy side that faced england in the pool stages . | 1995 world cup (misc) | world cup (misc) | boundary error | mô hình lấy world cup và bỏ mất năm 1995 nằm trong nhãn gốc. riêng world cup xuất hiện 28 lần dưới dạng thực thể trong tập train. |
| syria : 24 - salem bitar , 3 - bachar srour ; 4 - hassan abbas , 5 - tarek jabban , 6 - ammar awad ( 9 - louay taleb 69 ) , 8 - nihad al-boushi , 10 - mohammed afash , 12 - ali dib , 13 - abdul latif helou ( 17 - ammar rihawiy 46 ) , 14 - khaled zaher ; 16 - nader jokhadar . | salem bitar (per) | salem (loc) | boundary error | mô hình cắt mất bitar và còn gán salem là loc. tập train chưa có salem dưới dạng thực thể. |
| freestyle skiing-world cup mogul results . | skiing-world cup (misc) | cup mogul (misc) | boundary error | mô hình bắt đầu ở cup và kéo sang mogul nên lệch cả hai đầu. cụm cup mogul chưa xuất hiện dưới dạng thực thể trong tập train. |

## cùng tên, khác nhãn trong tập test

| surface | loại 1 và ngữ cảnh | loại 2 và ngữ cảnh |
|---|---|---|
| american | misc: american football conference | org: the conditions for clearance of the alliance were that british airways and american drop 168 slots at london heathrow airport , the busiest in europe . |
| arizona | loc: dallas at arizona | org: philadelphia , which fell from an nfc east tie with the dallas cowboys and washington redskins , go on the road against the new york jets and then entertain arizona . |
| atlanta | loc: on thursday , the 11th circuit u.s. court of appeals in atlanta denied his appeal in federal court . | org: atlanta 10 8 .556 7 |


trong tập test có 69 tên hoặc cụm từ được gán ít nhất hai loại nhãn. biểu diễn tĩnh dùng chung một vector cho cùng một từ. transformer cập nhật vector theo từng câu nên có thể dựa vào các từ xung quanh để phân biệt.

## đối chiếu dự đoán với kết quả

trên cùng 3.453 câu test, baseline đạt f1 0.716 và transformer hai lớp đạt 0.789. kết quả của lần chạy này phù hợp với dự đoán rằng transformer đạt f1 cao hơn baseline. checkpoint được chọn dựa trên tập validation.

f1 theo loại là 0.862 cho per, 0.704 cho org, 0.862 cho loc và 0.650 cho misc. tên tổ chức và các cụm thuộc misc có thể đa dạng về cách viết và ranh giới. trong tin tức, tên tổ chức cũng dễ lẫn với tên địa điểm hoặc tên đội. đây là những lý do có thể giải thích chênh lệch giữa các nhóm.

## kết quả theo độ dài và tần suất

thực thể một token đạt f1 0.781, hai token đạt 0.837, ba token đạt 0.653 và từ bốn token trở lên đạt 0.324. nhóm hai token có điểm cao nhất. hai nhóm dài hơn có điểm thấp hơn, trong đó nhóm từ bốn token chỉ có 62 thực thể gốc. nếu mô hình lấy sai ranh giới, thực thể gốc bị tính là fn và đoạn được dự đoán bị tính là fp.

có 2260 trong số 5648 thực thể gốc chứa ít nhất một từ bị tách thành nhiều subword. chỉ subword đầu tiên của mỗi từ được dùng để tính loss. các phần còn lại vẫn cung cấp ngữ cảnh, và việc một từ bị tách thành nhiều phần có thể khiến ranh giới khó nhận diện hơn.

nhóm tên xuất hiện từ 0 đến 1 lần dưới dạng thực thể trong tập train có recall 2012/2851, tức 0.706. nhóm xuất hiện từ 2 lần trở lên có recall 2528/2797, tức 0.904. kết quả này phù hợp với dự đoán tên hiếm dễ bị bỏ sót hơn, dù phép so sánh chưa kiểm soát độ dài và loại thực thể. mô hình có ít ví dụ để học các tên hiếm, nhưng vẫn có thể dựa vào ngữ cảnh và hình thức của từ. dữ liệu chủ yếu là tin tức và bài này chưa đo kết quả trên miền khác.
