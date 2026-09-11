# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** September 11, 2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python 3.10, PyTorch 2.x, Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  - `class_id`: 468, `class_name`: "cab", `rank`: 1, `score`: 0.510915, `taxonomy_name`: "ImageNet-1K"
  - Hình ảnh là cảnh giao thông (ảnh được gắn nhãn là "traffic" trong COCO 210273).

- Record này mô tả toàn ảnh như thế nào?
  - Mô tả duy nhất một lớp chính cho toàn bộ ảnh. Model xếp hạng "cab" (tính từ hạng 1 đến hạng 5) dựa trên các đặc trưng bao quát của toàn ảnh. Đây là decision cấp ảnh, không phải per-object.

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  - Class list ImageNet-1K được định nghĩa bởi ImageNet dataset và taxonomy. Checkpoint `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K, vì vậy nó chỉ có thể dự đoán 1000 lớp từ taxonomy này.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  - `class_id` là con số định danh duy nhất (ví dụ 468 = "cab"). `class_name` là tên người đọc được. `taxonomy_name` chỉ rõ nguồn định nghĩa (ImageNet-1K có thể khác COCO-80). Khi training/inference mô hình mới với taxonomy khác, class_id sẽ không match được.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  - Guideline phải quy định chiến lược:
    - Chọn chủ thể chiếm diện tích lớn nhất?
    - Chọn chủ thể nằm ở vị trí ưu tiên (trung tâm)?
    - Liệt kê tất cả các lớp nếu có? (nhưng classification cấp ảnh thường yêu cầu 1 nhãn duy nhất)
  - Ảnh traffic này chứa xe buýt, ô tô, nhưng model tìm được "cab" ở hạng 1 và "minibus/police_van" ở hạng 2–3. Annotator cần biết: có chọn lớp được predict hay phải gán nhãn thủ công theo chủ thể thực tế?

- Vì sao model score không phải ground truth?
  - Score 0.510915 cho "cab" chỉ là mức độ tự tin của mô hình, không phải xác suất đúng sai. Model được huấn luyện trên ImageNet, không phải trên ảnh giao thông. Ảnh này có xe buýt/ô tô (từ COCO), nên prediction của model có thể sai. Annotator/reviewer phải gán nhãn ground truth dựa trên định nghĩa guideline, không phải dựa vào score cao của model.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `traffic`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  - Example: `class_name`: "person", `score`: 0.769676, `bbox_xyxy`: [0.08, 256.79, 18.39, 313.12], `bbox_width`: 18.32, `bbox_height`: 56.33
  - Hộp này bắt đầu từ pixel (0.08, 256.79) ở góc trên trái và kết thúc tại (18.39, 313.12) ở góc dưới phải.

- Diễn giải vị trí box bằng lời:
  - Người ở phía cực trái ảnh, chiếm diện tích khoảng 18×56 pixel (hẹp và cao). Box bao quanh phần thân người. Người này có thể bị cắt mép ảnh vì nằm ở biên trái.

- So sánh số prediction ở hai threshold:
  - Ở `score_threshold=0.35`: JSON hiển thị ít nhất 2 detections (bus, car). Nếu giảm threshold xuống 0.25 hoặc 0.15, mô hình sẽ detect thêm những object có confidence thấp hơn (ví dụ các xe nhỏ hơn, che khuất). Ngược lại, nếu tăng threshold lên 0.5–0.6, chỉ còn những detection rất tự tin (mất detection của vật thể không rõ).

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  - Threshold thấp → detection nhiều → độ bao phủ cao nhưng reviewer phải xem nhiều false positive hơn → khối lượng QC tăng.
  - Threshold cao → detection ít → độ bao phủ giảm (miss object) nhưng reviewer chỉ xem những detection tự tin → khối lượng QC giảm, nhưng có risk bỏ sót object thực.

- Đề xuất một quy tắc box chặt:
  - "Box phải bao quanh phần nhìn thấy được của object, không thêm khoảng trống lớn. Nếu object bị che khuất 50%, vẫn box phần nhìn thấy được. Nếu object bị che khuất >70%, escalation quyết định có nên gán nhãn hay không."

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  - Che khuất từng phần (20–50% không nhìn thấy): annotator vẫn box phần nhìn thấy, ghi tag "partially_occluded".
  - Che khuất nặng (>70%): escalation → reviewer quyết định có lưu nhãn hay xóa.
  - Cắt mép ảnh (object bị cắt ngoài khung): guideline phải quy định rõ—box phần còn lại hay không box? (ví dụ: nếu <20% ngoài khung thì box; nếu >20% thì skip hoặc escalation).

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `traffic`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  - `instance_id`: "traffic-001", `class_name`: "bus", `score`: 0.925745, `polygon_point_count`: 120
  - Polygon gồm 120 điểm pixel theo thứ tự, ví dụ: [[148, 189], [147, 190], [145, 190], ...]. Mỗi điểm là tọa độ (x, y) trên cạnh của object.

- Polygon bổ sung chi tiết gì so với box?
  - Box là hình chữ nhật `bbox_xyxy: [95.3, 188.72, 224.08, 319.96]` bao quanh object, bao gồm khoảng trống xung quanh.
  - Polygon (120 điểm) follow biên tuyệt thực của object (thân xe, cửa sổ, bánh xe, v.v.), chi tiết hơn và loại bỏ background khoảng trống.
  - Ví dụ: xe buýt có hình dạng không phải hình chữ nhật (có bạt che nóc, khía cạnh), polygon sẽ follow đường viền chính xác, trong khi box vẫn là hình chữ nhật bao quanh toàn bộ.

- `instance_id` dùng để làm gì và không phải loại ID nào?
  - `instance_id`: "traffic-001" dùng để **đồng nhất instance cụ thể trong bộ output của bài lab**. Nó không phải `class_id` (class_id=5 là "bus"). Cũng không phải tracking_id (theo dõi object qua các frame video).
  - Trong một ảnh tĩnh, nếu có 2 xe buýt, sẽ có "traffic-001" và "traffic-002" để phân biệt 2 instance của cùng lớp "bus".

- Đề xuất một quy tắc biên mask:
  - "Polygon phải follow biên của object cẩn thận, không bao gồm shadow/reflection. Độ chính xác tối thiểu: nếu object rộng >100 pixel, sai số biên <5 pixel là chấp nhận được. Nếu object <50 pixel, sai số <2 pixel. Nếu có phần che khuất 50%, chỉ polygon phần nhìn thấy được rồi ghi tag 'partial'."

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  - Mờ (blur) phần 20–50% object: annotator vẫn polygon phần nhìn thấy, ghi tag "blurred_boundary".
  - Tiếp xúc với object khác (ví dụ: xe buýt chạm ô tô): polygon của mỗi object riêng, tất nhiên là chạm nhau ở cạnh. Không merge 2 polygon lại.
  - Che khuất >70%: escalation → reviewer quyết định polygon phần nhìn thấy hay skip instance này.
  - Vùng shadow/reflection lớn: guideline phải nêu rõ—có bao gồm shadow vào polygon hay chỉ object chính? (thường: không bao gồm shadow, chỉ object thực).

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp duy nhất (class_name, class_id) cho toàn ảnh | Ảnh multi-chủ-thể (giao thông có xe buýt, ô tô, xe máy); model predict "cab" nhưng ảnh chứa chủ yếu bus. Guideline chưa rõ: chọn object nào làm nhãn chính? | Annotator đọc guideline, chọn chủ thể ưu tiên (ví dụ: object lớn nhất hoặc ở center), gán class_name/class_id. | Reviewer kiểm tra: nhãn có match guideline không? Có miss class khác không? Confidence của model có cao không (có thể là dấu hiệu ảnh khó)? |
| Phát hiện vật thể | Danh sách hộp (bbox_xyxy, class_name, class_id) cho mỗi object rõ rệt | Xe nhỏ bị che khuất, ở xa (score thấp 0.5–0.6); xe bạt che (khó box chính xác). Model threshold 0.35 có thể bỏ sót hoặc false positive. | Annotator kiểm tra từng object: nếu score >0.7 và object rõ ràng, đồng ý; nếu score <0.5 hoặc object mơ hồ, escalate. Box phần nhìn thấy được. | Reviewer spot-check: box có khít object không? Có miss object rõ ràng không? Có false positive không (background được box nhầm)? |
| Instance segmentation | Danh sách polygon (polygon_xy, class_name, instance_id) cho mỗi object | Biên xe bạt (mờ), tiếp xúc xe khác, shadow lớn. Model polygon có 120 điểm, nhưng biên có đủ sắc nét không? | Annotator kiểm tra mỗi polygon: nếu boundary rõ ràng, tinh chỉnh polygon theo guideline; nếu mơ hồ (blur >50%), escalate. Ghi tag "blurred" nếu cần. | Reviewer kiểm tra: polygon follow biên chính xác không (độ sai <5 px)? Có miss instance không? Có overlap sai (2 polygon cross nhau không phải tiếp xúc bình thường)? |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  - Không tải ảnh chứa khuôn mặt người, biển số xe, hoặc dữ liệu nhân viên/khách hàng lên Colab/GitHub công khai.
  - Ba ảnh mẫu được từ COCO public dataset, có metadata CC BY 2.0, không chứa thông tin nhạy cảm.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  - Lab Coach (Slack hoặc email chỉ định).
  - Ghi nhận loại lỗi trong section "Thay đổi so với notebook" của báo cáo.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
