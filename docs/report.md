# Báo cáo tiền xử lý dữ liệu

Dự án: sinh từ tiếp theo dựa trên văn bản tiếng Việt thô (next-token generation).
Phần việc: tiền xử lý dữ liệu.

## Mô tả bộ dữ liệu

Nguồn: `VTSNLP/vietnamese_curated_dataset` trên Hugging Face, tải theo chế độ
streaming (không tải toàn bộ vào RAM). Bộ dữ liệu gồm văn bản tiếng Việt đã
tuyển chọn từ nhiều lĩnh vực (tin tức, web, sách, wiki), phục vụ huấn luyện
mô hình ngôn ngữ.

## Các trường dữ liệu

- `id`: định danh văn bản.
- `text`: nội dung văn bản thô.
- `domain`: lĩnh vực / nguồn của văn bản.

## Phương pháp lấy mẫu

Dùng bộ lấy mẫu hồ chứa (reservoir sampling) xác định theo seed: mọi quyết
định ngẫu nhiên đều suy ra từ SHA-256, nên kết quả lặp lại được giữa các lần
chạy. Không chọn N văn bản đầu tiên. Mỗi chế độ có giới hạn quét nguồn riêng:
smoke 50.000 văn bản (~2.000 chấp nhận), development 500.000 (~20.000),
final không giới hạn (~50.000).

Hạn chế (đã đo): mẫu development được lấy từ 500.000 văn bản đầu tiên của
luồng; nó đại diện cho *phần đã quét*, không phải toàn bộ bộ dữ liệu. Bản
thân văn bản đã quét có phân bố domain gần như bảo toàn sau lấy mẫu (xem
dưới), nhưng không có gì đảm bảo cho phần chưa quét.

## Quy tắc làm sạch

Giữ nguyên: dấu câu, chữ hoa, số, cấu trúc đoạn, dấu tiếng Việt (NFC).
Chuẩn hóa: kết thúc dòng về LF, khoảng trắng lặp về một khoảng trắng, tối đa
một dòng trống liên tiếp, cắt khoảng trắng đầu/cuối, bỏ ký tự điều khiển.
Loại bỏ kèm lý do: văn bản rỗng (`empty_text`), không có ký tự chữ/số có nghĩa
(`invalid_text`), quá ngắn dưới 20 ký tự (`too_short`), quá dài vượt ngưỡng
tùy chọn (`too_long`), trùng `id` (`duplicate_id`), trùng văn bản chính xác
sau chuẩn hóa (`duplicate_text`, SHA-256).

Lý do giữ cách làm sạch thận trọng (đã kiểm chứng trên mẫu 20.000 văn bản):
dấu câu (trung vị 2.7%), số (trung vị 1.4%) và ký tự tiếng Việt hóa (trung vị
20%) xuất hiện phổ biến và hữu ích cho mô hình ngôn ngữ. Việc hạ chữ thường,
bỏ dấu hay bỏ dấu câu sẽ phá vỡ chính tả tiếng Việt và làm mất thông tin ngữ
nghĩa cho đầu việc của các bạn làm tokenizer/mô hình.

## Khử trùng lặp

- Trùng `id`: loại bỏ, đếm `duplicate_id`.
- Trùng văn bản chính xác sau chuẩn hóa: SHA-256, đếm `duplicate_text`.
- Hai backend (`--dedup-backend memory|sqlite`) cho quyết định giống hệt nhau;
  xem mục "Khử trùng lặp quy mô lớn" dưới đây.
- Kết quả đo trên 500.000 văn bản quét: **5 văn bản trùng chính xác**
  (0.001%), không có trùng `id`. Bộ dữ liệu tuyển chọn rất sạch về trùng lặp.
- Khử trùng diễn ra trước khi chia tập, đảm bảo không có `id` hay văn bản nào
  xuất hiện ở hai tập.

## Kiểm định chất lượng thực nghiệm (mới)

Chạy bộ kiểm định không phá hủy (audit) trên mẫu development 20.000 văn bản:
`python -m src.data.audit --output-dir data/processed-dev --audit-dir data/audit-dev --seed 42 --near-duplicates`.

### Các vấn đề chất lượng mà bộ lọc cũ bỏ sót

- **Văn bản không phải tiếng Việt: 17 văn bản (0.09%)** có tỷ lệ ký tự tiếng
  Việt < 0.001: tiếng Anh (tin tức, rao bán), Armenia, Ả Rập (mojibake kép),
  Ewe, Azerbaijan, Hy Lạp (Septuagint). Ở ngưỡng < 0.05: **71 văn bản (0.35%)**.
- **Lỗi mã hóa:** tiếng Việt mã VNI chưa chuyển (`N?i di?n ra chung k?t`),
  Cyrillic hiển thị sai dạng Latin, dữ liệu nhị phân rác.
- **Trang ghép nối:** văn bản 121.145 ký tự nhưng chỉ 10 dòng (dòng dài nhất
  119.708 ký tự) - trang bị nối liền khi scrape.
- **Boilerplate:** có nhưng giới hạn - đoạn 60 ký tự lặp nhiều nhất chỉ xuất
  hiện trong 8/20.000 văn bản (menu web, banner Wikipedia).

Khuyến nghị cho nhóm: nếu muốn loại nhiễm ngoại ngữ, thêm lọc tỷ lệ ký tự tiếng
Việt < 0.05 khi chạy final (loại ~0.35%). Hiện pipeline **không tự loại** các
văn bản này - audit chỉ đánh dấu (`data/audit-dev/quality_signals.csv`).

### Phân tích văn bản dài (outliers)

Trung vị 2.780 ký tự; dài nhất 277.549 ký tự. Phân loại 8 văn bản dài nhất:
phần lớn là nội dung dài hợp lệ (văn bản pháp luật, bài Wikipedia, tài liệu
tôn giáo). Hai nhóm có vấn đề nhưng hiếm: trang ghép nối (1 văn bản) và nội
dung ngoại ngữ/lỗi mã hóa (2 văn bản: tiếng Anh, Cyrillic mojibake). Chi tiết:
`data/audit-dev/outlier_findings.md`.

### Kiểm định gần trùng lặp (SimHash, ngưỡng Hamming <= 3, shingle 4)

- **4 cặp gần trùng, 0 cặp vượt qua các split** (tất cả nằm trong train).
- Kiểm tra thủ công cả 4 cặp: **không có cặp nào trùng nội dung** - toàn bộ là
  trang dùng chung template: 1 cặp chương truyện Naruto cùng khung trang
  slbjbl.biz, 3 cặp tin rao bán xe khác nhau cùng khung sanotovietnam.com.vn.
- Đây là nhược điểm đã ghi nhận của SimHash (boilerplate chung làm trang ngắn
  giống nhau); không có bằng chứng rò rỉ nội dung gần trùng giữa các tập.
- Giới hạn: SimHash đo tương đồng bề mặt, không phải ngữ nghĩa; văn bản diễn
  giải lại (paraphrase) có thể bị sót (âm tính giả).

## Chia train / validation / test

Mỗi văn bản chấp nhận được gán tập qua băm ổn định của `id` và seed:
`train` 90%, `validation` 5%, `test` 5%. Cùng `id` luôn vào cùng tập với cùng
seed. Kết quả chạy 500.000: train 17.977 / validation 1.013 / test 1.010
(kỳ vọng 18.000/1.000/1.000 - chênh lệch nhỏ do ranh giới reservoir, xác định
được). Đã kiểm chứng: không trùng `id` hay `text_hash` giữa các tập; mọi dòng
`train.txt` khớp chính xác cột `text` của `train.parquet`.

## So sánh phân bố domain (500.000 quét, 20.000 giữ)

- Độ lệch phần trăm lớn nhất (giữ vs quét): **0.302 điểm phần trăm** (Health).
- Độ lệch split lớn nhất so với tập giữ: **2.02 pp** (People_and_Society).
- Chi tiết: `data/audit-dev/domain_drift.csv`, biểu đồ
  `docs/assets/domain_distribution.svg` (top 12 domain),
  `docs/assets/length_histogram.svg` (phân bố độ dài).

## Thống kê sinh ra

- Số văn bản nguồn đã duyệt, số chấp nhận, số loại bỏ theo từng lý do.
- Số lượng theo tập, theo domain (quét / chấp nhận / giữ / từng tập).
- Độ dài văn bản: min, max, trung bình, trung vị, p90, p95, p99.
- Cấu hình hiệu dụng và manifest tái lập (xem dưới).

## Hiệu năng (chạy development, seed 42)

- Thời gian: 768.8 giây cho 500.000 văn bản (chạy chưa xác thực HF token).
- Tốc độ: 650.4 văn bản/giây (quét), 26.0 văn bản/giây (chấp nhận).
- Bộ nhớ đỉnh: ~1.179 MiB; số băm SHA-256 giữ trong RAM: 499.995.
- Kích thước đầu ra: train.parquet 50 MB / train.txt 97 MB; validation ~2.8/5.7
  MB; test ~2.7/5.6 MB.

## Release 400k (ct219-400k-v1)

### Lệnh chạy

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 \
  .venv/bin/python -u -m src.data.preprocess \
  --mode release-400k \
  --dataset-revision b81fcce58945970117a1b56d50ec81be2628a5c3 \
  --seed 42 --output-dir data/releases/ct219-400k-v1 \
  --dedup-dir data/releases/dedup-400k --cache-dir data/cache/shards
```

Chế độ `release-400k`: 400.000 văn bản giữ, quét toàn bộ stream, dedup sqlite,
loader shard-by-shard (dataset nguồn ~34,65 GB / 132 shard không đủ chỗ đĩa
nên tải từng shard, xử lý rồi xóa; 2 lần chạy thử đầu chết vì lỗi mạng HF
(read timeout, peer closed) - chạy lại an toàn vì mọi thứ xác định).

### Kết quả đo được

- Văn bản nguồn đã duyệt: **12.169.131** (toàn bộ stream, 132 shard).
- Chấp nhận: 400.000; candidate hợp lệ: 12.137.321; sampled out: 11.737.321.
- Loại bỏ: **31.810** (encoding_corruption 14.650, binary_or_invalid_content
  16.499, duplicate_text 661; không trùng id).
- Split: **train 359.988 / validation 20.028 / test 19.984** (kỳ vọng
  360.000/20.000/20.000; chênh lệch do băm, không ép số).
- Thời gian: 28.777,5 giây (~8 giờ); 422,9 văn bản/s nguồn, 13,9 văn bản/s
  chấp nhận; RSS đỉnh ~4.302 MiB.
- SQLite: 2,63 GB trên đĩa (12.137.321 id + 12.137.321 hash), WAL, đã dọn
  sau khi thành công.
- Độ dài: min 202, max 417.047, trung bình 4.105, trung vị 2.763, p90 7.785,
  p95 8.201, p99 8.695.
- Đầu ra: train.parquet 1.044 MB / train.txt 1.936 MB; validation 58/108 MB;
  test 58/108 MB. 25 domain.
- Git commit của mã nguồn lúc chạy: `1afd4fa`; revision nguồn đã khóa:
  `b81fcce58945970117a1b56d50ec81be2628a5c3`.
- Lần chạy thử đầu tiên chết vì `httpx.ReadTimeout`, lần thứ hai vì
  `peer closed connection` khi tải shard qua streaming datasets; sau đó chuyển
  sang shard loader (hf_hub_download có resume + retry) và hoàn tất.

### Kiểm định (phase 6)

- `python -m src.data.release --output-dir data/releases/ct219-400k-v1`:
  **400.000 dòng, checksum khớp manifest, round-trip JSONL chính xác, 0 trùng
  id, 0 trùng hash, 0 chồng lấn giữa các split, 0 tệp tạm**, trạng thái
  complete (marker + manifest status).
- Audit toàn bộ 400.000 văn bản: 16.600 văn bản gắn cờ (4,15%), 5 đề xuất
  loại, 16.595 chỉ cần duyệt. Tín hiệu chính: unusual_unicode 13.347 (ZWSP/bidi
  rải rác), low_vietnamese 1.734, very_low_vietnamese 334, replacement_chars
  482, vni_pattern 404, concatenated_dump 165, foreign_script 152,
  low_alpha 851, extreme_length_few_lines 40.
- Near-duplicate (SimHash, subset xác định 20.000 văn bản, ngưỡng Hamming
  <= 3): **3 cặp, 0 cặp vượt split**; cả 3 đều là trang dùng chung template
  (Zippo/GEVENA cùng cửa hàng, tin rao xe sanotovietnam.com.vn và ban-oto.com),
  không phải trùng nội dung. Dân số thực tế được audit: 20.000/400.000.
- Review packet: `data/review-release-400k/` (16.600 văn bản, cột quyết định
  của người để trống).

### Đăng tải Hugging Face dataset

- Repo đề xuất: `wheevu/ct219-vietnamese-raw-400k` (repo_type=dataset, private
  mặc định, không khai báo licence vì nguồn không có).
- **Chưa thể đăng tải**: token hiện có (`nlp-project`) có role **read**, bị
  hub từ chối tạo repo (403 Forbidden). Đã chuẩn bị sẵn card, checksums.txt,
  data_contract.md, script và lệnh xác minh; cần token write để chạy:
  `python scripts/hf_release.py dataset --release-dir data/releases/ct219-400k-v1`
  rồi `python scripts/hf_release.py verify-dataset --release-dir data/releases/ct219-400k-v1`.

### Model

- **Không có checkpoint hợp lệ** cho model next-token: không có mã huấn luyện
  trong repo; chỉ tìm thấy model NER của bài tập W07 (không phù hợp). Không
  tạo repo model giả, không upload. Tài liệu bàn giao:
  `docs/model_release_handoff.md`; utility `scripts/hf_release.py model-validate`
  đã được kiểm chứng bằng fixture tổng hợp nhỏ (hợp lệ pass, thiếu weights fail).

## Khử trùng lặp quy mô lớn: backend memory vs sqlite

### Đo kiểm bộ nhớ (trước khi sửa)

Đo từng thành phần của pipeline trên máy thật:
- nền Python + imports (datasets/pyarrow/numpy/pandas): ~140 MiB;
- streaming HF (bộ đệm shard/arrow): ~490-540 MiB - hằng số, không phụ thuộc
  số văn bản;
- hồ chứa 20.000 văn bản: ~190 MiB - bị chặn bởi `max_documents`;
- tập băm trùng lặp 500.000 id + 500.000 hash: ~126 MiB - **tăng tuyến tính
  theo số văn bản quét, không có chặn trên** (ước lượng ~250 MiB mỗi triệu).

Kết luận: ở 500k, tập băm chưa phải thành phần lớn nhất, nhưng ở quy mô
hàng chục triệu văn bản (quét toàn bộ dữ liệu) nó sẽ vượt RAM. Do đó cần
backend lưu trên đĩa, không phải vì nhanh hơn ở quy mô hiện tại.

### So sánh backend (quét 200.000, seed 42, cùng điều kiện mạng, 20.000 giữ)

| | memory | sqlite |
| --- | --- | --- |
| Thời gian | 238.3 s | 241.7 s (chênh lệch nhiễu ~1%) |
| Tốc độ | 839.3 văn bản/s | 827.5 văn bản/s |
| RSS đỉnh | 1.525,5 MiB | 1.471,7 MiB (thấp hơn ~54 MiB, đúng bằng chi phí tập băm) |
| Trạng thái trùng lặp | 200k id + 200k hash trong RAM | tệp SQLite 42,8 MB trên đĩa |
| Đầu ra | - | byte-giống hệt (10/10 tệp dữ liệu) |

Kiểm chứng bổ sung: chạy development 500.000 với sqlite cho split counts
17.977/1.013/1.010 và 5 văn bản trùng - **giống hệt** chạy memory giai đoạn 2;
mọi tệp dữ liệu byte-giống hệt (chỉ khác `preprocessing_stats.json` do siêu
dữ liệu thời gian chạy). Backend sqlite: khóa PRIMARY KEY ép duy nhất, giao
dịch theo lô 10.000, WAL, tạo mới mỗi lần chạy (an toàn với lần chạy bị ngắt),
dọn tệp sau khi thành công trừ khi `--keep-dedup-db`. Ghi nhận vào manifest
(`dedup` key: backend, số dòng, kích thước tệp).

Lưu ý đo lường: so sánh với chạy giai đoạn 2 (768.8 s, 1.179 MiB) bị nhiễu bởi
tốc độ mạng khác nhau (chưa có token HF); cặp đo 200k ở trên mới là so sánh
cùng điều kiện. Một lần chạy memory-500k thử lại bị chết giữa chừng do nghẽn
HF (đã có token, không tái diễn) nên không có số liệu.

## Chính sách loại bỏ dữ liệu hỏng (giai đoạn 3)

### Quy tắc và ngưỡng mặc định (có cấu hình)

| Quy tắc | Mặc định | Điều kiện kích hoạt |
| --- | --- | --- |
| `encoding_corruption` | loại bỏ | >= 3 ký tự thay thế U+FFFD |
| `binary_or_invalid_content` | loại bỏ | >= 8 ký tự Unicode bất thường VÀ >= 0.5% tổng ký tự (icon font, bidi mark, ZWSP) |
| `foreign_script_dominant` | chỉ audit | ký tự phi-Latin (Armenia/Ả Rập/Hy Lạp/Cyrillic/CJK...) >= 30% |
| `concatenated_dump` | chỉ audit | một dòng > 20.000 ký tự |

### Vì sao các quyết định này (đều đo trên dữ liệu thật)

- **Phát hiện VNI tự động không đáng tin, chỉ để audit.** Mẫu chữ-cái + '?' +
  chữ-cái xuất hiện trong JS ternary (`n?e:void`), URL query (`watch?v=`),
  và câu hỏi không có khoảng trắng (`hông?cựu`) - cả ba đều có thật trong
  corpus. Văn bản VNI thật (như bài 'D? nh?n' 9.294 ký tự) vẫn được gắn cờ
  `vni_pattern` trong gói duyệt thủ công, chờ con người quyết định. Không tự
  chuyển VNI sang Unicode.
- **Tiếng Anh thuần không bị quy tắc nào loại** (Latin, không dấu hiệu hỏng);
  được gắn cờ `very_low_vietnamese` trong gói duyệt.
- **Hỗn hợp Việt-Anh không bao giờ bị loại vì có tiếng Anh.**
- **Văn bản dài không bị loại vì dài**; chỉ dấu hiệu trang ghép nối
  (dòng đơn > 20.000 ký tự) kích hoạt `concatenated_dump`.

### Kết quả đo trên 20.000 văn bản development (rules_report.json)

- Số văn bản bị gắn cờ: encoding 31, binary 22, concatenated 10, foreign 4;
  **61 văn bản duy nhất** (0.3%).
- Số bị loại theo quy tắc đang bật (encoding + binary): **50** (31 + 19; 3 văn
  bản binary đã trùng vào encoding).
- Chồng lấn: encoding+binary 2, cả ba 1, encoding+foreign 1, binary+foreign 1.
- **102 văn bản không chắc chắn** (chỉ có dấu hiệu audit: tiếng Việt thấp,
  replacement 1-2 ký tự, vni) - cần người duyệt, không tự phân loại.
- Ví dụ id bị loại: encoding - VI_open-0002568749, VI_open-0001230282,
  VI_open-0003775652...; binary - VI_open-0003383282, VI_open-0001970205...
- Gói duyệt thủ công: `data/review/quality_review.csv` (802 văn bản, 4.0%),
  `quality_review.html`, `quality_review_decisions.template.csv` (cột
  `human_decision`, `reviewer_notes` để trống). Chưa có quyết định của người
  duyệt nào được giả định - việc điều chỉnh ngưỡng chờ kết quả duyệt này.
- Văn bản bị loại xuất hiện trong `samples/rejected_examples.csv` (tối đa 20
  mẫu mỗi lý do, kèm id/domain/lý do/preview).

Lưu ý: không công bố precision/recall vì chưa có dữ liệu đánh giá có nhãn;
các con số trên chỉ là số lượng gắn cờ/loại bỏ quan sát được.


## Tái lập

Lệnh chạy mẫu:

```bash
python -m src.data.preprocess --mode smoke --max-documents 2000 --seed 42
python -m src.data.preprocess --mode development --seed 42 --output-dir data/processed-dev
```

- Mọi lần chạy ghi `manifest.json` kèm đầu ra: phiên bản bộ dữ liệu yêu cầu và
  đã phân giải (revision `b81fcce58945970117a1b56d50ec81be2628a5c3`), seed,
  cấu hình, môi trường (Python, thư viện), SHA-256 của từng tệp đầu ra, số dòng.
- Kiểm chứng lặp lại: chạy lại smoke cho **10/10 tệp dữ liệu byte-giống hệt**
  (chỉ `preprocessing_stats.json` khác do thời gian chạy ghi trong tệp).
- Có thể khóa phiên bản nguồn: `--dataset-revision <sha>`.
- Lưu ý: kho lưu trữ mã nguồn chưa có commit nào, nên trường `git_commit` trong
  manifest hiện là `null`; nên commit mã trước khi chạy final để ghi nhận được.
- Chưa đo: quét toàn bộ bộ dữ liệu (chế độ final, không giới hạn quét) -
  ước lượng hàng giờ nên chưa chạy; kết quả trên đây là mẫu 500.000 văn bản.

## Hạn chế

- Mẫu bị giới hạn quét chỉ đại diện phần đầu luồng (xem mục lấy mẫu).
- Bộ nhớ tăng theo số băm trùng lặp (499.995 băm ~ vài chục MB cho 500k).
- Kết quả ổn định theo phiên bản bộ dữ liệu; nếu nguồn cập nhật, kết quả có
  thể thay đổi với cùng seed. Manifest ghi revision đã phân giải để đối chiếu.
- Trường `git_commit` còn trống cho tới khi repo được commit.
