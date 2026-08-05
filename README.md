# Vietnamese NLP Preprocessing Pipeline

Pipeline streaming để tiền xử lý văn bản tiếng Việt thô, thuộc đồ án cuối kỳ NLP
của trường (next-token generation từ văn bản tiếng Việt thô). Repository này chỉ
bao gồm phần **tiền xử lý dữ liệu**: biến dataset gốc thành văn bản thô sạch, sẵn
sàng cho model. Tokenization, huấn luyện model và đánh giá do các thành viên
khác trong nhóm đảm nhiệm.

Dataset gốc: [VTSNLP/vietnamese_curated_dataset](https://huggingface.co/datasets/VTSNLP/vietnamese_curated_dataset)

## 1. Pipeline làm gì

- Tải Hugging Face dataset theo chế độ **streaming** mặc định (không bao giờ nạp
  toàn bộ dataset vào RAM).
- Áp dụng **làm sạch tiếng Việt thận trọng** (xem mục 7).
- Loại bỏ văn bản không dùng được, ghi lý do loại bỏ cho từng văn bản.
- Phát hiện trùng `id` và trùng văn bản chính xác sau chuẩn hóa (SHA-256 hash).
- Lấy mẫu xác định bằng reservoir sampler có seed (không thiên lệch theo kiểu
  chọn N văn bản đầu tiên).
- Chia tập xác định thành train / validation / test (mặc định 90 / 5 / 5).
- Xuất văn bản thô sạch thành tệp parquet + JSONL, kèm statistics, phân bố
  domain, removal counts và mẫu trước/sau khi xử lý.
- Lưu cấu hình hiệu dụng cùng đầu ra để tái lập được thí nghiệm.

## 2. Giới hạn

Pipeline **cố ý không làm** những việc sau:

- Không tokenization, subword splitting, word segmentation hay `input_ids`.
- Không huấn luyện model, fine-tuning, đánh giá, perplexity hay generation.
- Không sequence chunking hay chọn context window.
- Không hạ chữ thường, bỏ dấu tiếng Việt, bỏ dấu câu, stemming hay lemmatization.
- Văn bản giữ nguyên: dấu tiếng Việt, chữ hoa, số và cấu trúc đoạn đều được giữ
  vì chúng hữu ích cho language modelling.

## 3. Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Yêu cầu Python 3.10+. Chạy test: `pip install pytest` (đã có trong
`requirements.txt`) rồi `python -m pytest`.

## 4. Smoke test

```bash
python -m src.data.preprocess --mode smoke --max-documents 2000 --seed 42
```

Chế độ `smoke` nhắm tới ~2.000 văn bản được chấp nhận và quét tối đa 50.000 văn
bản nguồn. Đầu ra mặc định nằm trong `data/processed/`. Thêm `--overwrite` để
ghi đè đầu ra cũ.

## 5. Chạy preprocessing quy mô lớn hơn

```bash
# Development run (~20.000 văn bản)
python -m src.data.preprocess --mode development --seed 42

# Final run (~50.000 văn bản, quét toàn bộ stream)
python -m src.data.preprocess --mode final --seed 42 --overwrite

# Custom run: dataset khác, không giới hạn quét, tinh chỉnh threshold
python -m src.data.preprocess --dataset VTSNLP/vietnamese_curated_dataset \
  --mode final --max-documents 50000 --scan-limit 0 \
  --min-chars 30 --max-chars 10000 --train-ratio 0.9 --val-ratio 0.05 \
  --output-dir data/processed --seed 42 --overwrite
```

Mọi tùy chọn đều xem được bằng `python -m src.data.preprocess --help`.

### Deduplication backend

| Backend | Trạng thái | Bộ nhớ | Dùng khi |
| --- | --- | --- | --- |
| `memory` (mặc định) | hai Python set | ~126 MiB mỗi 500k candidate, không chặn trên | scan giới hạn |
| `sqlite` | SQLite trên đĩa, cột PRIMARY KEY, WAL, commit theo batch | RAM phẳng; tệp db nằm trên đĩa | scan toàn bộ dataset |

```bash
python -m src.data.preprocess --mode final --dedup-backend sqlite --dedup-dir data/dedup
```

Hai backend cho quyết định chấp nhận/loại bỏ giống hệt nhau (đã kiểm chứng: đầu
ra byte-giống hệt trên smoke và scan 200k/500k). Database sqlite được tạo mới
trong mỗi lần chạy, nằm trong `--dedup-dir`, và bị xóa sau khi chạy thành công
trừ khi truyền `--keep-dedup-db`. Backend đã dùng và statistics của database
được ghi trong `manifest.json` (key `dedup`).

### Quy tắc loại bỏ thận trọng

Bốn quy tắc tùy chọn chỉ loại bỏ văn bản rõ ràng không dùng được. Threshold
hiển thị trong cấu hình và mọi lần loại bỏ đều được đếm:

| Rule | Mặc định | Điều kiện kích hoạt (mặc định) |
| --- | --- | --- |
| `encoding_corruption` | reject | >= 3 ký tự thay thế (U+FFFD) |
| `binary_or_invalid_content` | reject | >= 8 ký tự Unicode bất thường VÀ >= 0.5% tổng ký tự (icon font, bidi marks, ZWSP) |
| `foreign_script_dominant` | audit-only | ký tự phi-Latin (Armenian/Arabic/Greek/Cyrillic/CJK/...) >= 30% |
| `concatenated_dump` | audit-only | một dòng duy nhất > 20.000 ký tự |

Bật/tắt từng rule bằng `--reject-encoding-corruption` /
`--no-reject-encoding-corruption` (tương tự cho các rule khác).

Các quyết định cố ý không làm, đều đã đo trên dữ liệu thật:

- **VNI mojibake chỉ để audit.** Mẫu chữ-cái + dấu hỏi + chữ-cái cũng xuất hiện
  trong JS ternary, URL query string và câu hỏi không có khoảng trắng, nên phát
  hiện VNI tự động không đáng tin (corpus có cả ba dạng này).
- **Tiếng Anh thuần không bị rule nào loại** (Latin script, không dấu hiệu hỏng);
  nó được gắn cờ `very_low_vietnamese` trong review packet.
- **Văn bản pha Việt-Anh không bao giờ bị loại** chỉ vì có tiếng Anh.
- **Văn bản dài không bao giờ bị loại chỉ vì dài**; chỉ dấu hiệu page dump mới
  kích hoạt `concatenated_dump`.
- **Không tự động chuyển VNI sang Unicode.**

Văn bản bị loại được ghi lại (giới hạn số lượng) trong
`samples/rejected_examples.csv` với id, domain, lý do và preview rút gọn.

### Khóa revision của dataset

Truyền `--dataset-revision <commit-sha-hoặc-tag>` để khóa đúng phiên bản
dataset. Mỗi lần chạy cũng ghi revision đã phân giải trong `manifest.json` để
đối chiếu kết quả giữa các lần chạy kể cả khi không khóa tường minh.

### Quality audit (chỉ đọc)

Sau khi chạy, kiểm định đầu ra mà không đụng vào chúng:

```bash
python -m src.data.audit --output-dir data/processed --audit-dir data/audit \
  --seed 42 --near-duplicates
```

Sinh ra quality signal cho từng văn bản, mẫu duyệt theo chỉ số cực trị, domain
drift (inspected -> accepted -> retained -> split), preview văn bản dài bất
thường, và (với `--near-duplicates`) báo cáo near-duplicate bằng SimHash. Audit
chỉ gắn cờ, không bao giờ loại bỏ hay sửa dữ liệu.

### Review packet thủ công

Audit cũng xuất mọi văn bản bị gắn cờ vào `data/review/`:

- `quality_review.csv` - mọi chỉ số kèm preview rút gọn
- `quality_review.html` - bảng HTML dễ đọc
- `quality_review_decisions.template.csv` - template điền tay với cột
  `proposed_decision`, cột `human_decision` và `reviewer_notes` để trống

802 trên 20.000 văn bản development bị gắn cờ (4.0%); 50 văn bản được đề xuất
loại. Các threshold về ngôn ngữ (`--min-non-latin-share`, xử lý VNI) phải được
chốt dựa trên kết quả duyệt thủ công packet này, không dựa vào phát hiện tự
động.

### Các chế độ

| Mode | Số văn bản chấp nhận | Giới hạn quét nguồn |
| --- | --- | --- |
| `smoke` | 2.000 | 50.000 |
| `development` | 20.000 | 500.000 |
| `final` | 50.000 | không giới hạn (toàn bộ stream) |

Các giá trị này là mặc định, không hard-code: `--max-documents` và
`--scan-limit` ghi đè được. `--scan-limit 0` nghĩa là không giới hạn.

## 6. Định dạng tệp đầu ra

```
data/processed/
├── processed/
│   ├── train.parquet        # id, domain, text, text_hash, character_count
│   ├── validation.parquet
│   ├── test.parquet
│   ├── train.txt            # JSONL: mỗi dòng là một văn bản JSON-escaped
│   ├── validation.txt
│   └── test.txt
├── statistics/
│   ├── preprocessing_stats.json   # config hiệu dụng, số đếm, split counts
│   ├── domain_distribution.csv    # inspected vs accepted theo domain
│   ├── removal_counts.csv         # mọi lý do loại bỏ, đều được đếm
│   └── length_statistics.json     # min/max/mean/median/p90/p95/p99
└── samples/
    └── before_after_examples.csv  # cặp mẫu rút gọn (mặc định 10)
```

Kèm theo `manifest.json` cạnh `processed/`: tên dataset, revision yêu cầu và đã
phân giải, seed, cấu hình hiệu dụng, phiên bản Python/các dependency, git
commit (khi repo có commit), SHA-256 checksum từng tệp, row count và thời gian
chạy. Xem `docs/data_contract.md` để biết hợp đồng bàn giao đầy đủ.

Các tệp `.txt` là **JSONL** (mỗi dòng là một văn bản JSON-escaped). Đây là chủ
đích: văn bản thô chứa dấu xuống dòng và dòng trống, nên định dạng plain text
không thể tách văn bản một cách đáng tin. Phía downstream đọc từng dòng rồi
`json.loads`; văn bản nhiều dòng round-trip chính xác. Các tệp `.parquet` chứa
cùng văn bản kèm metadata (`text_hash`, `character_count`).

## 7. Quy tắc làm sạch

Loại bỏ:

- văn bản thiếu hoặc chỉ có khoảng trắng (`empty_text`)
- văn bản không có nội dung có nghĩa, tức không có chữ cái hay chữ số
  (`invalid_text`)
- văn bản ngắn hơn `--min-chars` (mặc định 20) (`too_short`)
- văn bản dài hơn `--max-chars`, khi có đặt (`too_long`)
- trùng giá trị `id` (`duplicate_id`)
- trùng văn bản chính xác sau chuẩn hóa (`duplicate_text`)

Chuẩn hóa:

- Unicode về NFC (dấu tiếng Việt được giữ nguyên)
- kết thúc dòng về LF (`\r\n` / `\r` -> `\n`)
- khoảng trắng lặp về một khoảng trắng, tab thành khoảng trắng
- tối đa một dòng trống liên tiếp (ranh giới đoạn văn được giữ)
- cắt khoảng trắng đầu/cuối, bỏ khoảng trắng cuối mỗi dòng

Giữ nguyên:

- dấu câu, chữ hoa, số, cấu trúc đoạn và mọi ký hiệu Unicode tiếng Việt hợp lệ.
- Ký tự điều khiển bị bỏ; LF và tab được giữ.

Mỗi văn bản bị loại đều tăng một bộ đếm có tên, nên không có gì bị bỏ âm thầm.

## 8. Phương pháp chia tập

Mỗi văn bản được chấp nhận được gán vào một split bằng **stable hash**:

```python
value = sha256("split|{seed}|{doc_id}") / 2^64
train        nếu value < train_ratio
validation   nếu value < train_ratio + val_ratio
test         ngược lại
```

Cùng một văn bản luôn vào cùng một split với cùng seed, qua mọi lần chạy và mọi
máy. Duplicate bị loại trước khi chia tập, nên không có id hay văn bản chuẩn
hóa nào xuất hiện ở hai split. Tỷ lệ mặc định là 90 / 5 / 5.

## 9. Các giới hạn đã biết

- **Lấy mẫu:** reservoir sampler đồng đều trên mọi thứ nó quét. Khi có giới hạn
  quét (smoke/development), mẫu đồng đều trên phần prefix đó, có thể không đại
  diện cho toàn bộ dataset. `final` quét toàn bộ stream. Không đảm bảo phân
  tầng theo domain; phân bố inspected vs accepted được ghi lại để thiên lệch
  hiện ra, không bị giấu.
- **Phạm vi khử trùng:** hash trùng lặp được theo dõi cho mọi candidate được
  chấp nhận trong phạm vi quét. Chạy toàn dataset với backend `memory` giữ hash
  trong RAM; scan rất lớn sẽ dùng nhiều RAM hơn (32 byte cho mỗi candidate).
  Dùng `--dedup-backend sqlite` cho scan toàn bộ dataset.
- **Tính xác định:** kết quả ổn định với cùng phiên bản dataset và seed. Thứ tự
  streaming của Hugging Face cố định theo phiên bản dataset, nhưng nếu dataset
  được cập nhật thì kết quả lấy mẫu có thể đổi.
- **Fallback khi stream ngắn:** nếu stream có ít văn bản dùng được hơn mục tiêu,
  lần chạy vẫn thành công với số văn bản ít hơn và in cảnh báo.

## 10. Sử dụng đầu ra (cho teammate làm tokenizer / model)

- **Văn bản thô:** đọc từng dòng trong tệp `.txt`; `json.loads` mỗi dòng để lấy
  một văn bản hoàn chỉnh. Đưa thẳng vào tokenizer của bạn.
- **Metadata:** tệp `.parquet` chứa `id`, `domain`, `text_hash` và
  `character_count` bên cạnh văn bản, dùng để truy vết và lọc.
- Các split được đảm bảo rời nhau theo cả `id` lẫn văn bản chuẩn hóa. Dùng
  `train.parquet` / `train.txt`, `validation.*`, `test.*` nguyên trạng.
- Không làm sạch lại: văn bản đã NFC-normalized và khử trùng lặp.
