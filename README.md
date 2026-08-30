# Vietnamese NLP preprocessing pipeline

Pipeline streaming để làm sạch và chia tập văn bản tiếng Việt cho đồ án cuối kỳ CT219.

Pipeline giữ dấu tiếng Việt, cấu trúc đoạn và metadata truy vết.
Nó không tokenize, huấn luyện model hoặc tự động loại văn bản chỉ vì ngôn ngữ có vẻ lạ.

<table>
  <tr>
    <td><img src="docs/assets/domain_distribution.svg" alt="Phân bố domain qua pipeline"></td>
    <td><img src="docs/assets/length_histogram.svg" alt="Phân bố độ dài văn bản"></td>
  </tr>
</table>

## Đầu ra

- Parquet và JSONL cho train, validation và test
- Stable split theo seed và document ID
- Exact deduplication trong RAM hoặc SQLite
- Statistics, manifest, checksums và revision dataset
- Quality audit chỉ đọc cùng review packet cho người kiểm tra

Dữ liệu được xử lý theo streaming, nên pipeline không cần nạp toàn bộ dataset vào RAM.

[Hướng dẫn chạy, cấu hình và giới hạn](GUIDE.md).

- [Data contract](docs/data_contract.md)
- [Báo cáo](docs/report.md)
- [Model handoff](docs/model_release_handoff.md)
