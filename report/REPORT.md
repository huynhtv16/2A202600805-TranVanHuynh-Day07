# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Tên sinh viên]
**Nhóm:** [Tên nhóm]
**Ngày:** [Ngày nộp]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Cosine similarity đo mức độ hai vector hướng cùng chiều, tương đương với hai đoạn văn có nội dung gần giống nhau về ý nghĩa. Với text embeddings, điểm cao tức là các câu/ngữ đoạn có semantic tương đồng, dù từ ngữ khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A: The cat sat on the mat.
- Sentence B: A cat was sitting on a mat.
- Tại sao tương đồng: Cả hai câu đều mô tả cùng một hành động và đối tượng, chỉ khác cách diễn đạt.

**Ví dụ LOW similarity:**
- Sentence A: The weather is sunny and warm today.
- Sentence B: I will submit my report tonight.
- Tại sao khác: Một câu nói về thời tiết, câu kia nói về công việc và thời gian nộp báo cáo.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ đo hướng của vector và bỏ qua độ lớn, nên phù hợp với embeddings đã được chuẩn hóa. Điều này giúp phân biệt semantic similarity tốt hơn khi độ dài vector hoặc cường độ embedding khác nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Dùng công thức `ceil((doc_length - overlap) / (chunk_size - overlap))`.
> `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23`.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Với overlap=100, số chunk là `ceil((10000 - 100) / 400) = ceil(9900 / 400) = 25`.
> Overlap lớn hơn giúp giữ lại ngữ cảnh khi nội dung quan trọng nằm ở ranh giới giữa hai chunk, cải thiện khả năng retrieval và giảm mất nghĩa do cắt ngang.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** [ví dụ: Customer support FAQ, Vietnamese law, cooking recipes, ...]

**Tại sao nhóm chọn domain này?**
> [Thông tin domain và lý do chọn sẽ được nhóm bổ sung khi hoàn thành phần nhóm.]

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| | FixedSizeChunker (`fixed_size`) | | | |
| | SentenceChunker (`by_sentences`) | | | |
| | RecursiveChunker (`recursive`) | | | |

### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy]

**Mô tả cách hoạt động:**
> [Phần này sẽ hoàn thiện khi nhóm xác định domain và document set cụ thể.]

**Tại sao tôi chọn strategy này cho domain nhóm?**
> [Điền sau khi so sánh với các member khác trong nhóm.]

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | | | |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> [Hoàn thiện khi có dữ liệu so sánh thực tế.]

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Tôi dùng regex `(?<=[.!?])(?:\s+|\n+)` để tách câu bằng dấu chấm, chấm than, chấm hỏi, hoặc ngắt dòng. Sau đó strip whitespace và gom nhóm các câu thành chunk với tối đa `max_sentences_per_chunk` câu.

**`RecursiveChunker.chunk` / `_split`** — approach:
> `RecursiveChunker` thử split theo các separator ưu tiên `['\n\n','\n','. ',' ','']`. Nếu đoạn text vẫn lớn hơn `chunk_size`, nó sẽ đệ quy xuống separator tiếp theo. Base case là khi text đã đủ nhỏ hoặc không còn separator, khi đó đoạn text được cắt thẳng theo `chunk_size`.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` tạo record cho từng document bằng cách gọi hàm embedding trên `doc.content`, lưu vào bộ nhớ in-memory và cố gắng thêm vào ChromaDB nếu có thể. `search` tạo embedding của truy vấn rồi xếp hạng các record theo dot product (tương đương cosine similarity khi embeddings đã chuẩn hóa).

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` lọc record theo metadata trước khi tính score, giúp hạn chế candidate space và giữ top-k liên quan. `delete_document` xóa tất cả record có id trùng `doc.id` hoặc metadata `doc_id`, rồi trả về `True` nếu có bản ghi bị xóa.

### KnowledgeBaseAgent

**`answer`** — approach:
> Agent lấy top-k chunk từ store, ghép nội dung và metadata vào prompt dưới dạng context, rồi gọi `llm_fn(prompt)` để trả về câu trả lời. Cách này tuân theo pattern retrieval-augmented generation (RAG).

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\huynh\AppData\Local\Python\pythoncore-3.14-64\python.exe
cachedir: .pytest_cache
rootdir: C:\Documents\Vin\buoi7\2A202600805-TranVanHuynh-Day07
plugins: anyio-4.13.0
collected 42 items

... (42 passed output omitted for brevity) ...

============================= 42 passed in 0.12s =============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | The cat sat on the mat. | A cat was sitting on a mat. | high | -0.158218 | ❌ |
| 2 | The weather is sunny and warm today. | I will submit my report tonight. | low | 0.023952 | ✅ |
| 3 | OpenAI develops powerful AI models. | The model generates text based on patterns. | high | -0.033331 | ❌ |
| 4 | Vietnamese cuisine uses fish sauce and rice. | My car needs a new tire. | low | 0.033787 | ✅ |
| 5 | Please delete the document from the store. | The store must remove the file with matching id. | high | 0.060186 | ❌ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Mock embeddings trong repository sử dụng hàm băm MD5 và tạo vector định lượng theo seed, nên điểm similarity không phản ánh hoàn toàn nghĩa tự nhiên. Điều này chứng tỏ khi đánh giá retrieval, backend embedding chất lượng cao rất quan trọng; mô hình mock chỉ phù hợp cho test cấu trúc, không phải semantic fidelity.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi hiểu rõ hơn cách metadata có thể giúp lọc các đoạn thông tin chính xác hơn, nhất là khi bộ tài liệu có nhiều chủ đề gần giống.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Một số nhóm dùng chunking theo section/header thay vì câu, điều này giúp giữ nguyên bối cảnh khi tài liệu có cấu trúc rõ ràng.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ ưu tiên chọn tài liệu có cấu trúc nhất quán và gán metadata rõ ràng từ đầu, để giảm khả năng top-k trả về chunk rác do thiếu bối cảnh.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 0 / 10 |
| Chunking strategy | Nhóm | 0 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 0 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 0 / 5 |
| **Tổng** | | **50 / 100** |
