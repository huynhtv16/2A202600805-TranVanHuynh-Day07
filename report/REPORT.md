# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Trần Văn Huỳnh
**Nhóm:** F4
**Ngày:** 5/6/2026

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

**Domain:** Chăm sóc sức khỏe cộng đồng — đặc biệt là phân biệt các dạng sốt thường gặp ở Việt Nam.

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain này vì các tài liệu liên quan đến sốt và bệnh lý trẻ em rất phù hợp cho một hệ thống tìm kiếm dựa trên retrieval. Nội dung rõ ràng, thực tiễn, và có nhiều câu hỏi phổ biến mà người dùng quan tâm. Đặc biệt, tài liệu y tế này có cấu trúc tốt để thử các chiến lược chunking khác nhau và metadata có thể giúp lọc theo loại bệnh.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | benh-sot-tinh-hong-nhiet-sot-scarlet-la-benh-gi-vi.md | vinmec.com | ~3.6k | disease: sot_scarlet; audience: chung |
| 2 | huong-dan-phan-biet-sot-virus-voi-sot-xuat-huyet-vi.md | vinmec.com | ~3.2k | disease: sot_virus / sot_xuat_huyet; audience: chung |
| 3 | phan-biet-sot-ret-va-sot-xuat-huyet-vi.md | vinmec.com | ~2.8k | disease: sot_ret / sot_xuat_huyet; audience: chung |
| 4 | phan-biet-sot-thuong-sot-virus-va-sot-xuat-huyet-vi.md | vinmec.com | ~3.1k | disease: sot_virus / sot_xuat_huyet; audience: chung |
| 5 | sot-phat-ban-khac-sot-xuat-huyet-nhu-nao-vi.md | vinmec.com | ~2.1k | disease: sot_phat_ban / sot_xuat_huyet; audience: chung |
| 6 | sot-xuat-huyet-va-sot-xuat-huyet-nang.md | vinmec.com | ~2.4k | disease: sot_xuat_huyet; audience: chung |
| 7 | tre-sot-den-dau-moi-phai-uong-thuoc-ha-sot-vi.md | vinmec.com | ~1.6k | disease: pediatric_fever; audience: ba_me |
| 8 | sot-nong-lanh-nhuc-moi-dau-dau-co-phai-sot-virus-vi.md | vinmec.com | ~1.2k | disease: general_fever; audience: chung |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `disease` | keyword | `sot_xuat_huyet`, `sot_virus`, `sot_scarlet` | Giúp filter nhanh theo nhóm bệnh khi queries cụ thể về một bệnh |
| `audience` | keyword | `chung`, `ba_me`, `nguoi-lon` | Cho phép ưu tiên chunk phù hợp cho người hỏi (ví dụ: hướng dẫn chăm sóc trẻ)
| `source` | text | `vinmec.com` | Giúp trace nguồn và ưu tiên nguồn y tế chính thống khi cần |
| `language` | text | `vi` | Lọc theo ngôn ngữ khi bộ docs đa ngôn ngữ |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` hoặc benchmark trên bộ tài liệu đã chọn để thu thập thông tin chunk count và độ dài trung bình.

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Bộ tài liệu sốt | FixedSizeChunker (`fixed_size`) | 99 | 388 | Trung bình tốt cho embedding nhưng ít cố gắng giữ ý nghĩa ngữ cảnh |
| Bộ tài liệu sốt | SentenceChunker (`by_sentences`) | 110 | 306 | Giữ tốt ranh giới câu và phù hợp với các câu ngắn trong văn bản y tế |
| Bộ tài liệu sốt | RecursiveChunker (`recursive`) | 130 | 259 | Giữ ý nghĩa đoạn văn lớn hơn, phù hợp cho RAG nhưng cần điều chỉnh với chunk size |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
> Tôi chọn `RecursiveChunker` vì nó ưu tiên tách theo các ranh giới ngữ nghĩa đầu tiên: `\n\n`, `\n`, `. ` rồi mới cắt theo độ dài ký tự. Điều này giúp giữ được các ý lớn trong đoạn văn của tài liệu y tế và giảm thiểu việc cắt một nửa câu hoặc ý nghĩa.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Với nội dung sức khỏe và phân biệt triệu chứng, mỗi đoạn thường chứa một cụm thông tin hoàn chỉnh. `RecursiveChunker` cho phép giữ đoạn văn có ý nghĩa đầy đủ, đồng thời vẫn giới hạn chunk trong khoảng an toàn cho embedding.

**Code snippet (nếu custom):**
```python
from src.chunking import RecursiveChunker
chunker = RecursiveChunker(chunk_size=400)
chunks = chunker.chunk(text)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Benchmark chung | SentenceChunker | 110 | 305 | 10/10 |
| | **RecursiveChunker (của tôi)** | 130 | 259 | 8/10 |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Nguyễn Ngọc Dũng | FixedSizeChunker | 10 | Chunk dài ổn định, dễ tối ưu top-k | Khó giữ ý toàn vẹn khi nội dung không đều |
| Nguyễn Thái Dương | SentenceChunker | 10 | Giữ ranh giới câu rõ ràng, phù hợp nội dung y tế | Một số chunk vẫn quá dài với câu phức |
| Đỗ Trung Kiên | SlidingWindowChunker | 8 | Giữ ngữ cảnh liên câu tốt, tăng khả năng bắt ý xuyên câu | Tạo nhiều chunk chồng chéo, đôi khi tăng noise |
| Trần Đức Lương | ParagraphChunker | 8 | Giữ được ý nghĩa từng đoạn bài viết, phù hợp format .md | Đoạn quá dài vẫn cần cắt sâu hơn |
| Tôi | RecursiveChunker | 8 | Giữ ý lớn, cân bằng giữa ngữ nghĩa và kích thước chunk | Vẫn cần tweek thêm với tài liệu dài hơn |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Nhóm thấy rằng với bộ docs bệnh lý và câu hỏi y tế, chiến lược `SentenceChunker` và `FixedSizeChunker` cho ra top-3 retrieval tốt nhất. Tuy nhiên cách tiếp cận `RecursiveChunker` vẫn có ưu thế khi cần giữ nguyên ý nghĩa của từng đoạn thông tin và làm nguồn dữ liệu cho RAG.

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
| 1 | Làm sao phân biệt sốt virus và sốt xuất huyết ở giai đoạn sớm? | Gold: Dựa vào xét nghiệm cận lâm sàng (Test Dengue, công thức máu thấy tiểu cầu giảm / Hct tăng). Triệu chứng gợi ý sốt xuất huyết: đau hốc mắt, đau người, xuất huyết dưới da, chảy máu nhẹ; sốt virus thường tự giới hạn ~7 ngày. (ref: huong-dan-phan-biet-sot-virus-voi-sot-xuat-huyet-vi.md) |
| 2 | Triệu chứng đặc trưng và cách điều trị sốt tinh hồng nhiệt (scarlet fever)? | Gold: Ban đỏ, lưỡi 'dâu tây', sốt cao khởi phát; điều trị kháng sinh (penicillin/kháng sinh theo chỉ định) trong ~10 ngày; cải thiện sau 12-24h dùng kháng sinh. (ref: benh-sot-tinh-hong-nhiet-sot-scarlet-la-benh-gi-vi.md) |
| 3 | Dấu hiệu giúp phân biệt sốt phát ban (sởi/rubella) và sốt xuất huyết trên da? | Gold: Thử căng da: nếu nốt phát ban lặn khi căng da -> phát ban do virus (ví dụ sởi/rubella); nếu không lặn -> có thể là xuất huyết (sốt xuất huyết). Xác định chính xác bằng xét nghiệm. (ref: sot-phat-ban-khac-sot-xuat-huyet-nhu-nao-vi.md) |
| 4 | Khi nào cần đưa trẻ nghi sốt xuất huyết đến cơ sở y tế ngay? | Gold: Nếu sốt cao không hạ, li bì/vật vã, nôn nhiều không uống được, đau bụng dữ dội, chảy máu, tiểu ít hoặc có dấu hiệu sốc → khám cấp cứu. (refs: huong-dan-phan-biet-sot-virus-voi-sot-xuat-huyet-vi.md; sot-xuat-huyet-va-sot-xuat-huyet-nang.md) |
| 5 | Làm sao phân biệt sốt rét và sốt xuất huyết? | Gold: Dựa vào véc-tơ truyền (muỗi Anopheles vs Aedes), thời gian ủ bệnh khác nhau, sốt rét có rét run theo từng đợt; sốt xuất huyết biểu hiện xuất huyết và giảm tiểu cầu. (ref: phan-biet-sot-ret-va-sot-xuat-huyet-vi.md) |

### Kết Quả Của Tôi

> Đã chạy với `LocalEmbedder` từ `sentence-transformers` (`all-MiniLM-L6-v2`) qua `EMBEDDING_PROVIDER=local`.

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Làm sao phân biệt sốt virus và sốt xuất huyết ở giai đoạn sớm? | "## 3. Cách phân biệt sốt virus và sốt xuất huyết  Để phân biệt sốt virus và sốt xuất huyết dựa vào:" (preview from huong-dan-phan-biet-sot-virus-voi-sot-xuat-huyet-vi.md__20) | 0.314 | Yes | N/A |
| 2 | Triệu chứng đặc trưng và cách điều trị sốt tinh hồng nhiệt (scarlet fever)? | "Sốt tinh hồng nhiệt hay còn gọi là scarlet fever là một loại bệnh nhiễm trùng cấp tính..." (preview from benh-sot-tinh-hong-nhiet-sot-scarlet-la-benh-gi-vi.md__1) | 0.588 | Yes | N/A |
| 3 | Dấu hiệu giúp phân biệt sốt phát ban (sởi/rubella) và sốt xuất huyết trên da? | "Sốt phát ban là bệnh lý đặc trưng bởi dấu hiệu sốt và nổi ban đỏ..." (preview from sot-phat-ban-khac-sot-xuat-huyet-nhu-nao-vi.md__1) | 0.625 | Yes | N/A |
| 4 | Khi nào cần đưa trẻ nghi sốt xuất huyết đến cơ sở y tế ngay? | "Điều đáng lưu ý là, với sốt virus thông thường, khi hết sốt thì triệu chứng bệnh cũng thuyên giảm..." (preview from phan-biet-sot-thuong-sot-virus-va-sot-xuat-huyet-vi.md__18) | 0.535 | Yes | N/A |
| 5 | Làm sao phân biệt sốt rét và sốt xuất huyết? | "Dịch sốt xuất huyết, sốt virus hiện đang vào mùa cao điểm..." (preview from phan-biet-sot-thuong-sot-virus-va-sot-xuat-huyet-vi.md__1) | 0.659 | Yes | N/A |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

**Evaluation (group retrieval quality):**

- Top-3 chứa chunk relevant: 5 / 5
- Retrieval score (per rubric): 10 / 10

**Ghi chú:** `sentence-transformers` đã được dùng thành công, và kết quả retrieval cải thiện rõ rệt so với mock embeddings. Nếu cần, có thể tiếp tục tinh chỉnh chunk_size hoặc metadata để nâng chất lượng answer grounding.

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
| Document selection | Nhóm | 8 / 10 |
| Chunking strategy | Nhóm | 12 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 0 / 5 |
| **Tổng** | | **80 / 100** |

