# Medical Imaging Multimodal RAG — 內部技術文件

一個針對放射科報告的 Retrieval-Augmented Generation 系統。使用者可以用自然語言提問胸部 X 光相關問題，也可以上傳 X 光影像，系統會從知識庫中檢索相關報告，再由 LLM 生成有引用來源的回答。

---

## 目錄

- [功能總覽](#功能總覽)
- [系統架構](#系統架構)
- [各模組說明](#各模組說明)
- [環境設定與執行](#環境設定與執行)
- [評估系統](#評估系統)
- [設計決策與取捨](#設計決策與取捨)
- [專案結構](#專案結構)

---

## 功能總覽

### RAG Pipeline（核心功能）

| 功能 | 說明 |
|---|---|
| 多模態檢索 | 純文字問答 + 可選上傳胸部 X 光。影像會先經過 Vision 後端轉成文字描述，再與文字 query 合併送去做向量搜尋 |
| 雙重 Vision 後端 | Gemini Vision（雲端，品質最好）和 Ollama llava（本地，完全免費）。透過 `.env` 裡的 `VISION_BACKEND` 切換，含自動 fallback |
| Section-based Chunking | 放射科報告有固定結構（indication / findings / impression），按這些自然段落切 chunk，而不是用固定 token 數硬切。好處是每個 chunk 語意完整，不會把一句話切斷 |
| 雙重 Embedding 後端 | TF-IDF（離線開發，不需要 API key）和 Gemini `text-embedding-004`（正式環境，品質更好）。透過 `.env` 裡的 `EMBEDDING_BACKEND` 切換，不用改程式碼 |
| ChromaDB 向量資料庫 | 內嵌式資料庫，cosine similarity 搜尋，資料持久化在 `data/chroma_db/`。不需要額外啟動服務 |
| 雙重 Generation 後端 | Gemini 2.0 Flash（雲端，品質最好）和 Ollama（本地，完全離線）。透過 `.env` 裡的 `GENERATION_BACKEND` 切換 |
| LLM 生成回答 | Temperature 0.3（醫療領域需要保守、精確的回答）。會強制引用來源 `[Source 1]`、`[Source 2]` 並附加免責聲明 |
| 自動 Fallback 機制 | 主要後端失敗 → 嘗試另一個 LLM 後端 → 最後退回本地純文字模式。確保系統永遠不會完全掛掉 |
| 離線降級模式 | 沒有 API key 時，embedding 走 TF-IDF，生成回答直接回傳檢索到的原文而非 LLM 摘要 |

### 評估系統

| 功能 | 說明 |
|---|---|
| RAGAS 風格指標 | Context Precision、Context Recall（純計算）、Faithfulness、Answer Relevancy（LLM 評分） |
| LLM-as-Judge | 一次 Gemini 呼叫同時評三個面向：醫學用語正確性、引用準確度、回答完整度 |
| Golden QA 測試集 | 40 題，手動從 20 份報告撰寫，按難度和類型分層 |
| 互動式 HTML 報告 | Plotly 雷達圖、長條圖、摘要表、逐題明細表，自包含 HTML 可直接用瀏覽器開 |

### Web 介面

Gradio UI（`gr.themes.Soft()` 主題）在 port 7860，雙欄佈局：

- **左欄**：影像上傳（可選）+ 文字輸入框 + 5 道範例題（肺炎、心臟肥大、肋膜積液等常見主題）
- **右欄**：AI 回答（Markdown 格式）+ 檢索來源（report UID、section、相關度百分比、文字預覽）
- **自動 Ingestion**：首次啟動時如果沒有向量索引，會自動跑 ingestion
- **影像分析區塊**：上傳 X 光時，額外顯示 Gemini Vision 的影像描述

---

## 系統架構

### 查詢流程

![RAG Query Pipeline](assets/rag_query_pipeline.png)

### 資料處理流程（Ingestion，只需跑一次）

![Data Ingestion Pipeline](assets/data_ingestion_pipeline.png)

---

## 各模組說明

### `config.py` — 集中設定

所有路徑、API key、模型名稱、RAG 超參數都在這裡。API key 從 `.env` 讀取，不會進版控。

關鍵參數：
- `CHUNK_SIZE = 512` — chunk 大小（本專案用 section-based 所以這個值主要供參考）
- `TOP_K = 5` — 檢索回傳的文件數
- `EMBEDDING_DIMENSION = 768` — embedding 向量維度
- `EMBEDDING_BACKEND` — `"local"`（TF-IDF）或 `"gemini"`
- `GENERATION_BACKEND` — `"gemini"` 或 `"ollama"`
- `OLLAMA_MODEL` / `OLLAMA_BASE_URL` — Ollama 相關設定

### `src/data_loader.py` — 資料載入

- `MedicalReport` dataclass：uid、indication、findings、impression、images
- `load_openi_from_huggingface()` — 從 HuggingFace 下載 Indiana Chest X-ray 資料集
- `load_reports_from_json()` / `save_reports_to_json()` — 本地 JSON 快取，避免重複下載

### `src/chunking.py` — 切 Chunk

為什麼不用固定大小切？因為放射科報告有天然結構。一個 "findings" 段落就是一個完整語意單元，硬切 512 tokens 可能會把 "No pleural effusion." 跟前面的 findings 拆開，導致檢索品質下降。

每份報告產生 ~4 個 chunks，metadata 帶 `uid` 和 `section`，讓最後回答可以引用來源。

### `src/embedding.py` — Embedding 模組

兩個後端共用同一個 public API（`embed_texts()` / `embed_query()`）：

| 後端 | 運作方式 | 使用時機 |
|---|---|---|
| `tfidf` | scikit-learn TF-IDF，pad 到 768 維 | 開發 / 沒有 API key |
| `gemini` | Gemini `text-embedding-004`，每批 100 筆 | 正式環境 |

`get_client()` 是整個專案共用的 Gemini client 入口，評估模組也用同一個。

### `src/vector_store.py` — 向量資料庫

ChromaDB `PersistentClient`，cosine similarity。`index_chunks()` 做一次就好，之後只需 `search()`。有防重複索引的檢查。

### `src/retriever.py` — 檢索

核心邏輯：如果有影像 → Gemini Vision 轉文字 → 合併成 `query + image description` → 向量搜尋。

`RetrievalResult` 包含 documents、metadatas（uid + section）、distances。`.context` 屬性把檢索結果格式化成 prompt 可用的字串。

### `src/generator.py` — 生成回答

支援兩個 LLM 後端：

| 後端 | 運作方式 | 使用時機 |
|---|---|---|
| `gemini` | Gemini 2.0 Flash API，支援多模態（可傳影像） | 預設，品質最好 |
| `ollama` | 本地 Ollama REST API（`/api/generate`），預設模型 `llama3.2:3b` | 完全離線，不需要 API key |

System prompt 限制模型只能根據提供的參考文件回答、必須引用來源、必須附免責聲明。Temperature 0.3 避免模型亂編。

**Fallback 機制**：主要後端失敗 → 嘗試另一個 LLM → 最後退回 `_generate_local()`（直接回傳格式化後的檢索結果）。確保任何網路或 API 異常都不會讓系統完全無法回答。

### `src/rag_pipeline.py` — 管線整合

`MedicalImagingRAG` class 兩個方法：
- `ingest()` — 載入資料 → chunk → 建索引
- `query()` — 檢索 → 生成

優先順序：cache → 本地 JSON → HuggingFace 下載。

---

## 環境設定與執行

### 安裝

```bash
pip install -r requirements.txt
```

主要依賴：`google-genai`、`chromadb`、`gradio`、`Pillow`、`scikit-learn`、`plotly`、`datasets`

### 設定 API Key（可選）

複製 `.env.example` 並填入設定：

```bash
cp .env.example .env
```

完整的環境變數：

```bash
# 必要（使用 Gemini 時）
GEMINI_API_KEY=你的_key          # 從 https://aistudio.google.com/apikey 取得

# Embedding 後端（預設 local）
EMBEDDING_BACKEND=gemini          # "local"（TF-IDF）或 "gemini"

# Generation 後端（預設 gemini）
GENERATION_BACKEND=gemini         # "gemini" 或 "ollama"

# Vision 後端（預設 gemini）
VISION_BACKEND=gemini             # "gemini" 或 "ollama"

# Ollama 設定（僅使用 ollama 後端時需要）
OLLAMA_MODEL=llama3.2:3b          # Ollama 文字生成模型
OLLAMA_VISION_MODEL=llava:7b      # Ollama 影像分析模型
OLLAMA_BASE_URL=http://localhost:11434  # Ollama 伺服器位址
```

沒有 API key 也能跑，差別如下：

| | 有 Gemini API Key | 用 Ollama（離線） | 完全沒有 LLM |
|---|---|---|---|
| Embedding | Gemini（高品質） | TF-IDF | TF-IDF |
| 生成回答 | Gemini Flash | Ollama 本地模型 | 直接回傳檢索原文 |
| 評估指標 | 全部 7 個 | 全部 7 個 | 只有 2 個 retrieval 指標 |

### 使用 Ollama（可選，完全離線）

如果想用本地 LLM 而非 Gemini API：

```bash
# 1. 安裝 Ollama（macOS）
brew install ollama

# 2. 啟動 Ollama 服務
ollama serve

# 3. 下載模型
ollama pull llama3.2:3b   # 文字生成（約 2GB）
ollama pull llava:7b      # 影像分析（約 4.7GB，可選）

# 4. 設定 .env
GENERATION_BACKEND=ollama
OLLAMA_MODEL=llama3.2:3b
VISION_BACKEND=ollama          # 影像分析也用 Ollama（可選）
OLLAMA_VISION_MODEL=llava:7b
```

Ollama 透過 REST API 通訊（`http://localhost:11434/api/generate`），不需要安裝額外的 Python 套件。

### 執行步驟

**1. 建立索引（只需一次）**

```bash
python ingest.py
```

載入 20 份樣本報告 → 切成 ~80 個 chunks → 建立向量索引到 `data/chroma_db/`。跑完後會自動做一個測試查詢確認 pipeline 正常。

要重建索引的話，刪掉 `data/chroma_db/` 再跑一次。

**2. 啟動 Web UI**

```bash
python app.py
```

瀏覽器開 http://localhost:7860。可以打字提問或點範例題，也可以上傳胸部 X 光影像。

**3. 跑評估**

```bash
# 完整評估（需要 GEMINI_API_KEY）
python run_eval.py

# 只跑 retrieval 指標（不需要 API key）
python run_eval.py --retrieval-only

# 安靜模式
python run_eval.py --quiet
```

結果輸出：
- `data/eval_results.json` — 原始分數（所有 40 題的逐題分數 + 彙總統計）
- `data/eval_report.html` — 互動式圖表報告（用瀏覽器開）

**4. 比較不同實驗**

```bash
# 比較兩組實驗
python compare_experiments.py experiments/2026-02-20_baseline experiments/2026-02-23_topk5

# 自訂顯示名稱
python compare_experiments.py exp1/ exp2/ --names "Baseline" "TOP_K=5"

# 指定輸出路徑
python compare_experiments.py exp1/ exp2/ exp3/ -o experiments/my_comparison.html
```

比較報告包含：
- **設定差異表** — 自動找出不同實驗間的設定差異
- **雷達圖疊加** — 所有實驗的 7 項指標在同一張圖上對比
- **分組長條圖** — 每個指標各實驗並排比較
- **分類比較** — 按題目類別的 precision/recall 對比
- **逐題 Delta 表** — 最大改進和退步一目瞭然（按 delta 大小排序）

---

## 評估系統

![Evaluation System](assets/evaluation_system.png)

### 為什麼自己實作而不用 `ragas` library？

`ragas` 綁死 OpenAI / LangChain，會拉進 ~15 個額外依賴。我們的指標本質上就是「結構化 prompt → LLM 回傳 JSON 分數」，自己寫更透明、可控，且可以針對醫療領域調 prompt。

### 指標詳解

**Retrieval 指標（純計算，不需要 API）：**

| 指標 | 公式 | 意義 |
|---|---|---|
| Context Precision | \|retrieved ∩ relevant\| / \|retrieved\| | 檢索到的文件中，有多少比例是真正相關的？低 = 檢索了太多雜訊 |
| Context Recall | \|retrieved ∩ relevant\| / \|relevant\| | 真正相關的文件中，有多少比例被檢索到？低 = 漏掉重要文件 |

比對方式：用 report UID 做 set intersection，每個 golden QA pair 都標註了 `relevant_report_uids`。

**Generation 指標（Gemini LLM 呼叫）：**

| 指標 | Prompt 邏輯 | 意義 |
|---|---|---|
| Faithfulness | 把 context + answer 給 LLM，問「answer 的每個 claim 是否都能在 context 中找到依據？」 | 偵測幻覺（hallucination） |
| Answer Relevancy | 把 question + answer 給 LLM，問「answer 是否切題、回應了問題？」 | 偵測答非所問 |

**LLM Judge（單次 Gemini 呼叫同時評三項）：**

| 標準 | 評什麼 |
|---|---|
| Medical Appropriateness | 醫學用語是否正確？臨床上是否合理？放射科醫師看了會不會搖頭？ |
| Citation Accuracy | 有沒有引用來源？引用的來源跟實際內容對得上嗎？ |
| Answer Completeness | 跟 ground truth 比，重要的 findings 有沒有都提到？ |

所有 LLM 指標都回傳 0.0–1.0 的分數。如果 API 呼叫失敗，分數會是 -1.0 並附上錯誤訊息，不會讓整個評估中斷。

### Golden QA 資料集

40 題，手動根據 `data/sample_reports.json` 中的 20 份報告撰寫：

| 類型 | 難度 | 數量 | 說明 |
|---|---|---|---|
| factual | easy | 10 | 直接問某份報告的內容，例如「CXR_0001 的 findings 是什麼？」 |
| factual | medium | 10 | 不指定報告 UID，需要系統自己找到對的報告，例如「哪份報告有 Kerley B lines？」 |
| comparative | medium | 10 | 需要跨報告比較，例如「比較 CXR_0002 和 CXR_0018 的心臟發現」 |
| diagnostic | hard | 5 | 給臨床情境，要配對到正確報告，例如「車禍病人左側胸痛呼吸困難」 |
| procedural | easy | 5 | 關於管路/裝置，例如「CXR_0019 有哪些管路？位置在哪？」 |

每題都標註了 `relevant_report_uids` 作為 retrieval 評估的 ground truth。

### HTML 報告內容

1. **雷達圖** — 7 個指標一目瞭然
2. **分組長條圖** — 按題目類別分組的分數
3. **摘要表** — 每個指標的 mean / min / max
4. **逐題明細表** — 40 題的個別分數，顏色編碼（綠 ≥ 0.7、黃 ≥ 0.4、紅 < 0.4）

---

## 設計決策與取捨

| 決策 | 原因 |
|---|---|
| Section-based chunking（非固定大小） | 放射科報告有天然語意邊界，切開反而降品質 |
| TF-IDF / Gemini 雙 Embedding 後端 | 開發時不想每次都花 API quota |
| Gemini / Ollama 雙 Generation 後端 | 雲端品質好，本地完全離線不花錢，按需切換 |
| 自動 Fallback chain | 主要 → 備用 LLM → 本地純文字，確保永遠有回答 |
| ChromaDB（內嵌式） | 不需要另外裝 Milvus / Pinecone，部署簡單 |
| Ollama 用 REST API 而非 pip 套件 | 減少依賴，`requests` 本來就有用到 |
| Gemini / Ollama 雙 Vision 後端 | 知識庫是純文字，影像必須先轉成文字才能做向量搜尋。Gemini Vision 品質好，llava 本地免費 |
| Temperature 0.3 | 醫療領域不能讓模型發揮創意 |
| 自製 RAGAS 指標 | 避免 `ragas` 拉進 OpenAI / LangChain 依賴 |
| Score -1.0 表示失敗 | 讓 pipeline 不會因為單一 API 錯誤就中斷 |

---

## 專案結構

```
medical-imaging-rag/
├── app.py                    # Gradio web 介面
├── ingest.py                 # 資料索引 CLI
├── run_eval.py               # 評估 CLI
├── compare_experiments.py    # 跨實驗比較 CLI
├── config.py                 # 集中設定
├── requirements.txt          # Python 依賴
├── .env.example              # 環境變數範本
├── .gitignore                # Git 忽略規則
├── README.md                 # English README
├── README_zh.md              # 這份文件
├── ARCHITECTURE.md           # 架構文件（英文，更技術細節）
│
├── src/
│   ├── data_loader.py        # 資料載入（HuggingFace / JSON）
│   ├── chunking.py           # Section-based chunking
│   ├── embedding.py          # TF-IDF / Gemini embedding
│   ├── vector_store.py       # ChromaDB 向量資料庫
│   ├── retriever.py          # 多模態檢索
│   ├── generator.py          # Gemini / Ollama 生成回答
│   ├── rag_pipeline.py       # Pipeline 整合（ingest + query）
│   └── evaluation/
│       ├── metrics.py        # 4 個 RAGAS 風格指標
│       ├── llm_judge.py      # LLM Judge（3 個評分標準）
│       ├── runner.py         # 評估執行器
│       └── visualization.py  # Plotly HTML 報告產生器
│
└── data/
    ├── sample_reports.json   # 20 份合成放射科報告
    ├── golden_qa.json        # 40 題 QA 測試集
    ├── reports_cache.json    # 資料快取（首次載入後自動產生）
    ├── chroma_db/            # 向量資料庫（ingest 後產生）
    ├── eval_results.json     # 評估結果 JSON（eval 後產生）
    └── eval_report.html      # 互動式報告（eval 後產生）
```
