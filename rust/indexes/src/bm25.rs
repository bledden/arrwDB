/// BM25 full-text search index for hybrid vector+keyword retrieval.
///
/// Implements the Okapi BM25 scoring algorithm with a simple inverted index.
/// Designed to be used alongside HNSW for hybrid search via Reciprocal Rank Fusion.
///
/// BM25 formula:
///   score(q, d) = Σ IDF(qi) * (f(qi,d) * (k1+1)) / (f(qi,d) + k1*(1 - b + b*|d|/avgdl))
///
/// Where:
///   f(qi,d) = term frequency of qi in document d
///   |d| = document length (in tokens)
///   avgdl = average document length
///   k1 = 1.2, b = 0.75 (standard defaults)
///   IDF(qi) = ln((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)

use std::collections::HashMap;
use parking_lot::RwLock;

/// Simple whitespace tokenizer: lowercase, strip punctuation, split on whitespace.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| c.is_whitespace() || c == ',' || c == '.' || c == '!' || c == '?' || c == ';' || c == ':' || c == '"' || c == '\'' || c == '(' || c == ')' || c == '[' || c == ']')
        .filter(|s| s.len() >= 2)  // Skip single-char tokens
        .map(|s| s.to_string())
        .collect()
}

/// Term frequency for a single document.
struct DocTerms {
    /// term → count in this document
    tf: HashMap<String, u32>,
    /// Total tokens in this document
    length: u32,
}

pub struct BM25Index {
    k1: f32,
    b: f32,

    /// Document storage: doc_id → term frequencies
    docs: Vec<DocTerms>,

    /// Inverted index: term → list of (doc_id, term_frequency)
    postings: HashMap<String, Vec<(usize, u32)>>,

    /// Number of documents containing each term
    doc_freq: HashMap<String, u32>,

    /// Total number of documents
    num_docs: usize,

    /// Sum of all document lengths (for avgdl)
    total_length: u64,
}

impl BM25Index {
    pub fn new(k1: f32, b: f32) -> Self {
        Self {
            k1,
            b,
            docs: Vec::new(),
            postings: HashMap::new(),
            doc_freq: HashMap::new(),
            num_docs: 0,
            total_length: 0,
        }
    }

    /// Add a document. Returns its internal index.
    pub fn add_document(&mut self, text: &str) -> usize {
        let tokens = tokenize(text);
        let doc_id = self.docs.len();
        let doc_len = tokens.len() as u32;

        // Count term frequencies
        let mut tf: HashMap<String, u32> = HashMap::new();
        for token in &tokens {
            *tf.entry(token.clone()).or_insert(0) += 1;
        }

        // Update inverted index and doc frequencies
        for (term, count) in &tf {
            self.postings
                .entry(term.clone())
                .or_insert_with(Vec::new)
                .push((doc_id, *count));

            *self.doc_freq.entry(term.clone()).or_insert(0) += 1;
        }

        self.docs.push(DocTerms { tf, length: doc_len });
        self.num_docs += 1;
        self.total_length += doc_len as u64;

        doc_id
    }

    /// Remove a document by zeroing its entry (lazy deletion).
    pub fn remove_document(&mut self, doc_id: usize) {
        if doc_id < self.docs.len() {
            let doc = &self.docs[doc_id];
            self.total_length -= doc.length as u64;
            // We don't remove from postings (lazy) — search skips dead docs
            self.docs[doc_id] = DocTerms {
                tf: HashMap::new(),
                length: 0,
            };
            // Note: num_docs is NOT decremented to keep IDF stable.
            // For exact behavior, call rebuild().
        }
    }

    fn avgdl(&self) -> f32 {
        if self.num_docs == 0 {
            return 1.0;
        }
        self.total_length as f32 / self.num_docs as f32
    }

    fn idf(&self, term: &str) -> f32 {
        let n = self.num_docs as f32;
        let df = *self.doc_freq.get(term).unwrap_or(&0) as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Search for documents matching the query. Returns (doc_id, score) sorted by score desc.
    pub fn search(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let avgdl = self.avgdl();
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for token in &query_tokens {
            let idf = self.idf(token);

            if let Some(postings) = self.postings.get(token) {
                for &(doc_id, tf) in postings {
                    // Skip removed documents
                    if self.docs[doc_id].length == 0 {
                        continue;
                    }

                    let doc_len = self.docs[doc_id].length as f32;
                    let tf_f = tf as f32;

                    let numerator = tf_f * (self.k1 + 1.0);
                    let denominator = tf_f + self.k1 * (1.0 - self.b + self.b * doc_len / avgdl);
                    let score = idf * numerator / denominator;

                    *scores.entry(doc_id).or_insert(0.0) += score;
                }
            }
        }

        let mut results: Vec<(usize, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    pub fn len(&self) -> usize {
        self.num_docs
    }

    pub fn clear(&mut self) {
        self.docs.clear();
        self.postings.clear();
        self.doc_freq.clear();
        self.num_docs = 0;
        self.total_length = 0;
    }
}

/// Reciprocal Rank Fusion: combine ranked lists from vector search and BM25.
/// RRF(d) = Σ 1 / (k + rank_i(d)) for each ranking i
/// k=60 is the standard constant.
pub fn reciprocal_rank_fusion(
    vector_results: &[(usize, f32)],  // (id, distance) — lower = better
    bm25_results: &[(usize, f32)],    // (id, score) — higher = better
    k_rrf: f32,
    limit: usize,
) -> Vec<(usize, f32)> {
    let mut rrf_scores: HashMap<usize, f32> = HashMap::new();

    // Vector results: rank by distance (ascending)
    for (rank, (id, _dist)) in vector_results.iter().enumerate() {
        *rrf_scores.entry(*id).or_insert(0.0) += 1.0 / (k_rrf + rank as f32 + 1.0);
    }

    // BM25 results: already sorted by score descending
    for (rank, (id, _score)) in bm25_results.iter().enumerate() {
        *rrf_scores.entry(*id).or_insert(0.0) += 1.0 / (k_rrf + rank as f32 + 1.0);
    }

    let mut fused: Vec<(usize, f32)> = rrf_scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(limit);
    fused
}

// -------------------------------------------------------------------------
// PyO3 wrapper
// -------------------------------------------------------------------------

use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyclass]
pub struct RustBM25Index {
    inner: RwLock<BM25Index>,
    /// Internal index → String ID mapping
    idx_to_id: RwLock<Vec<String>>,
    /// String ID → internal index
    id_to_idx: RwLock<HashMap<String, usize>>,
}

#[pymethods]
impl RustBM25Index {
    #[new]
    #[pyo3(signature = (k1=1.2, b=0.75))]
    fn new(k1: f32, b: f32) -> Self {
        Self {
            inner: RwLock::new(BM25Index::new(k1, b)),
            idx_to_id: RwLock::new(Vec::new()),
            id_to_idx: RwLock::new(HashMap::new()),
        }
    }

    /// Add a document with its text content.
    fn add_document(&self, doc_id: String, text: String) -> PyResult<()> {
        if self.id_to_idx.read().contains_key(&doc_id) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Document ID {} already exists", doc_id)
            ));
        }

        let idx = self.inner.write().add_document(&text);
        self.id_to_idx.write().insert(doc_id.clone(), idx);
        let mut ids = self.idx_to_id.write();
        if ids.len() <= idx {
            ids.resize(idx + 1, String::new());
        }
        ids[idx] = doc_id;
        Ok(())
    }

    /// Search for documents matching the query text.
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: String,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let results = self.inner.read().search(&query, k);
        let idx_to_id = self.idx_to_id.read();
        let result_list = PyList::empty(py);
        for (idx, score) in results {
            if idx < idx_to_id.len() && !idx_to_id[idx].is_empty() {
                result_list.append(
                    (idx_to_id[idx].as_str(), score)
                        .into_pyobject(py)
                        .unwrap()
                        .into_any()
                        .unbind(),
                )?;
            }
        }
        Ok(result_list)
    }

    /// Hybrid search: combine vector search results with BM25 keyword results
    /// using Reciprocal Rank Fusion.
    ///
    /// vector_results: list of (id_str, distance) from HNSW search
    /// query: text query for BM25
    /// k: number of results to return
    /// vector_weight: not used directly (RRF is rank-based), but controls
    ///   how many vector results to consider (vector_k = k * 2)
    #[pyo3(signature = (vector_results, query, k, rrf_k=60.0))]
    fn hybrid_search<'py>(
        &self,
        py: Python<'py>,
        vector_results: Vec<(String, f32)>,
        query: String,
        k: usize,
        rrf_k: f32,
    ) -> PyResult<Bound<'py, PyList>> {
        let id_to_idx = self.id_to_idx.read();
        let idx_to_id = self.idx_to_id.read();

        // Convert vector results to internal indices
        let vec_results: Vec<(usize, f32)> = vector_results
            .iter()
            .filter_map(|(id, dist)| id_to_idx.get(id).map(|&idx| (idx, *dist)))
            .collect();

        // BM25 search
        let bm25_results = self.inner.read().search(&query, k * 2);

        // Fuse with RRF
        let fused = reciprocal_rank_fusion(&vec_results, &bm25_results, rrf_k, k);

        let result_list = PyList::empty(py);
        for (idx, score) in fused {
            if idx < idx_to_id.len() && !idx_to_id[idx].is_empty() {
                result_list.append(
                    (idx_to_id[idx].as_str(), score)
                        .into_pyobject(py)
                        .unwrap()
                        .into_any()
                        .unbind(),
                )?;
            }
        }
        Ok(result_list)
    }

    fn size(&self) -> usize {
        self.inner.read().len()
    }

    fn clear(&self) {
        self.inner.write().clear();
        self.idx_to_id.write().clear();
        self.id_to_idx.write().clear();
    }
}
