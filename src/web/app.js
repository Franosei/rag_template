const sourceTypeEl = document.getElementById("source-type");
const locationFieldEl = document.getElementById("location-field");
const uploadFieldEl = document.getElementById("upload-field");
const uploadInputEl = document.getElementById("upload-input");
const folderNameEl = document.getElementById("folder-name");
const sourceLocationEl = document.getElementById("source-location");
const sourceStatusEl = document.getElementById("source-status");
const bootstrapButtonEl = document.getElementById("bootstrap-button");
const sourceSubmitEl = document.getElementById("source-submit");
const sourcesListEl = document.getElementById("sources-list");
const folderCountEl = document.getElementById("folder-count");
const chunkCountEl = document.getElementById("chunk-count");
const rootsCountEl = document.getElementById("roots-count");
const llmPillEl = document.getElementById("llm-pill");
const graphPillEl = document.getElementById("graph-pill");
const querySubmitEl = document.getElementById("query-submit");
const questionInputEl = document.getElementById("question-input");
const answerBoxEl = document.getElementById("answer-box");
const answerMetaEl = document.getElementById("answer-meta");
const answerTextEl = document.getElementById("answer-text");
const answerReferencesEl = document.getElementById("answer-references");
const citationsListEl = document.getElementById("citations-list");
const warningBoxEl = document.getElementById("warning-box");
const graphConceptsEl = document.getElementById("graph-concepts");
const graphPathsEl = document.getElementById("graph-paths");
const semanticHitsEl = document.getElementById("semantic-hits");
const hybridHitsEl = document.getElementById("hybrid-hits");
const graphHitsEl = document.getElementById("graph-hits");
const traceSummaryEl = document.getElementById("trace-summary");
const drawerOverlayEl = document.getElementById("drawer-overlay");
const citationDrawerEl = document.getElementById("citation-drawer");
const drawerTitleEl = document.getElementById("drawer-title");
const drawerDocumentEl = document.getElementById("drawer-document");
const drawerPageEl = document.getElementById("drawer-page");
const drawerModeEl = document.getElementById("drawer-mode");
const drawerTermsEl = document.getElementById("drawer-terms");
const drawerEvidenceEl = document.getElementById("drawer-evidence");
const drawerPathEl = document.getElementById("drawer-path");
const drawerCloseEl = document.getElementById("drawer-close");

let citationsByIndex = new Map();

function toggleSourceFields() {
  const uploadMode = sourceTypeEl.value === "inline_upload";
  locationFieldEl.classList.toggle("hidden", uploadMode);
  uploadFieldEl.classList.toggle("hidden", !uploadMode);
}

function selectedFolderIds() {
  return Array.from(document.querySelectorAll("input[data-folder-id]:checked")).map((input) => input.dataset.folderId);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function compactText(value) {
  return String(value ?? "").replace(/\s+/g, " ").trim();
}

function previewText(value, limit = 260) {
  const compact = compactText(value);
  if (compact.length <= limit) {
    return compact;
  }
  return `${compact.slice(0, limit).trim()}...`;
}

function renderInlineMarkdown(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|[^*])\*([^*]+)\*(?!\*)/g, "$1<em>$2</em>");
  return html;
}

function renderMarkdown(markdown) {
  const normalized = String(markdown ?? "").replace(/\r\n/g, "\n").trim();
  if (!normalized) {
    return "";
  }

  const lines = normalized.split("\n");
  const blocks = [];
  let paragraphLines = [];
  let listItems = [];
  let listType = null;

  function flushParagraph() {
    if (!paragraphLines.length) {
      return;
    }
    const content = paragraphLines.join(" ").trim();
    if (content) {
      blocks.push(`<p>${renderInlineMarkdown(content)}</p>`);
    }
    paragraphLines = [];
  }

  function flushList() {
    if (!listItems.length || !listType) {
      listItems = [];
      listType = null;
      return;
    }
    const items = listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("");
    blocks.push(`<${listType}>${items}</${listType}>`);
    listItems = [];
    listType = null;
  }

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      flushParagraph();
      flushList();
      continue;
    }

    if (/^(-{3,}|\*{3,}|_{3,})$/.test(line)) {
      flushParagraph();
      flushList();
      blocks.push('<hr class="answer-divider">');
      continue;
    }

    const headingMatch = line.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      flushParagraph();
      flushList();
      const level = Math.min(headingMatch[1].length + 1, 5);
      blocks.push(`<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
      continue;
    }

    const orderedMatch = line.match(/^\d+\.\s+(.+)$/);
    if (orderedMatch) {
      flushParagraph();
      if (listType && listType !== "ol") {
        flushList();
      }
      listType = "ol";
      listItems.push(orderedMatch[1]);
      continue;
    }

    const unorderedMatch = line.match(/^[-*]\s+(.+)$/);
    if (unorderedMatch) {
      flushParagraph();
      if (listType && listType !== "ul") {
        flushList();
      }
      listType = "ul";
      listItems.push(unorderedMatch[1]);
      continue;
    }

    if (listType) {
      flushList();
    }
    paragraphLines.push(line);
  }

  flushParagraph();
  flushList();
  return blocks.join("");
}

function formatPage(pageNumber) {
  return pageNumber ? `Page ${pageNumber}` : "Page not tagged";
}

function formatMode(mode) {
  const normalized = String(mode || "hybrid")
    .replaceAll("_", " ")
    .trim();
  if (!normalized) {
    return "Hybrid";
  }
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function setStatus(message) {
  sourceStatusEl.textContent = message;
}

function renderHealth(payload) {
  const stats = payload.stats || {};
  folderCountEl.textContent = String(stats.folder_count ?? 0);
  chunkCountEl.textContent = String(stats.indexed_chunks ?? 0);
  rootsCountEl.textContent = String((stats.allowed_local_roots || []).length);
  llmPillEl.textContent = `LLM: ${stats.llm_enabled ? "connected" : "fallback mode"}`;
  graphPillEl.textContent = `Retrieval: ${stats.semantic_retrieval_ready ? "semantic + graph" : "hybrid + graph"}`;
}

function renderSources(payload) {
  const sources = payload.sources || [];
  if (!sources.length) {
    sourcesListEl.innerHTML = `<div class="source-card">No indexed collections yet.</div>`;
    return;
  }

  sourcesListEl.innerHTML = sources
    .map(
      (source) => `
    <div class="source-card">
      <label>
        <input type="checkbox" data-folder-id="${escapeHtml(source.folder_id)}" checked>
        <div>
          <strong>${escapeHtml(source.folder_name)}</strong>
          <small>${escapeHtml(source.summary || "No summary available.")}</small>
          <small>Topics: ${escapeHtml((source.key_topics || []).join(", ") || "n/a")}</small>
        </div>
      </label>
    </div>
  `,
    )
    .join("");
}

function renderList(target, items, renderer, emptyMessage = "No data yet.") {
  if (!items || !items.length) {
    target.innerHTML = `<div class="citation-card">${escapeHtml(emptyMessage)}</div>`;
    return;
  }
  target.innerHTML = items.map(renderer).join("");
}

function renderReferencePills(citations) {
  if (!citations.length) {
    answerReferencesEl.innerHTML = `<div class="citation-card">No answer references available.</div>`;
    return;
  }

  answerReferencesEl.innerHTML = citations
    .map(
      (citation) => `
      <button
        type="button"
        class="reference-pill"
        data-citation-index="${escapeHtml(citation.index)}"
        aria-label="Open evidence reference ${escapeHtml(citation.index)}"
      >
        <span>[${escapeHtml(citation.index)}]</span>
        <span>${escapeHtml(citation.file_name || citation.citation)}</span>
        <small>${escapeHtml(formatPage(citation.page_number))}</small>
      </button>
    `,
    )
    .join("");
}

function renderCitationCard(citation) {
  const evidence = previewText(citation.evidence_text || citation.excerpt || "");
  const matchedTerms = citation.matched_terms || [];
  const firstTerms = matchedTerms.slice(0, 3).join(", ") || "general evidence";

  return `
    <button type="button" class="citation-card clickable-card" data-citation-index="${escapeHtml(citation.index)}">
      <div class="citation-head">
        <div class="citation-title">
          <strong>[${escapeHtml(citation.index)}] ${escapeHtml(citation.citation)}</strong>
          <small>${escapeHtml(citation.file_name || "Unknown document")}</small>
        </div>
      </div>
      <div class="citation-badges">
        <span class="meta-pill">${escapeHtml(formatPage(citation.page_number))}</span>
        <span class="meta-pill">${escapeHtml(formatMode(citation.retrieval_mode))}</span>
        <span class="meta-pill">${escapeHtml(firstTerms)}</span>
      </div>
      <div class="citation-evidence">${escapeHtml(evidence || "No supporting passage available.")}</div>
      <div class="citation-path">${escapeHtml(citation.file_path || "")}</div>
    </button>
  `;
}

function renderAnswer(payload) {
  const citations = payload.citations || [];
  citationsByIndex = new Map(citations.map((citation) => [String(citation.index), citation]));

  answerBoxEl.classList.remove("hidden");
  answerMetaEl.textContent = `Run ID: ${payload.run_id} | ${citations.length} evidence reference${citations.length === 1 ? "" : "s"}`;
  answerTextEl.innerHTML = renderMarkdown(payload.answer || "No answer returned.");

  if (payload.warnings && payload.warnings.length) {
    warningBoxEl.classList.remove("hidden");
    warningBoxEl.textContent = payload.warnings.join(" ");
  } else {
    warningBoxEl.classList.add("hidden");
    warningBoxEl.textContent = "";
  }

  renderReferencePills(citations);
  renderList(citationsListEl, citations, renderCitationCard, "No citations returned.");
  renderRetrievalDiagnostics(payload.retrieval_diagnostics || {});
  renderTrace(payload.trace || {}, payload.selected_folders || []);
}

function renderRetrievalDiagnostics(diagnostics) {
  const concepts = diagnostics.query_concepts || [];
  graphConceptsEl.innerHTML = concepts.length
    ? concepts.map((concept) => `<span class="chip">${escapeHtml(concept)}</span>`).join("")
    : `<span class="chip">No graph concepts</span>`;

  renderList(
    graphPathsEl,
    diagnostics.graph_paths || [],
    (path) => `
      <div class="path-card">
        <strong>${escapeHtml(path.concept)}</strong>
        <small>${escapeHtml(path.folder_name)} -> ${escapeHtml(path.file_name)}</small>
        <small>Chunk ${escapeHtml(path.chunk_id)} | score ${escapeHtml(path.score)}</small>
      </div>
    `,
    "No graph paths yet.",
  );

  renderList(
    semanticHitsEl,
    diagnostics.semantic_hits || [],
    (hit) => `
      <div class="citation-card">
        <strong>${escapeHtml(hit.citation)}</strong>
        <small>${escapeHtml(formatPage(hit.page_number))}</small>
        <small>Score ${escapeHtml(Number(hit.score || 0).toFixed(3))} | ${escapeHtml(formatMode(hit.mode || "semantic"))}</small>
      </div>
    `,
    "No semantic hits available.",
  );

  renderList(
    hybridHitsEl,
    diagnostics.hybrid_hits || [],
    (hit) => `
      <div class="citation-card">
        <strong>${escapeHtml(hit.citation)}</strong>
        <small>${escapeHtml(formatPage(hit.page_number))}</small>
        <small>Score ${escapeHtml(Number(hit.score || 0).toFixed(3))} | ${escapeHtml(formatMode(hit.mode || "hybrid"))}</small>
      </div>
    `,
    "No hybrid hits available.",
  );

  renderList(
    graphHitsEl,
    diagnostics.graph_hits || [],
    (hit) => `
      <div class="citation-card">
        <strong>${escapeHtml(hit.citation)}</strong>
        <small>${escapeHtml(formatPage(hit.page_number))}</small>
        <small>Score ${escapeHtml(Number(hit.score || 0).toFixed(3))} | concepts ${escapeHtml((hit.matched_concepts || []).join(", ") || "n/a")}</small>
      </div>
    `,
    "No graph hits available.",
  );
}

function renderTrace(trace, selectedFolders) {
  const nodes = trace.nodes || [];
  const edges = trace.edges || [];
  const tiles = [
    { label: "Selected folders", value: selectedFolders.length || 0 },
    { label: "Trace nodes", value: nodes.length || 0 },
    { label: "Trace edges", value: edges.length || 0 },
    { label: "Retrieved chunks", value: trace.total_chunks_retrieved || 0 },
  ];

  traceSummaryEl.innerHTML = tiles
    .map(
      (tile) => `
    <div class="trace-tile">
      <span>${escapeHtml(tile.label)}</span>
      <strong>${escapeHtml(tile.value)}</strong>
    </div>
  `,
    )
    .join("");
}

function closeDrawer() {
  drawerOverlayEl.classList.add("hidden");
  citationDrawerEl.classList.add("hidden");
  citationDrawerEl.setAttribute("aria-hidden", "true");
  document.body.classList.remove("drawer-open");
}

function openDrawer(index) {
  const citation = citationsByIndex.get(String(index));
  if (!citation) {
    return;
  }

  drawerTitleEl.textContent = `[${citation.index}] ${citation.citation}`;
  drawerDocumentEl.textContent = citation.file_name || "Unknown document";
  drawerPageEl.textContent = formatPage(citation.page_number);
  drawerModeEl.textContent = formatMode(citation.retrieval_mode);
  drawerEvidenceEl.textContent = citation.evidence_text || citation.excerpt || "No supporting passage available.";
  drawerPathEl.textContent = citation.file_path || "-";

  const terms = citation.matched_terms || [];
  drawerTermsEl.innerHTML = terms.length
    ? terms.map((term) => `<span class="chip">${escapeHtml(term)}</span>`).join("")
    : `<span class="chip">No matched terms captured</span>`;

  drawerOverlayEl.classList.remove("hidden");
  citationDrawerEl.classList.remove("hidden");
  citationDrawerEl.setAttribute("aria-hidden", "false");
  document.body.classList.add("drawer-open");
  drawerCloseEl.focus();
}

function handleCitationClick(event) {
  const trigger = event.target.closest("[data-citation-index]");
  if (!trigger) {
    return;
  }
  openDrawer(trigger.dataset.citationIndex);
}

async function refreshHealth() {
  const response = await fetch("/api/health");
  const payload = await response.json();
  renderHealth(payload);
}

async function refreshSources() {
  const response = await fetch("/api/sources");
  const payload = await response.json();
  renderSources(payload);
  if (payload.stats) {
    renderHealth({ stats: payload.stats });
  }
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      const commaIndex = result.indexOf(",");
      resolve(commaIndex >= 0 ? result.slice(commaIndex + 1) : result);
    };
    reader.onerror = () => reject(reader.error || new Error("Could not read file."));
    reader.readAsDataURL(file);
  });
}

async function ingestSource(sourceTypeOverride = null) {
  const sourceType = sourceTypeOverride || sourceTypeEl.value;
  const payload = {
    source_type: sourceType,
    folder_name: folderNameEl.value || null,
  };

  if (sourceType === "inline_upload") {
    const file = uploadInputEl.files[0];
    if (!file) {
      setStatus("Choose a file first.");
      return;
    }
    payload.filename = file.name;
    payload.content_base64 = await fileToBase64(file);
  } else if (sourceType !== "seed_dataset") {
    payload.location = sourceLocationEl.value;
  }

  setStatus("Working...");
  const response = await fetch("/api/sources", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus(data.detail || "Source ingestion failed.");
    return;
  }

  setStatus(JSON.stringify(data.result || data, null, 2));
  await refreshSources();
}

async function runQuery() {
  const question = questionInputEl.value.trim();
  if (!question) {
    answerBoxEl.classList.remove("hidden");
    answerMetaEl.textContent = "Validation";
    answerTextEl.innerHTML = renderMarkdown("Enter a question first.");
    return;
  }

  closeDrawer();
  answerBoxEl.classList.remove("hidden");
  answerMetaEl.textContent = "Running query";
  answerTextEl.innerHTML = renderMarkdown("Working through semantic, hybrid, and graph retrieval...");
  answerReferencesEl.innerHTML = "";
  citationsListEl.innerHTML = `<div class="citation-card">Building evidence set...</div>`;

  const response = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      folder_ids: selectedFolderIds(),
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    answerMetaEl.textContent = "Query error";
    answerTextEl.innerHTML = renderMarkdown(payload.detail || "Query failed.");
    return;
  }

  renderAnswer(payload);
}

sourceTypeEl.addEventListener("change", toggleSourceFields);
bootstrapButtonEl.addEventListener("click", () => ingestSource("seed_dataset"));
sourceSubmitEl.addEventListener("click", () => ingestSource());
querySubmitEl.addEventListener("click", runQuery);
answerReferencesEl.addEventListener("click", handleCitationClick);
citationsListEl.addEventListener("click", handleCitationClick);
drawerOverlayEl.addEventListener("click", closeDrawer);
drawerCloseEl.addEventListener("click", closeDrawer);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeDrawer();
  }
});

toggleSourceFields();
Promise.all([refreshHealth(), refreshSources()]).catch((error) => {
  setStatus(`Failed to load dashboard: ${error.message}`);
});
