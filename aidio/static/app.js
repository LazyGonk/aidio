const AUDIO_EXTENSIONS = new Set([
    "mp3", "flac", "wav", "m4a", "mp4", "ogg", "opus", "aac", "aiff", "aif", "wma",
]);
const EMBED_EXTENSIONS = new Set(["mp3", "flac", "m4a", "mp4"]);
const DEFAULT_SYSTEM_PROMPT =
    "transcribe lyrics of song and analyze voice and instrumentation characteristics";

document.addEventListener("DOMContentLoaded", () => {
    const el = (id) => document.getElementById(id);

    const dropZone = el("dropZone");
    const fileInput = el("fileInput");
    const folderInput = el("folderInput");
    const browseFilesBtn = el("browseFilesBtn");
    const browseFolderBtn = el("browseFolderBtn");
    const queueWrap = el("queueWrap");
    const queueList = el("queueList");
    const queueSummary = el("queueSummary");
    const clearQueueBtn = el("clearQueueBtn");

    const optionsCard = el("optionsCard");
    const engineInputs = Array.from(document.querySelectorAll('input[name="engine"]'));
    const whisperOptions = el("whisperOptions");
    const gemmaOptions = el("gemmaOptions");
    const deviceSelect = el("deviceSelect");
    const languageSelect = el("languageSelect");
    const modelInput = el("modelInput");
    const backendSelect = el("backendSelect");
    const separateVocals = el("separateVocals");
    const timestamps = el("timestamps");
    const correctionNotes = el("correctionNotes");
    const systemPrompt = el("systemPrompt");
    const autoTxt = el("autoTxt");
    const autoEmbed = el("autoEmbed");
    const runBtn = el("runBtn");
    const runHint = el("runHint");

    const progressCard = el("progressCard");
    const progressLabel = el("progressLabel");
    const progressValue = el("progressValue");
    const progressBar = el("progressBar");
    const progressStage = el("progressStage");
    const cancelJobBtn = el("cancelJobBtn");

    const resultsSection = el("resultsSection");
    const resultsList = el("resultsList");
    const clearResultsBtn = el("clearResultsBtn");
    const downloadZipBtn = el("downloadZipBtn");
    const themeBtn = el("themeBtn");

    const settingsModal = el("settingsModal");
    const settingsBtn = el("settingsBtn");
    const settingsClose = el("settingsClose");
    const settingsCancel = el("settingsCancel");
    const settingsSave = el("settingsSave");
    const settingsStatus = el("settingsStatus");
    const setEndpoint = el("setEndpoint");
    const setApiKey = el("setApiKey");
    const setLlmModel = el("setLlmModel");
    const noticesBtn = el("noticesBtn");

    let queue = [];
    let running = false;
    let progressHideTimer = null;
    let activeJobId = null;
    let activeController = null;
    let cancelRequested = false;

    class CancelledError extends Error {
        constructor(message) {
            super(message || "Cancelled");
            this.name = "CancelledError";
        }
    }

    function applyTheme(theme) {
        document.documentElement.dataset.theme = theme;
        try {
            localStorage.setItem("aidio-theme", theme);
        } catch {
            /* private mode: theme still applies for this page */
        }
    }

    function initTheme() {
        let stored = null;
        try {
            stored = localStorage.getItem("aidio-theme");
        } catch {
            stored = null;
        }
        applyTheme(stored === "day" ? "day" : "neon");
    }

    initTheme();

    if (systemPrompt && !systemPrompt.value.trim()) {
        systemPrompt.value = DEFAULT_SYSTEM_PROMPT;
    }

    function selectedEngine() {
        const chosen = engineInputs.find((input) => input.checked);
        return chosen ? chosen.value : "whisper";
    }

    function engineLabel() {
        return selectedEngine() === "gemma" ? "Gemma multimodal" : "Whisper";
    }

    function isAudioFile(file) {
        const dot = file.name.lastIndexOf(".");
        const ext = dot >= 0 ? file.name.slice(dot + 1).toLowerCase() : "";
        return AUDIO_EXTENSIONS.has(ext) || (file.type || "").startsWith("audio/");
    }

    function fileExtension(name) {
        const dot = name.lastIndexOf(".");
        return dot >= 0 ? name.slice(dot + 1).toLowerCase() : "";
    }

    function fileStem(name) {
        const dot = name.lastIndexOf(".");
        return dot > 0 ? name.slice(0, dot) : name;
    }

    function formatSize(bytes) {
        if (!Number.isFinite(bytes) || bytes <= 0) return "";
        const units = ["B", "KB", "MB", "GB"];
        let value = bytes;
        let unit = 0;
        while (value >= 1024 && unit < units.length - 1) {
            value /= 1024;
            unit += 1;
        }
        return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
    }

    function formatDuration(seconds) {
        if (!Number.isFinite(seconds) || seconds <= 0) return "";
        const total = Math.round(seconds);
        const minutes = Math.floor(total / 60);
        const rest = total % 60;
        return minutes > 0 ? `${minutes}:${String(rest).padStart(2, "0")}` : `${rest}s`;
    }

    function escapeHtml(value) {
        const div = document.createElement("div");
        div.textContent = String(value ?? "");
        return div.innerHTML;
    }

    function errorText(payload, fallback) {
        if (payload && typeof payload.detail === "string") return payload.detail;
        if (payload && Array.isArray(payload.detail)) {
            return payload.detail.map((item) => item.msg || String(item)).join("; ");
        }
        if (payload && payload.error) return payload.error;
        return fallback;
    }

    function addFiles(files) {
        let added = 0;
        for (const file of files) {
            if (!isAudioFile(file)) continue;
            const key = `${file.name}:${file.size}:${file.lastModified}`;
            if (queue.some((item) => item.key === key)) continue;
            queue.push({ key, file, status: "queued", result: null, meta: null, error: null });
            added += 1;
        }
        if (added > 0) {
            syncUi();
            renderQueue();
        }
    }

    function syncUi() {
        const count = queue.length;
        queueWrap.classList.toggle("hidden", count === 0);
        optionsCard.classList.toggle("hidden", count === 0);
        queueSummary.textContent = `${count} file${count === 1 ? "" : "s"} ready`;
        runBtn.disabled = count === 0 || running;
        const verb = selectedEngine() === "gemma" ? "Analyze" : "Transcribe";
        runBtn.textContent = count <= 1 ? verb : `${verb} ${count} files`;
        if (count === 0) {
            runHint.textContent = "";
        } else if (running) {
            runHint.textContent = "Processing…";
        } else {
            runHint.textContent = `${engineLabel()} selected`;
        }
        browseFilesBtn.disabled = running;
        browseFolderBtn.disabled = running;
        clearQueueBtn.disabled = running;
        engineInputs.forEach((input) => { input.disabled = running; });
        [
            deviceSelect, languageSelect, modelInput, backendSelect,
            separateVocals, timestamps, correctionNotes, systemPrompt,
            autoTxt, autoEmbed,
        ].forEach((control) => { control.disabled = running; });
        if (!running) cancelJobBtn.disabled = false;
        syncBackendControls();
        updateResultsActions();
    }

    function updateResultsActions() {
        const done = queue.filter((item) => item.status === "done" && item.result != null).length;
        downloadZipBtn.disabled = done === 0 || running;
        downloadZipBtn.textContent = done > 0
            ? `Download all (.zip) · ${done}`
            : "Download all (.zip)";
    }

    function setCancelVisible(visible) {
        cancelJobBtn.classList.toggle("hidden", !visible);
        if (visible) cancelJobBtn.disabled = false;
    }

    function syncBackendControls() {
        const remote = backendSelect.value === "remote";
        const locked = remote || running;
        if (locked) {
            deviceSelect.disabled = true;
            separateVocals.disabled = true;
        }
    }

    function renderQueue() {
        queueList.innerHTML = "";
        queue.forEach((item) => {
            const li = document.createElement("li");
            li.className = "queue-item";
            li.dataset.status = item.status;

            const icon = document.createElement("span");
            icon.className = "file-icon";
            icon.textContent = fileExtension(item.file.name).slice(0, 3).toUpperCase() || "AUD";

            const name = document.createElement("span");
            name.className = "queue-name";
            name.textContent = item.file.name;
            name.title = item.file.name;

            const size = document.createElement("span");
            size.className = "queue-size";
            size.textContent = formatSize(item.file.size);

            const state = document.createElement("span");
            state.className = "queue-state";
            state.textContent = {
                queued: "Ready",
                running: "Running",
                done: "Done",
                error: "Failed",
                cancelled: "Cancelled",
            }[item.status] || item.status;

            const remove = document.createElement("button");
            remove.type = "button";
            remove.className = "queue-remove";
            remove.textContent = "×";
            remove.title = "Remove from queue";
            remove.disabled = running;
            remove.addEventListener("click", () => {
                queue = queue.filter((entry) => entry !== item);
                syncUi();
                renderQueue();
            });

            li.append(icon, name, size, state, remove);
            queueList.append(li);
        });
    }

    function updateEnginePanels() {
        const gemma = selectedEngine() === "gemma";
        whisperOptions.classList.toggle("hidden", gemma);
        gemmaOptions.classList.toggle("hidden", !gemma);
        if (!running) {
            const count = queue.length;
            const verb = gemma ? "Analyze" : "Transcribe";
            runBtn.textContent = count <= 1 ? verb : `${verb} ${count} files`;
            if (count > 0) runHint.textContent = `${engineLabel()} selected`;
        }
    }

    function setProgress(percent, label, stage, indeterminate) {
        if (progressHideTimer) {
            clearTimeout(progressHideTimer);
            progressHideTimer = null;
        }
        progressCard.classList.remove("hidden");
        progressBar.classList.toggle("indeterminate", Boolean(indeterminate));
        progressBar.style.width = indeterminate ? "40%" : `${Math.max(0, Math.min(100, percent))}%`;
        progressValue.textContent = indeterminate ? "" : `${Math.round(percent)}%`;
        progressLabel.textContent = label || "";
        progressStage.textContent = stage || "";
    }

    function hideProgressLater() {
        progressHideTimer = setTimeout(() => {
            progressCard.classList.add("hidden");
        }, 1400);
    }

    function sleep(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    async function filesFromDataTransfer(dataTransfer) {
        const entries = [];
        if (dataTransfer.items) {
            for (const item of dataTransfer.items) {
                if (typeof item.webkitGetAsEntry !== "function") continue;
                const entry = item.webkitGetAsEntry();
                if (entry) entries.push(entry);
            }
        }
        if (entries.length === 0) return Array.from(dataTransfer.files || []);

        const collected = [];
        async function walk(entry) {
            if (entry.isFile) {
                const file = await new Promise((resolve, reject) => {
                    entry.file(resolve, reject);
                });
                collected.push(file);
                return;
            }
            if (!entry.isDirectory) return;
            const reader = entry.createReader();
            let batch = [];
            do {
                batch = await new Promise((resolve, reject) => {
                    reader.readEntries(resolve, reject);
                });
                for (const child of batch) {
                    await walk(child);
                }
            } while (batch.length > 0);
        }
        for (const entry of entries) {
            await walk(entry);
        }
        return collected;
    }

    function downloadBlob(name, blob) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = name;
        document.body.append(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 30000);
    }

    function downloadText(item, text) {
        const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
        downloadBlob(`${fileStem(item.file.name)}.txt`, blob);
    }

    async function fetchEmbedded(item, text) {
        const form = new FormData();
        form.append("audio_file", item.file, item.file.name);
        form.append("lyrics", text);
        const response = await fetch("/embed", { method: "POST", body: form });
        if (!response.ok) {
            const payload = await response.json().catch(() => ({}));
            throw new Error(errorText(payload, `Could not save lyrics (HTTP ${response.status})`));
        }
        return response.blob();
    }

    async function embedAndDownload(item, text) {
        const blob = await fetchEmbedded(item, text);
        const ext = fileExtension(item.file.name);
        downloadBlob(`${fileStem(item.file.name)}-lyrics.${ext}`, blob);
    }

    async function autoSave(item) {
        if (autoTxt.checked) {
            downloadText(item, item.result);
        }
        if (autoEmbed.checked && EMBED_EXTENSIONS.has(fileExtension(item.file.name))) {
            try {
                await embedAndDownload(item, item.result);
            } catch (error) {
                addInlineError(item.card, `Automatic metadata save failed: ${error.message}`);
            }
        }
    }

    function addInlineError(card, message) {
        if (!card) return;
        const previous = card.querySelector(".inline-error.auto-save");
        if (previous) previous.remove();
        const node = document.createElement("p");
        node.className = "inline-error auto-save";
        node.textContent = message;
        card.append(node);
    }

    async function transcribeFile(item, index, total) {
        const form = new FormData();
        form.append("audio_file", item.file, item.file.name);
        form.append("backend", backendSelect.value);
        form.append("device", deviceSelect.value);
        form.append("model", modelInput.value.trim() || "large-v3-turbo");
        form.append("language", languageSelect.value);
        form.append("separate_vocals", String(separateVocals.checked));
        form.append("timestamps", String(timestamps.checked));

        if (cancelRequested) throw new CancelledError();

        const startResponse = await fetch("/transcribe", { method: "POST", body: form });
        const start = await startResponse.json().catch(() => ({}));
        if (!startResponse.ok || !start.success) {
            throw new Error(errorText(start, `Could not start transcription (HTTP ${startResponse.status})`));
        }

        activeJobId = start.job_id;
        item.jobId = start.job_id;
        if (cancelRequested) {
            fetch(`/jobs/${start.job_id}/cancel`, { method: "POST" }).catch(() => {});
        }
        let job = null;
        try {
            while (true) {
                const response = await fetch(`/jobs/${start.job_id}`);
                if (!response.ok) {
                    const payload = await response.json().catch(() => ({}));
                    throw new Error(errorText(payload, `Lost track of the job (HTTP ${response.status})`));
                }
                job = await response.json();
                const percent = Number.isFinite(job.progress) ? job.progress : 0;
                setProgress(
                    percent,
                    total > 1 ? `File ${index + 1} of ${total}` : "Working",
                    `${item.file.name} · ${job.stage || "Transcribing"}`,
                    job.status === "pending" || job.status === "cancelling",
                );
                if (["done", "error", "cancelled"].includes(job.status)) break;
                await sleep(700);
            }
        } finally {
            activeJobId = null;
        }

        if (job.status === "cancelled") {
            throw new CancelledError("Cancelled");
        }
        if (job.status === "error") {
            throw new Error(job.error || "Transcription failed");
        }
        return { text: job.result.lyrics, meta: job.result };
    }

    async function analyzeFile(item, index, total) {
        if (cancelRequested) throw new CancelledError();
        const form = new FormData();
        form.append("audio_file", item.file, item.file.name);
        form.append("prompt", systemPrompt.value.trim() || DEFAULT_SYSTEM_PROMPT);
        setProgress(
            0,
            total > 1 ? `File ${index + 1} of ${total}` : "Analyzing",
            `${item.file.name} · Analyzing with the LLM`,
            true,
        );
        activeController = new AbortController();
        let response;
        try {
            response = await fetch("/analyze", {
                method: "POST",
                body: form,
                signal: activeController.signal,
            });
        } catch (error) {
            if (error.name === "AbortError") throw new CancelledError("Cancelled");
            throw error;
        } finally {
            activeController = null;
        }
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data.success) {
            throw new Error(errorText(data, `Analysis failed (HTTP ${response.status})`));
        }
        return {
            text: data.analysis,
            meta: {
                engine: "gemma",
                chunks: data.chunk_count,
                duration: data.total_duration,
            },
        };
    }

    function resultSubtitle(item) {
        const meta = item.meta || {};
        if (item.engine === "gemma") {
            const parts = ["Gemma multimodal"];
            if (meta.chunks) parts.push(`${meta.chunks} chunk${meta.chunks === 1 ? "" : "s"}`);
            const duration = formatDuration(meta.duration);
            if (duration) parts.push(duration);
            return parts.join(" · ");
        }
        const parts = ["Whisper"];
        if (meta.device) parts.push(meta.device);
        if (meta.used_demucs) parts.push("Demucs");
        if (meta.corrections) parts.push("LLM polished");
        if (meta.backend === "remote") parts.push("remote API");
        return parts.join(" · ");
    }

    function renderResult(item) {
        const card = document.createElement("article");
        card.className = "card result-card";

        const head = document.createElement("div");
        head.className = "result-head";
        const titleWrap = document.createElement("div");
        titleWrap.className = "result-title";
        const title = document.createElement("h3");
        title.textContent = item.file.name;
        const sub = document.createElement("p");
        sub.className = "result-sub";
        sub.textContent = resultSubtitle(item);
        titleWrap.append(title, sub);
        const badge = document.createElement("span");
        badge.className = "badge badge-ok";
        badge.textContent = "Done";
        head.append(titleWrap, badge);

        const editor = document.createElement("textarea");
        editor.className = "lyrics-editor";
        editor.spellcheck = false;
        editor.value = item.result;
        editor.setAttribute("aria-label", `Lyrics for ${item.file.name}`);

        const actions = document.createElement("div");
        actions.className = "result-actions";

        const saveTxt = document.createElement("button");
        saveTxt.type = "button";
        saveTxt.className = "btn btn-primary";
        saveTxt.textContent = "Download .txt";
        saveTxt.addEventListener("click", () => {
            downloadText(item, editor.value);
            saveTxt.textContent = "Downloaded";
            setTimeout(() => { saveTxt.textContent = "Download .txt"; }, 1400);
        });

        const saveEmbed = document.createElement("button");
        saveEmbed.type = "button";
        saveEmbed.className = "btn";
        saveEmbed.textContent = "Download with lyrics";
        const embeddable = EMBED_EXTENSIONS.has(fileExtension(item.file.name));
        saveEmbed.disabled = !embeddable;
        saveEmbed.title = embeddable
            ? "Download a copy of the audio with the current lyrics embedded"
            : "Metadata embedding supports MP3, FLAC, M4A and MP4";
        saveEmbed.addEventListener("click", async () => {
            saveEmbed.disabled = true;
            saveEmbed.textContent = "Preparing…";
            try {
                await embedAndDownload(item, editor.value);
                saveEmbed.textContent = "Downloaded";
                setTimeout(() => { saveEmbed.textContent = "Download with lyrics"; }, 1400);
            } catch (error) {
                addInlineError(card, error.message);
                saveEmbed.textContent = "Download with lyrics";
            } finally {
                saveEmbed.disabled = !embeddable;
            }
        });

        const copyBtn = document.createElement("button");
        copyBtn.type = "button";
        copyBtn.className = "btn btn-ghost";
        copyBtn.textContent = "Copy";
        copyBtn.addEventListener("click", async () => {
            try {
                await navigator.clipboard.writeText(editor.value);
                copyBtn.textContent = "Copied";
            } catch {
                copyBtn.textContent = "Copy failed";
            }
            setTimeout(() => { copyBtn.textContent = "Copy"; }, 1400);
        });

        actions.append(saveTxt, saveEmbed, copyBtn);

        let analysis = null;
        if (item.engine !== "gemma") {
            const polishBtn = document.createElement("button");
            polishBtn.type = "button";
            polishBtn.className = "btn btn-ghost";
            polishBtn.textContent = "Polish lyrics (LLM)";
            polishBtn.title = "Fix punctuation and misheard words without paraphrasing";
            polishBtn.addEventListener("click", async () => {
                if (polishBtn.disabled) return;
                polishBtn.disabled = true;
                polishBtn.textContent = "Polishing…";
                try {
                    const response = await fetch("/correct", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            lyrics: editor.value,
                            notes: correctionNotes.value.trim(),
                        }),
                    });
                    const data = await response.json().catch(() => ({}));
                    if (!response.ok || !data.success) {
                        throw new Error(errorText(data, `Polish failed (HTTP ${response.status})`));
                    }
                    if (!data.lyrics) throw new Error("The LLM returned no lyrics");
                    editor.value = data.lyrics;
                    polishBtn.textContent = data.changed ? "Polished" : "No changes";
                } catch (error) {
                    addInlineError(card, error.message);
                    polishBtn.textContent = "Polish lyrics (LLM)";
                } finally {
                    polishBtn.disabled = false;
                    setTimeout(() => { polishBtn.textContent = "Polish lyrics (LLM)"; }, 1600);
                }
            });
            actions.append(polishBtn);

            const analyzeBtn = document.createElement("button");
            analyzeBtn.type = "button";
            analyzeBtn.className = "btn btn-ghost";
            analyzeBtn.textContent = "Analyze lyrics (LLM)";
            analyzeBtn.addEventListener("click", async () => {
                if (!analysis) {
                    analysis = document.createElement("details");
                    analysis.className = "llm-analysis";
                    const summary = document.createElement("summary");
                    summary.textContent = "LLM lyric analysis";
                    const body = document.createElement("div");
                    body.className = "analysis-body";
                    analysis.append(summary, body);
                    card.append(analysis);
                }
                analysis.classList.remove("hidden");
                const body = analysis.querySelector(".analysis-body");
                body.innerHTML = '<p class="hint">Analyzing…</p>';
                try {
                    const response = await fetch("/explain", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ lyrics: editor.value }),
                    });
                    const data = await response.json().catch(() => ({}));
                    if (!response.ok || !data.success) {
                        throw new Error(errorText(data, `Analysis failed (HTTP ${response.status})`));
                    }
                    body.innerHTML = renderExplanation(data);
                    analysis.open = true;
                } catch (error) {
                    body.innerHTML = `<p class="inline-error">${escapeHtml(error.message)}</p>`;
                }
            });
            actions.append(analyzeBtn);
        }

        card.append(head, editor, actions);
        resultsList.append(card);
        item.card = card;
        resultsSection.classList.remove("hidden");
    }

    function renderError(item) {
        const card = document.createElement("article");
        card.className = "card result-card error-card";
        const head = document.createElement("div");
        head.className = "result-head";
        const title = document.createElement("h3");
        title.textContent = item.file.name;
        const badge = document.createElement("span");
        badge.className = "badge badge-fail";
        badge.textContent = "Failed";
        head.append(title, badge);
        const message = document.createElement("p");
        message.textContent = item.error || "Unknown error";
        card.append(head, message);
        resultsList.append(card);
        resultsSection.classList.remove("hidden");
    }

    function renderCancelled(item) {
        const card = document.createElement("article");
        card.className = "card result-card cancelled-card";
        const head = document.createElement("div");
        head.className = "result-head";
        const title = document.createElement("h3");
        title.textContent = item.file.name;
        const badge = document.createElement("span");
        badge.className = "badge badge-cancel";
        badge.textContent = "Cancelled";
        head.append(title, badge);
        const message = document.createElement("p");
        message.className = "hint";
        message.textContent = "Stopped before this file finished. Add it to the queue to retry.";
        card.append(head, message);
        resultsList.append(card);
        resultsSection.classList.remove("hidden");
    }

    function renderExplanation(data) {
        const list = (items) => (Array.isArray(items) && items.length
            ? `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
            : "");
        let html = "";
        if (data.title) html += `<p><strong>${escapeHtml(data.title)}</strong></p>`;
        if (data.tone) html += `<p><em>Tone:</em> ${escapeHtml(data.tone)}</p>`;
        if (data.themes) html += `<p><em>Themes:</em></p>${list(data.themes)}`;
        if (data.motifs) html += `<p><em>Motifs:</em></p>${list(data.motifs)}`;
        if (data.characters) html += `<p><em>Characters:</em></p>${list(data.characters)}`;
        if (data.narrative_arc) html += `<p>${escapeHtml(data.narrative_arc)}</p>`;
        if (data.worldbuilding_notes) {
            html += `<p><em>Worldbuilding:</em></p>${list(data.worldbuilding_notes)}`;
        }
        if (Array.isArray(data.suspicious_lines) && data.suspicious_lines.length) {
            html += "<p><em>Possible transcription errors:</em></p><ul>";
            for (const line of data.suspicious_lines) {
                html += `<li>“${escapeHtml(line.line)}” — ${escapeHtml(line.reason || "")}</li>`;
            }
            html += "</ul>";
        }
        return html || '<p class="hint">No analysis returned.</p>';
    }

    async function runAll() {
        if (running || queue.length === 0) return;
        running = true;
        cancelRequested = false;
        activeJobId = null;
        activeController = null;
        setCancelVisible(true);
        const total = queue.length;
        resultsList.innerHTML = "";
        resultsSection.classList.remove("hidden");
        queue.forEach((item) => {
            item.status = "queued";
            item.result = null;
            item.meta = null;
            item.error = null;
            item.card = null;
        });
        syncUi();
        renderQueue();

        const engine = selectedEngine();
        let finished = 0;
        for (let index = 0; index < queue.length; index += 1) {
            if (cancelRequested) break;
            const item = queue[index];
            item.status = "running";
            item.engine = engine;
            renderQueue();
            setProgress(0, total > 1 ? `File ${index + 1} of ${total}` : "Working", item.file.name);
            try {
                const output = engine === "gemma"
                    ? await analyzeFile(item, index, total)
                    : await transcribeFile(item, index, total);
                if (cancelRequested) throw new CancelledError();
                item.result = output.text;
                item.meta = output.meta;
                item.status = "done";
                renderResult(item);
                setProgress(100, total > 1 ? `File ${index + 1} of ${total}` : "Done", `${item.file.name} · Done`);
                await autoSave(item);
            } catch (error) {
                if (error instanceof CancelledError || cancelRequested) {
                    item.status = "cancelled";
                    item.error = "Cancelled";
                    renderCancelled(item);
                } else {
                    item.status = "error";
                    item.error = error.message || String(error);
                    renderError(item);
                }
            }
            finished += 1;
            renderQueue();
            if (cancelRequested) break;
        }

        const wasCancelled = cancelRequested;
        running = false;
        activeJobId = null;
        activeController = null;
        setCancelVisible(false);
        cancelRequested = false;
        syncUi();
        renderQueue();
        if (wasCancelled) {
            setProgress(0, "Cancelled", "Batch stopped");
        } else {
            setProgress(100, "All done", `${finished} file${finished === 1 ? "" : "s"} processed`);
        }
        hideProgressLater();
    }

    function requestCancel() {
        if (!running || cancelRequested) return;
        cancelRequested = true;
        cancelJobBtn.disabled = true;
        progressStage.textContent = "Cancelling after the current step…";
        progressBar.classList.remove("indeterminate");
        if (activeJobId) {
            fetch(`/jobs/${activeJobId}/cancel`, { method: "POST" }).catch(() => {});
        }
        if (activeController) {
            activeController.abort();
        }
    }

    function resultText(item) {
        const editor = item.card ? item.card.querySelector(".lyrics-editor") : null;
        return editor ? editor.value : (item.result || "");
    }

    async function downloadZip() {
        const finished = queue.filter((item) => item.status === "done" && item.result != null);
        if (finished.length === 0) return;
        const files = finished.map((item) => ({
            name: `${fileStem(item.file.name)}${item.engine === "gemma" ? "-analysis" : ""}.txt`,
            text: resultText(item),
        }));
        downloadZipBtn.disabled = true;
        downloadZipBtn.textContent = "Zipping…";
        try {
            const response = await fetch("/bundle/zip", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ files }),
            });
            if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                throw new Error(errorText(payload, `Could not build the zip (HTTP ${response.status})`));
            }
            downloadBlob("aidio-lyrics.zip", await response.blob());
        } catch (error) {
            downloadZipBtn.textContent = "Download failed";
            downloadZipBtn.title = error.message;
            setTimeout(updateResultsActions, 2200);
            return;
        }
        downloadZipBtn.textContent = "Downloaded";
        setTimeout(updateResultsActions, 1400);
    }

    browseFilesBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        fileInput.click();
    });
    browseFolderBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        folderInput.click();
    });

    fileInput.addEventListener("change", () => {
        addFiles(Array.from(fileInput.files || []));
        fileInput.value = "";
    });
    folderInput.addEventListener("change", () => {
        addFiles(Array.from(folderInput.files || []));
        folderInput.value = "";
    });

    dropZone.addEventListener("click", (event) => {
        if (event.target.closest("button")) return;
        fileInput.click();
    });
    dropZone.addEventListener("dragenter", (event) => {
        event.preventDefault();
        dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", (event) => {
        if (event.relatedTarget && dropZone.contains(event.relatedTarget)) return;
        dropZone.classList.remove("dragover");
    });
    dropZone.addEventListener("drop", async (event) => {
        event.preventDefault();
        dropZone.classList.remove("dragover");
        try {
            const files = await filesFromDataTransfer(event.dataTransfer);
            addFiles(files);
        } catch {
            addFiles(Array.from(event.dataTransfer.files || []));
        }
    });
    window.addEventListener("dragover", (event) => event.preventDefault());
    window.addEventListener("drop", (event) => event.preventDefault());

    clearQueueBtn.addEventListener("click", () => {
        if (running) return;
        queue = [];
        syncUi();
        renderQueue();
    });

    engineInputs.forEach((input) => {
        input.addEventListener("change", updateEnginePanels);
    });

    backendSelect.addEventListener("change", () => {
        const remote = backendSelect.value === "remote";
        deviceSelect.disabled = remote || running;
        separateVocals.disabled = remote || running;
    });

    runBtn.addEventListener("click", runAll);
    cancelJobBtn.addEventListener("click", requestCancel);
    downloadZipBtn.addEventListener("click", downloadZip);

    themeBtn.addEventListener("click", () => {
        const next = document.documentElement.dataset.theme === "neon" ? "day" : "neon";
        applyTheme(next);
    });

    clearResultsBtn.addEventListener("click", () => {
        resultsList.innerHTML = "";
        resultsSection.classList.add("hidden");
        queue.forEach((item) => { item.card = null; });
    });

    async function loadSettings() {
        try {
            const response = await fetch("/settings/effective");
            if (!response.ok) return;
            const data = await response.json();
            setEndpoint.value = data.endpoint || "";
            setLlmModel.value = data.llm_model || "";
        } catch {
            /* server defaults stay in place */
        }
    }

    function openSettings() {
        settingsStatus.textContent = "";
        settingsModal.classList.remove("hidden");
        loadSettings();
    }

    function closeSettings() {
        settingsModal.classList.add("hidden");
        setApiKey.value = "";
    }

    settingsBtn.addEventListener("click", openSettings);
    settingsClose.addEventListener("click", closeSettings);
    settingsCancel.addEventListener("click", closeSettings);
    settingsModal.addEventListener("click", (event) => {
        if (event.target === settingsModal) closeSettings();
    });
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !settingsModal.classList.contains("hidden")) {
            closeSettings();
        }
    });

    settingsSave.addEventListener("click", async () => {
        settingsSave.disabled = true;
        settingsStatus.textContent = "Saving…";
        try {
            const response = await fetch("/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    endpoint: setEndpoint.value,
                    api_key: setApiKey.value,
                    llm_model: setLlmModel.value,
                }),
            });
            if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                throw new Error(errorText(payload, `HTTP ${response.status}`));
            }
            settingsStatus.textContent = "Saved for this browser.";
            setApiKey.value = "";
            setTimeout(closeSettings, 700);
        } catch (error) {
            settingsStatus.textContent = `Could not save: ${error.message}`;
        } finally {
            settingsSave.disabled = false;
        }
    });

    noticesBtn.addEventListener("click", () => {
        window.open("/notices", "_blank", "noopener");
    });

    fetch("/device")
        .then((response) => (response.ok ? response.json() : null))
        .then((data) => {
            if (data && data.device) deviceSelect.value = data.device;
        })
        .catch(() => {});

    window.addEventListener("beforeunload", (event) => {
        if (running) {
            event.preventDefault();
            event.returnValue = "";
        }
    });

    syncUi();
    renderQueue();
    updateEnginePanels();
});
