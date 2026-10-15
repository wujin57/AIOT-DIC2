/* CSV import helpers. Dataset text is treated as data, never HTML or code. */
(function (global) {
  'use strict';
  const MAX_BYTES = 5 * 1024 * 1024;
  function parseCSV(text) {
    if (new TextEncoder().encode(text).length > MAX_BYTES) throw new Error('CSV 上限為 5 MB。');
    text = text.replace(/^\uFEFF/, '');
    const rows = []; let row = [], cell = '', quoted = false, closed = false;
    const pushCell = () => { row.push(cell); cell = ''; closed = false; };
    const pushRow = () => { pushCell(); if (row.some(value => value.trim() !== '')) rows.push(row); row = []; };
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (quoted) {
        if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; }
        else if (c === '"') { quoted = false; closed = true; }
        else cell += c;
      } else if (c === '"') {
        if (cell.length || closed) throw new Error('CSV 引號格式錯誤。');
        quoted = true;
      } else if (c === ',') pushCell();
      else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; pushRow(); }
      else { if (closed && c.trim()) throw new Error('CSV 引號後有不合法內容。'); if (!closed) cell += c; }
    }
    if (quoted) throw new Error('CSV 有未關閉的引號。');
    if (cell.length || row.length || closed) pushRow();
    if (rows.length < 2) throw new Error('CSV 必須包含欄位標題及至少一列資料。');
    const headers = rows.shift().map(value => value.trim());
    if (headers.length < 2 || headers.length > 100) throw new Error('請提供 2–100 個欄位。');
    if (headers.some(value => !value) || new Set(headers).size !== headers.length) throw new Error('欄位名稱不可空白或重複。');
    if (rows.length > 10000) throw new Error('最多支援 10,000 列資料。');
    rows.forEach((values, i) => { if (values.length !== headers.length) throw new Error(`第 ${i + 2} 列欄位數與標題不符。`); });
    const columns = headers.map((name, i) => {
      const values = rows.map(values => values[i].trim()).filter(Boolean);
      const numeric = values.length > 0 && values.every(value => Number.isFinite(Number(value)));
      return { name, numeric, missing: rows.length - values.length, unique: new Set(values).size };
    });
    return { headers, rows, columns };
  }
  async function readURL(value, signal) {
    let url;
    try { url = new URL(value); } catch (_) { throw new Error('請輸入完整的 CSV 網址（https://…）。'); }
    if (url.protocol !== 'https:' && url.protocol !== 'http:') throw new Error('只支援 HTTP 或 HTTPS 網址。');
    if (url.username || url.password) throw new Error('請使用不含帳號密碼的公開 CSV 網址。');
    let response;
    try { response = await fetch(url.href, { signal, credentials: 'omit' }); }
    catch (error) { if (error.name === 'AbortError') throw error; throw new Error('無法下載：請確認網址、網路與來源的 CORS 設定，或改用上傳 CSV。'); }
    if (!response.ok) throw new Error(`下載失敗：HTTP ${response.status}。`);
    if (Number(response.headers.get('content-length')) > MAX_BYTES) throw new Error('CSV 上限為 5 MB。');
    const reader = response.body.getReader(); const chunks = []; let bytes = 0;
    while (true) { const {done, value} = await reader.read(); if (done) break; bytes += value.length; if (bytes > MAX_BYTES) { await reader.cancel(); throw new Error('CSV 上限為 5 MB。'); } chunks.push(value); }
    const buffer = new Uint8Array(bytes); let offset = 0;
    chunks.forEach(chunk => { buffer.set(chunk, offset); offset += chunk.length; });
    return new TextDecoder('utf-8', {fatal: true}).decode(buffer);
  }
  global.DatasetInput = { parseCSV, readURL, MAX_BYTES };
})(window);
