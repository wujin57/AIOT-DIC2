/* User dataset preview is independent of the precomputed example models. */
(() => {
  'use strict';
  const el = id => document.getElementById(id);
  let dataset, controller, request = 0;
  const problems = {
    startups: {
      url: 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv',
      target: 'Profit',
      description: '50 Startups：根據研發、行政、行銷支出與州別預測公司獲利（Profit）。'
    },
    boston: {
      url: 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv',
      target: 'medv',
      description: 'Boston Housing：根據住宅與區域特徵預測房價（medv）。'
    },
    custom: { description: '自訂 Dataset：輸入 CSV 網址或上傳檔案，再選擇預測目標欄位。' }
  };
  function describe() {
    el('problem-description').textContent = problems[el('problem-type').value].description;
    if (!dataset) return;
    const column = dataset.columns[el('import-target').selectedIndex];
    let message = `${column.name}：${column.numeric ? '數值' : '文字'}，${column.unique} 個不同值。`;
    if (column.missing) message += ` 有 ${column.missing} 筆缺值，訓練前需處理。`;
    el('target-status').textContent = message;
  }
  async function load(read, source) {
    const current = ++request;
    if (controller) controller.abort();
    controller = new AbortController();
    const signal = controller.signal;
    const timer = setTimeout(() => controller && current === request && controller.abort(), 30000);
    dataset = null;
    el('import-preview').replaceChildren();
    el('import-target').replaceChildren(new Option('請先載入 Dataset', ''));
    el('import-target').disabled = true;
    el('target-status').textContent = '';
    el('import-status').textContent = '正在載入 Dataset…';
    try {
      const parsed = DatasetInput.parseCSV(await read(signal));
      if (current !== request) return;
      dataset = parsed;
      el('import-target').replaceChildren(...dataset.headers.map(name => new Option(name, name)));
      el('import-target').disabled = false;
      const target = problems[el('problem-type').value].target;
      el('import-target').value = dataset.headers.includes(target) ? target : dataset.headers.at(-1);
      const table = el('import-preview');
      const head = table.createTHead().insertRow();
      dataset.headers.forEach(name => {
        const cell = document.createElement('th'); cell.scope = 'col'; cell.textContent = name; head.append(cell);
      });
      const body = table.createTBody();
      dataset.rows.slice(0, 20).forEach(values => {
        const row = body.insertRow(); values.forEach(value => { row.insertCell().textContent = value; });
      });
      el('import-status').textContent = `${source}：${dataset.rows.length} 列、${dataset.headers.length} 欄（預覽前 20 列）。`;
      describe();
    } catch (error) {
      if (current === request) el('import-status').textContent = error.name === 'AbortError'
        ? '下載逾時，請重試或上傳 CSV。' : `載入失敗：${error.message}`;
    } finally { clearTimeout(timer); }
  }
  el('dataset-url-form').addEventListener('submit', event => {
    event.preventDefault();
    const url = el('dataset-url').value.trim();
    const preset = Object.keys(problems).find(key => problems[key].url === url);
    el('problem-type').value = preset || 'custom';
    describe();
    window.loadProblem(el('problem-type').value);
    load(signal => DatasetInput.readURL(url, signal), url);
  });
  el('dataset-file').addEventListener('change', event => {
    const file = event.target.files[0];
    if (!file) return;
    el('problem-type').value = 'custom';
    window.loadProblem('custom');
    describe();
    load(() => {
      if (file.size > DatasetInput.MAX_BYTES) throw new Error('CSV 上限為 5 MB。');
      return file.text();
    }, file.name);
  });
  el('problem-type').addEventListener('change', () => {
    const problem = problems[el('problem-type').value];
    window.loadProblem(el('problem-type').value);
    dataset = null;
    describe();
    el('dataset-file').value = '';
    if (problem.url) {
      el('dataset-url').value = problem.url;
      load(signal => DatasetInput.readURL(problem.url, signal), problem.url);
    } else {
      ++request;
      if (controller) controller.abort();
      el('dataset-url').value = '';
      el('import-preview').replaceChildren();
      el('import-target').replaceChildren(new Option('請先載入 Dataset', ''));
      el('import-target').disabled = true;
      el('target-status').textContent = '';
      el('import-status').textContent = '請輸入 CSV 網址或上傳檔案。';
    }
  });
  el('import-target').addEventListener('change', describe);
  describe();
})();
