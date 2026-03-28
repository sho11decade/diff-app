(() => {
  function initResultGame() {
    const app = document.getElementById('app');
    if (!app || app.dataset.resultBound === '1') {
      return;
    }

    const dataEl = document.getElementById('result-data');
    const sourceImage = document.getElementById('sourceImage');
    const puzzleImage = document.getElementById('puzzleImage');
    const answerImage = document.getElementById('answerImage');
    const downloadPuzzle = document.getElementById('downloadPuzzle');
    const downloadAnswer = document.getElementById('downloadAnswer');
    const markersLeft = document.getElementById('markers-left');
    const markersRight = document.getElementById('markers-right');
    const stageLeft = document.getElementById('stage-left');
    const stageRight = document.getElementById('stage-right');
    const progress = document.getElementById('progress');
    const meta = document.getElementById('meta');
    const stepMeta = document.getElementById('stepMeta');
    const stepImage = document.getElementById('stepImage');
    const stepIndex = document.getElementById('stepIndex');
    const stepPrev = document.getElementById('stepPrev');
    const stepNext = document.getElementById('stepNext');
    const stepThumbs = document.getElementById('stepThumbs');

    if (
      !dataEl || !sourceImage || !puzzleImage || !answerImage || !downloadPuzzle || !downloadAnswer ||
      !markersLeft || !markersRight || !stageLeft || !stageRight || !progress || !meta ||
      !stepMeta || !stepImage || !stepIndex || !stepPrev || !stepNext || !stepThumbs
    ) {
      return;
    }

    let payload;
    try {
      payload = JSON.parse(dataEl.textContent || '{}');
    } catch {
      return;
    }

    if (!payload.positions || !Array.isArray(payload.positions)) {
      return;
    }

    app.dataset.resultBound = '1';

    const sourceSrc = `data:image/png;base64,${payload.source_image_base64}`;
    const puzzleSrc = `data:image/png;base64,${payload.puzzle_image_base64}`;
    const answerSrc = `data:image/png;base64,${payload.answer_image_base64}`;

    sourceImage.src = sourceSrc;
    puzzleImage.src = puzzleSrc;
    answerImage.src = answerSrc;
    downloadPuzzle.href = puzzleSrc;
    downloadAnswer.href = answerSrc;

    meta.textContent = `difficulty=${payload.difficulty} / differences=${payload.num_differences} / time=${payload.processing_time_ms}ms`;

    const steps = Array.isArray(payload.step_images) ? payload.step_images : [];
    let currentStep = 0;

    function normalizeStepName(name) {
      if (!name) return 'step';
      if (name === 'step_00_source') return '0: source';
      return name.replace('step_', '').replaceAll('_', ' ');
    }

    function renderStep() {
      if (!steps.length) {
        stepMeta.textContent = '処理過程データはありません。';
        stepImage.removeAttribute('src');
        stepIndex.textContent = '';
        stepPrev.disabled = true;
        stepNext.disabled = true;
        return;
      }

      currentStep = Math.max(0, Math.min(currentStep, steps.length - 1));
      const s = steps[currentStep];
      stepImage.src = `data:image/png;base64,${s.image_base64}`;
      stepImage.alt = s.name;
      stepMeta.textContent = 'step_00_source は処理開始時点です。以降は差分適用後の状態を示します。';
      stepIndex.textContent = `${currentStep + 1} / ${steps.length} (${normalizeStepName(s.name)})`;
      stepPrev.disabled = currentStep === 0;
      stepNext.disabled = currentStep === steps.length - 1;

      const thumbButtons = stepThumbs.querySelectorAll('button[data-step-idx]');
      thumbButtons.forEach((btn) => {
        const idx = Number(btn.dataset.stepIdx);
        btn.classList.toggle('active', idx === currentStep);
      });
    }

    function buildStepThumbs() {
      stepThumbs.innerHTML = '';
      if (!steps.length) {
        return;
      }
      steps.forEach((s, idx) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'step-thumb';
        btn.dataset.stepIdx = String(idx);
        btn.title = s.name;

        const img = document.createElement('img');
        img.alt = s.name;
        img.src = `data:image/png;base64,${s.image_base64}`;
        btn.appendChild(img);

        const label = document.createElement('span');
        label.textContent = normalizeStepName(s.name);
        btn.appendChild(label);

        btn.addEventListener('click', () => {
          currentStep = idx;
          renderStep();
        });

        stepThumbs.appendChild(btn);
      });
    }

    const found = new Set();

    function updateProgress() {
      progress.textContent = `発見数: ${found.size} / ${payload.positions.length}`;
    }

    function drawMarker(container, cx, cy, radius) {
      const marker = document.createElement('div');
      marker.className = 'marker';
      marker.style.left = `${cx}px`;
      marker.style.top = `${cy}px`;
      marker.style.width = `${radius * 2}px`;
      marker.style.height = `${radius * 2}px`;
      container.appendChild(marker);
    }

    function findHit(xOrig, yOrig) {
      for (let i = 0; i < payload.positions.length; i += 1) {
        if (found.has(i)) {
          continue;
        }
        const p = payload.positions[i];
        const cx = p.x + p.width / 2;
        const cy = p.y + p.height / 2;
        const hitRadius = Math.max(p.width, p.height) * 0.7;
        const dist = Math.hypot(xOrig - cx, yOrig - cy);
        if (dist <= hitRadius) {
          return { index: i, pos: p };
        }
      }
      return null;
    }

    function onStageClick(event, imageEl, markersA, markersB, rectA, rectB) {
      const rect = imageEl.getBoundingClientRect();
      const xLocal = event.clientX - rect.left;
      const yLocal = event.clientY - rect.top;

      if (xLocal < 0 || yLocal < 0 || xLocal > rect.width || yLocal > rect.height) {
        return;
      }

      const xOrig = xLocal * (payload.image_width / rect.width);
      const yOrig = yLocal * (payload.image_height / rect.height);

      const hit = findHit(xOrig, yOrig);
      if (!hit) {
        return;
      }

      found.add(hit.index);
      const cxA = (hit.pos.x + hit.pos.width / 2) * (rectA.width / payload.image_width);
      const cyA = (hit.pos.y + hit.pos.height / 2) * (rectA.height / payload.image_height);
      const rA = Math.max(hit.pos.width, hit.pos.height) * 0.52 * (rectA.width / payload.image_width);

      const cxB = (hit.pos.x + hit.pos.width / 2) * (rectB.width / payload.image_width);
      const cyB = (hit.pos.y + hit.pos.height / 2) * (rectB.height / payload.image_height);
      const rB = Math.max(hit.pos.width, hit.pos.height) * 0.52 * (rectB.width / payload.image_width);

      drawMarker(markersA, cxA, cyA, Math.max(10, rA));
      drawMarker(markersB, cxB, cyB, Math.max(10, rB));
      updateProgress();
    }

    stageLeft.addEventListener('click', (event) => {
      onStageClick(
        event,
        sourceImage,
        markersLeft,
        markersRight,
        sourceImage.getBoundingClientRect(),
        puzzleImage.getBoundingClientRect(),
      );
    });

    stageRight.addEventListener('click', (event) => {
      onStageClick(
        event,
        puzzleImage,
        markersRight,
        markersLeft,
        puzzleImage.getBoundingClientRect(),
        sourceImage.getBoundingClientRect(),
      );
    });

    stepPrev.addEventListener('click', () => {
      currentStep -= 1;
      renderStep();
    });

    stepNext.addEventListener('click', () => {
      currentStep += 1;
      renderStep();
    });

    buildStepThumbs();
    renderStep();
    updateProgress();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initResultGame, { once: true });
  } else {
    initResultGame();
  }

  document.body.addEventListener('htmx:afterSwap', initResultGame);
})();
