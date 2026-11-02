/* ============================================================================
   quiz.js — interactive components shared across lessons
   Linked by every lesson. Two behaviours, both progressive enhancements:

     1. Retrieval quizzes. Mark up a .q with .opt buttons; mark the right one
        with data-correct. Optional data-fb on an option (or the .q) supplies
        the explanation shown after answering. A .quiz__score element, if
        present, is updated automatically.
     2. Copy buttons. A .code-head containing a .copy button copies the text of
        the immediately following <pre>.

   Everything degrades to a readable static document when JS is off: the
   stylesheet keys off html.js, which only exists once this file runs.
   ========================================================================= */
(function () {
  document.documentElement.classList.add('js');

  /* ---- retrieval quizzes ------------------------------------------------- */
  document.querySelectorAll('.quiz').forEach(function (quiz) {
    var questions = Array.prototype.slice.call(quiz.querySelectorAll('.q'));
    var score = quiz.querySelector('.quiz__score');
    var answered = 0, right = 0;

    function paint() {
      if (!score) return;
      var s = 'Answered <b>' + answered + '</b> of <b>' + questions.length + '</b>';
      if (answered) s += ' &middot; correct <b>' + right + '</b>';
      score.innerHTML = s;
    }

    questions.forEach(function (q) {
      var opts = Array.prototype.slice.call(q.querySelectorAll('.opt'));
      var fb = q.querySelector('.q__fb');

      opts.forEach(function (opt) {
        opt.addEventListener('click', function () {
          if (q.dataset.done) return;          // one attempt per question
          q.dataset.done = '1';

          var choseRight = opt.hasAttribute('data-correct');
          opts.forEach(function (o) {
            o.disabled = true;
            o.setAttribute('aria-disabled', 'true');
            if (o.hasAttribute('data-correct')) o.classList.add('is-correct');
          });
          if (!choseRight) opt.classList.add('is-wrong');

          answered++;
          if (choseRight) right++;

          if (fb) {
            var why = opt.dataset.fb || q.dataset.fb || '';
            fb.innerHTML = (choseRight ? '<b>Correct.</b> ' : '<b>Not quite.</b> ') + why;
          }
          paint();
        });
      });
    });

    paint();
  });

  /* ---- copy buttons ------------------------------------------------------ */
  document.querySelectorAll('.copy').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var head = btn.closest('.code-head');
      var block = head && head.nextElementSibling;
      if (!block) return;
      var text = block.innerText;
      var label = btn.textContent;

      function show(ok) {
        btn.textContent = ok ? 'Copied' : 'Selected — press Ctrl+C';
        setTimeout(function () { btn.textContent = label; }, 1800);
      }

      try {
        navigator.clipboard.writeText(text).then(
          function () { show(true); },
          function () { select(block, show); }
        );
      } catch (err) {
        select(block, show);   // older engines: no clipboard API at all
      }
    });
  });

  // Fallback: put the snippet in the selection so a manual copy still works.
  function select(block, show) {
    try {
      var range = document.createRange();
      range.selectNodeContents(block);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      show(false);
    } catch (err) {
      show(false);
    }
  }
})();
