/** Configure MathJax for Markdown-authored inline and display equations.
 *
 * Author: fangtoast
 * Email: fangtoast@foxmail.com
 * Copyright (c) 2026 fangtoast. All rights reserved.
 */

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
  },
  options: {
    ignoreHtmlClass: "tex2jax_ignore",
    processHtmlClass: "tex2jax_process",
  },
};
