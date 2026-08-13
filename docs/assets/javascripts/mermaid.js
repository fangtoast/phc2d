/** Initialize Mermaid diagrams in the generated documentation site.
 *
 * Author: fangtoast
 * Email: fangtoast@foxmail.com
 * Copyright (c) 2026 fangtoast. All rights reserved.
 */

document.addEventListener("DOMContentLoaded", () => {
  if (window.mermaid) {
    window.mermaid.initialize({ startOnLoad: true, securityLevel: "strict" });
  }
});
