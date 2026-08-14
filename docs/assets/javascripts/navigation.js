/* Direct-page and expandable-section navigation for the documentation site.
 * Author: fangtoast
 * Email: fangtoast@foxmail.com
 * Copyright (c) 2026 fangtoast. All rights reserved.
 */

(function () {
  "use strict";

  function directChild(parent, selector) {
    return Array.from(parent.children).find(function (child) {
      return child.matches(selector);
    });
  }

  function setExpanded(container, list, button, expanded) {
    list.hidden = !expanded;
    container.classList.toggle("phc-nav-collapsed", !expanded);
    button.setAttribute("aria-expanded", String(expanded));
    button.textContent = expanded ? "▾" : "›";
  }

  function addToggle(container, list, title, expanded) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "phc-nav-toggle";
    button.setAttribute("aria-label", "展开或收起“" + title + "”");
    button.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      setExpanded(
        container,
        list,
        button,
        button.getAttribute("aria-expanded") !== "true"
      );
    });
    if (list.parentElement === container) {
      container.insertBefore(button, list);
    } else {
      container.appendChild(button);
    }
    setExpanded(container, list, button, expanded);
  }

  function prepareTopLevelSections(menu) {
    menu.querySelectorAll("p.caption").forEach(function (caption) {
      var list = caption.nextElementSibling;
      var label = caption.querySelector(".caption-text");
      var landing = list && list.querySelector("a.reference.internal[href]");
      if (!list || list.tagName !== "UL" || !label || !landing) {
        return;
      }

      var title = label.textContent.trim();
      var link = document.createElement("a");
      link.className = "phc-section-link";
      link.href = landing.getAttribute("href");
      link.textContent = title;
      label.replaceWith(link);
      caption.classList.add("phc-nav-caption");
      addToggle(caption, list, title, Boolean(list.querySelector(".current")));
    });
  }

  function prepareNestedSections(menu) {
    menu.querySelectorAll("li").forEach(function (item) {
      var list = directChild(item, "ul");
      var originalLink = directChild(item, "a.reference.internal");
      if (!list || !originalLink || originalLink.hasAttribute("href")) {
        return;
      }

      var landing = list.querySelector("a.reference.internal[href]");
      if (!landing) {
        return;
      }

      var link = originalLink.cloneNode(true);
      link.querySelectorAll(".toctree-expand").forEach(function (icon) {
        icon.remove();
      });
      link.href = landing.getAttribute("href");
      link.classList.add("phc-section-link");
      originalLink.replaceWith(link);

      var title = link.textContent.trim();
      item.classList.add("phc-nav-group");
      addToggle(item, list, title, Boolean(item.querySelector(".current")));
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var menu = document.querySelector(".wy-menu-vertical");
    if (!menu) {
      return;
    }
    prepareTopLevelSections(menu);
    prepareNestedSections(menu);
  });
})();
