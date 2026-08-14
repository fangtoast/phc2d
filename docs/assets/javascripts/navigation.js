/* Direct-page and expandable-section navigation for the documentation site.
 * Author: fangtoast
 * Email: fangtoast@foxmail.com
 * Copyright (c) 2026 fangtoast. All rights reserved.
 */

(function () {
  "use strict";

  var animationDurationMs = 180;
  var activeTransitions = new WeakMap();
  var scrollStorageKey = "phc2d-doc-navigation-scroll";

  function directChild(parent, selector) {
    return Array.from(parent.children).find(function (child) {
      return child.matches(selector);
    });
  }

  function reducedMotionRequested() {
    return Boolean(
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    );
  }

  function updateToggle(button, expanded) {
    button.setAttribute("aria-expanded", String(expanded));
  }

  function finishImmediately(container, list, button, expanded) {
    var transition = activeTransitions.get(list);
    if (transition) {
      window.clearTimeout(transition.timer);
      list.removeEventListener("transitionend", transition.onEnd);
      activeTransitions.delete(list);
    }
    list.classList.remove("phc-nav-animating");
    list.style.height = "";
    list.style.opacity = "";
    list.style.willChange = "";
    container.classList.toggle("phc-nav-collapsed", !expanded);
    list.hidden = !expanded;
    updateToggle(button, expanded);
  }

  function animateExpanded(container, list, button, expanded) {
    updateToggle(button, expanded);

    if (reducedMotionRequested()) {
      finishImmediately(container, list, button, expanded);
      return;
    }

    var running = activeTransitions.get(list);
    var startHeight = list.hidden ? 0 : list.getBoundingClientRect().height;
    var startOpacity = list.hidden ? 0.72 : Number(getComputedStyle(list).opacity);
    if (running) {
      startHeight = list.getBoundingClientRect().height;
      startOpacity = Number(getComputedStyle(list).opacity);
      window.clearTimeout(running.timer);
      list.removeEventListener("transitionend", running.onEnd);
      activeTransitions.delete(list);
      list.classList.remove("phc-nav-animating");
    }

    container.classList.remove("phc-nav-collapsed");
    list.hidden = false;
    var targetHeight = expanded ? list.scrollHeight : 0;

    if (Math.abs(startHeight - targetHeight) < 1) {
      finishImmediately(container, list, button, expanded);
      return;
    }

    list.classList.add("phc-nav-animating");
    list.style.height = startHeight + "px";
    list.style.opacity = String(startOpacity);
    list.style.willChange = "height, opacity";
    list.getBoundingClientRect();

    var transition = {
      onEnd: null,
      timer: null
    };
    var complete = function () {
      if (activeTransitions.get(list) !== transition) {
        return;
      }
      activeTransitions.delete(list);
      window.clearTimeout(transition.timer);
      list.removeEventListener("transitionend", transition.onEnd);
      container.classList.toggle("phc-nav-collapsed", !expanded);
      list.hidden = !expanded;
      list.classList.remove("phc-nav-animating");
      list.style.height = "";
      list.style.opacity = "";
      list.style.willChange = "";
    };
    transition.onEnd = function (event) {
      if (event.target === list && event.propertyName === "height") {
        complete();
      }
    };
    transition.timer = window.setTimeout(complete, animationDurationMs + 80);
    activeTransitions.set(list, transition);
    list.addEventListener("transitionend", transition.onEnd);

    window.requestAnimationFrame(function () {
      if (activeTransitions.get(list) !== transition) {
        return;
      }
      list.style.height = targetHeight + "px";
      list.style.opacity = expanded ? "1" : "0.72";
    });
  }

  function addToggle(container, list, title, expanded) {
    var button = document.createElement("button");
    var chevron = document.createElement("span");
    button.type = "button";
    button.className = "phc-nav-toggle";
    button.setAttribute("aria-label", "展开或收起“" + title + "”");
    chevron.className = "phc-nav-chevron";
    chevron.setAttribute("aria-hidden", "true");
    chevron.textContent = "›";
    button.appendChild(chevron);
    button.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      animateExpanded(
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
    finishImmediately(container, list, button, expanded);
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

  function navigationScroller() {
    return document.querySelector(".wy-side-scroll");
  }

  function saveScrollPosition() {
    var scroller = navigationScroller();
    if (!scroller) {
      return;
    }
    try {
      window.sessionStorage.setItem(scrollStorageKey, String(scroller.scrollTop));
    } catch (error) {
      return;
    }
  }

  function restoreScrollPosition() {
    var scroller = navigationScroller();
    if (!scroller) {
      return;
    }
    try {
      var storedPosition = Number(window.sessionStorage.getItem(scrollStorageKey));
      if (!Number.isFinite(storedPosition) || storedPosition < 0) {
        return;
      }
      window.requestAnimationFrame(function () {
        scroller.scrollTop = storedPosition;
      });
    } catch (error) {
      return;
    }
  }

  function initializeNavigation() {
    var menu = document.querySelector(".wy-menu-vertical");
    if (!menu || menu.dataset.phcNavigationReady === "true") {
      return;
    }

    prepareTopLevelSections(menu);
    prepareNestedSections(menu);
    menu.dataset.phcNavigationReady = "true";
    menu.classList.add("phc-nav-ready");
    restoreScrollPosition();
    menu.addEventListener("click", function (event) {
      if (event.target.closest("a[href]")) {
        saveScrollPosition();
      }
    });
    window.addEventListener("pagehide", saveScrollPosition);
  }

  if (document.querySelector(".wy-menu-vertical")) {
    initializeNavigation();
  } else {
    document.addEventListener("DOMContentLoaded", initializeNavigation, {
      once: true
    });
  }
})();
