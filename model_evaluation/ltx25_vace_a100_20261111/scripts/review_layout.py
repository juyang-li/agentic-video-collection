"""Offline review navigation and optional VACE examples (no media processing)."""
import html


def show_condition_preview(role, sample_index):
    return sample_index < 2 or role not in ("prepared_input", "overlay_video")


def task_navigation(groups, labels):
    links = "".join(f'<a href="#task-{html.escape(g, quote=True)}">{html.escape(labels[g])}</a>' for g in groups)
    return '<nav class="task-nav" aria-label="任务跳转"><a href="#top">顶部</a>' + links + '</nav>'


NAV_CSS = ".task-nav{position:sticky;top:0;z-index:10;background:#202832;padding:10px;display:flex;flex-wrap:wrap;gap:12px}.task-nav a{color:#9dd6ff}section{scroll-margin-top:110px}html{scroll-behavior:smooth}"
