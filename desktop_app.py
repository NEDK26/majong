# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

"""牌理镜原生桌面客户端。

界面使用 Python 标准库 Tk，在 Windows 上直接显示为本机应用窗口。程序不启动
浏览器、不开放 HTTP 端口，也不向外部网络发送屏幕画面。
"""

from __future__ import annotations

import ctypes
import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from desktop_utils import fit_preview_size, friendly_error_message, shanten_label
from ocr_input import DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR, build_mahjong_soul_templates
from screen_capture import analyze_capture, capture_screen, list_monitors


COLORS = {
    "felt": "#123E39",
    "felt_light": "#1F6257",
    "paper": "#FFFDF8",
    "ivory": "#F4F0E6",
    "ink": "#182522",
    "muted": "#667570",
    "gold": "#D5A545",
    "red": "#B94338",
    "line": "#D9D5C9",
    "canvas": "#0D2926",
    "green_pale": "#E4EEE8",
    "red_pale": "#F7E2DF",
}


class MahjongDesktopApp:
    """单窗口屏幕框选与手牌分析客户端。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("牌理镜 · 本地麻将研究")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1320, max(960, screen_width - 70))
        window_height = min(820, max(640, screen_height - 100))
        left = max(0, (screen_width - window_width) // 2)
        top = max(0, (screen_height - window_height) // 2)
        self.root.geometry(f"{window_width}x{window_height}+{left}+{top}")
        self.root.minsize(min(1040, window_width), min(680, window_height))
        self.root.configure(bg=COLORS["green_pale"])
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.monitors: list[dict[str, int | str]] = []
        self.preview_frame: Any | None = None
        self.preview_photo: Any | None = None
        self.preview_box = (0, 0, 1, 1)
        self.region: dict[str, int] | None = None
        self.drag_start: tuple[int, int] | None = None
        self.running = False
        self.analysis_inflight = False
        self.timer_id: str | None = None
        self.resize_id: str | None = None
        self.overlay: tk.Toplevel | None = None
        self.overlay_main_var = tk.StringVar(value="正在识别手牌…")
        self.overlay_meta_var = tk.StringVar(value="牌理镜 · 本机分析")

        self.monitor_var = tk.StringVar()
        self.count_var = tk.StringVar(value="自动")
        self.interval_var = tk.StringVar(value="1.5 秒")
        self.status_var = tk.StringVar(value="准备中")
        self.region_var = tk.StringVar(value="尚未选择")
        self.captured_var = tk.StringVar(value="--:--:--")
        self.shanten_var = tk.StringVar(value="—")
        self.confidence_var = tk.StringVar(value="等待识别")
        self.tip_var = tk.StringVar(value="在左侧预览上拖动鼠标，框住自己的暗牌。")

        self._configure_styles()
        self._build_layout()
        self._build_overlay()
        self.root.after(80, self._load_monitors)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(
            "Study.TCombobox",
            fieldbackground=COLORS["paper"],
            background=COLORS["paper"],
            foreground=COLORS["ink"],
            bordercolor=COLORS["line"],
            arrowcolor=COLORS["felt"],
            padding=7,
        )
        style.configure(
            "Study.Treeview",
            background=COLORS["paper"],
            fieldbackground=COLORS["paper"],
            foreground=COLORS["ink"],
            rowheight=34,
            borderwidth=0,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Study.Treeview.Heading",
            background=COLORS["green_pale"],
            foreground=COLORS["muted"],
            relief="flat",
            padding=(8, 8),
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.map("Study.Treeview", background=[("selected", "#DCE9E2")])

    def _build_layout(self) -> None:
        header = tk.Frame(self.root, bg=COLORS["green_pale"], padx=26, pady=18)
        header.pack(fill="x")
        brand = tk.Frame(header, bg=COLORS["green_pale"])
        brand.pack(side="left")
        tk.Label(
            brand,
            text="牌理镜",
            bg=COLORS["green_pale"],
            fg=COLORS["felt"],
            font=("Microsoft YaHei UI", 27, "bold"),
        ).pack(anchor="w")
        tk.Label(
            brand,
            text="本地麻将牌理研究 · 画面不离开电脑",
            bg=COLORS["green_pale"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", pady=(3, 0))

        self._button(
            header,
            "校准雀魂牌面",
            self.calibrate_templates,
            COLORS["paper"],
            COLORS["felt"],
            border=COLORS["line"],
        ).pack(side="right", padx=(10, 0))
        tk.Label(
            header,
            text="● 纯本机运行",
            bg=COLORS["green_pale"],
            fg="#28825C",
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(side="right", padx=8)

        body = tk.Frame(self.root, bg=COLORS["green_pale"], padx=18, pady=(0, 18))
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=3, uniform="main")
        body.grid_columnconfigure(1, weight=2, uniform="main")
        body.grid_rowconfigure(0, weight=1)

        self.left_card = self._card(body)
        self.left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        self.right_card = self._card(body)
        self.right_card.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        self._build_capture_panel(self.left_card)
        self._build_analysis_panel(self.right_card)

    def _card(self, parent: tk.Widget) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=COLORS["paper"],
            highlightbackground="#CAD7D0",
            highlightthickness=1,
            padx=18,
            pady=16,
        )

    def _section_title(self, parent: tk.Widget, kicker: str, title: str) -> tk.Frame:
        frame = tk.Frame(parent, bg=COLORS["paper"])
        tk.Label(
            frame,
            text=kicker,
            bg=COLORS["paper"],
            fg=COLORS["red"],
            font=("Consolas", 8, "bold"),
        ).pack(anchor="w")
        tk.Label(
            frame,
            text=title,
            bg=COLORS["paper"],
            fg=COLORS["felt"],
            font=("Microsoft YaHei UI", 14, "bold"),
        ).pack(anchor="w", pady=(1, 0))
        return frame

    def _build_capture_panel(self, parent: tk.Frame) -> None:
        heading = tk.Frame(parent, bg=COLORS["paper"])
        heading.pack(fill="x")
        self._section_title(heading, "画面", "选择研究区域").pack(side="left")
        self.status_label = tk.Label(
            heading,
            textvariable=self.status_var,
            bg="#F4EAD3",
            fg="#79623A",
            padx=10,
            pady=5,
            font=("Microsoft YaHei UI", 8, "bold"),
        )
        self.status_label.pack(side="right", anchor="n")

        controls = tk.Frame(parent, bg=COLORS["paper"], pady=12)
        controls.pack(fill="x")
        self.monitor_control = self._combo(controls, "显示器", self.monitor_var, 17)
        self.monitor_control.pack(side="left", padx=(0, 8))
        self.monitor_control.combobox.bind(  # type: ignore[attr-defined]
            "<<ComboboxSelected>>", lambda _: self.refresh_preview()
        )
        self._combo(
            controls,
            "暗牌张数",
            self.count_var,
            10,
            ["自动", "14", "13", "11", "10", "8", "7", "5", "4", "2", "1"],
        ).pack(side="left", padx=(0, 8))
        self._combo(
            controls,
            "刷新间隔",
            self.interval_var,
            10,
            ["1 秒", "1.5 秒", "2.5 秒"],
        ).pack(side="left", padx=(0, 8))
        preview_actions = tk.Frame(parent, bg=COLORS["paper"])
        preview_actions.pack(fill="x", pady=(0, 9))
        self._button(
            preview_actions,
            "刷新预览",
            self.refresh_preview,
            COLORS["green_pale"],
            COLORS["felt"],
        ).pack(side="left")
        self._button(
            preview_actions,
            "3 秒后预览",
            self.delayed_preview,
            COLORS["paper"],
            COLORS["red"],
            border=COLORS["line"],
        ).pack(side="left", padx=(6, 0))
        tk.Label(
            preview_actions,
            text="单显示器时点击后切回游戏",
            bg=COLORS["paper"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8),
        ).pack(side="left", padx=10)

        self.canvas = tk.Canvas(
            parent,
            bg=COLORS["canvas"],
            highlightbackground=COLORS["felt"],
            highlightthickness=6,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True, pady=(0, 10))
        self.canvas.create_text(
            260,
            190,
            text="正在读取显示器预览…",
            fill="#A9C4BA",
            font=("Microsoft YaHei UI", 11),
            tags="empty",
        )
        self.canvas.bind("<ButtonPress-1>", self._drag_begin)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)
        self.canvas.bind("<Configure>", self._canvas_resized)

        info = tk.Frame(parent, bg=COLORS["paper"])
        info.pack(fill="x")
        tk.Label(
            info,
            text="框选区域",
            bg=COLORS["paper"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8),
        ).pack(side="left")
        tk.Label(
            info,
            textvariable=self.region_var,
            bg=COLORS["paper"],
            fg=COLORS["ink"],
            font=("Consolas", 8, "bold"),
        ).pack(side="left", padx=8)
        self._link_button(info, "恢复底部区域", self.default_bottom_region).pack(
            side="right"
        )

        actions = tk.Frame(parent, bg=COLORS["paper"], pady=(10, 0))
        actions.pack(fill="x")
        self.start_button = self._button(
            actions,
            "开始实时观察",
            self.start_monitoring,
            COLORS["felt"],
            "white",
        )
        self.start_button.pack(side="left", padx=(0, 8))
        self._button(
            actions,
            "分析一次",
            self.analyze_once,
            COLORS["red"],
            "white",
        ).pack(side="left", padx=(0, 8))
        self.stop_button = self._button(
            actions,
            "停止",
            self.stop_monitoring,
            COLORS["green_pale"],
            COLORS["felt"],
        )
        self.stop_button.configure(state="disabled")
        self.stop_button.pack(side="left")

    def _build_analysis_panel(self, parent: tk.Frame) -> None:
        heading = tk.Frame(parent, bg=COLORS["paper"])
        heading.pack(fill="x")
        self._section_title(heading, "判断", "当前牌理").pack(side="left")
        tk.Label(
            heading,
            textvariable=self.captured_var,
            bg=COLORS["paper"],
            fg=COLORS["muted"],
            font=("Consolas", 9),
        ).pack(side="right", anchor="n", pady=8)

        decision = tk.Frame(
            parent,
            bg=COLORS["ivory"],
            highlightbackground=COLORS["line"],
            highlightthickness=1,
            padx=14,
            pady=12,
        )
        decision.pack(fill="x", pady=(16, 14))
        metric = tk.Frame(decision, bg=COLORS["ivory"])
        metric.pack(side="left", fill="y", padx=(0, 20))
        tk.Label(
            metric,
            text="当前向听",
            bg=COLORS["ivory"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8, "bold"),
        ).pack(anchor="w")
        tk.Label(
            metric,
            textvariable=self.shanten_var,
            bg=COLORS["ivory"],
            fg=COLORS["felt"],
            font=("Microsoft YaHei UI", 26, "bold"),
        ).pack(anchor="w", pady=(4, 0))
        recommend = tk.Frame(decision, bg=COLORS["ivory"])
        recommend.pack(side="left", fill="both", expand=True)
        tk.Label(
            recommend,
            text="推荐舍牌",
            bg=COLORS["ivory"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8, "bold"),
        ).pack(anchor="w")
        self.recommend_frame = tk.Frame(recommend, bg=COLORS["ivory"])
        self.recommend_frame.pack(fill="x", pady=(6, 0))

        rack_head = tk.Frame(parent, bg=COLORS["paper"])
        rack_head.pack(fill="x")
        tk.Label(
            rack_head,
            text="识别暗牌",
            bg=COLORS["paper"],
            fg=COLORS["ink"],
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side="left")
        tk.Label(
            rack_head,
            textvariable=self.confidence_var,
            bg=COLORS["paper"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8),
        ).pack(side="right")
        self.rack_frame = tk.Frame(
            parent,
            bg="#D6E1DA",
            highlightbackground="#B9C9C0",
            highlightthickness=1,
            padx=10,
            pady=12,
        )
        self.rack_frame.pack(fill="x", pady=(7, 14))

        self.tip_label = tk.Label(
            parent,
            textvariable=self.tip_var,
            bg=COLORS["green_pale"],
            fg=COLORS["felt"],
            anchor="w",
            justify="left",
            padx=10,
            pady=8,
            wraplength=460,
            font=("Microsoft YaHei UI", 8),
        )
        self.tip_label.pack(fill="x", pady=(0, 12))

        table_head = tk.Frame(parent, bg=COLORS["paper"])
        table_head.pack(fill="x")
        tk.Label(
            table_head,
            text="候选舍牌",
            bg=COLORS["paper"],
            fg=COLORS["ink"],
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side="left")
        tk.Label(
            table_head,
            text="先向听，后有效枚数",
            bg=COLORS["paper"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8),
        ).pack(side="right")

        tree_frame = tk.Frame(parent, bg=COLORS["paper"])
        tree_frame.pack(fill="both", expand=True, pady=(7, 0))
        columns = ("discard", "shanten", "effective", "ukeire")
        self.tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            style="Study.Treeview",
            selectmode="browse",
        )
        self.tree.heading("discard", text="舍牌")
        self.tree.heading("shanten", text="打后向听")
        self.tree.heading("effective", text="有效进张")
        self.tree.heading("ukeire", text="枚数")
        self.tree.column("discard", width=62, anchor="center", stretch=False)
        self.tree.column("shanten", width=82, anchor="center", stretch=False)
        self.tree.column("effective", width=250, anchor="w")
        self.tree.column("ukeire", width=48, anchor="center", stretch=False)
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.tree.tag_configure("best", background="#FBF0D8")

    def _build_overlay(self) -> None:
        """创建实时观察时使用的顶部提示条。"""
        overlay = tk.Toplevel(self.root)
        self.overlay = overlay
        overlay.withdraw()
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        try:
            overlay.attributes("-alpha", 0.94)
        except tk.TclError:
            pass
        overlay.configure(bg=COLORS["gold"])

        content = tk.Frame(overlay, bg=COLORS["felt"], padx=16, pady=9)
        content.pack(fill="both", expand=True, padx=1, pady=(1, 3))
        tk.Label(
            content,
            textvariable=self.overlay_meta_var,
            bg=COLORS["felt"],
            fg="#AFC9BF",
            font=("Microsoft YaHei UI", 8),
        ).pack(anchor="w")
        tk.Label(
            content,
            textvariable=self.overlay_main_var,
            bg=COLORS["felt"],
            fg="white",
            font=("Microsoft YaHei UI", 14, "bold"),
        ).pack(anchor="w", pady=(2, 0))
        overlay.update_idletasks()
        width, height = min(560, overlay.winfo_screenwidth() - 36), 70
        x = max(8, (overlay.winfo_screenwidth() - width) // 2)
        overlay.geometry(f"{width}x{height}+{x}+14")

    def _combo(
        self,
        parent: tk.Widget,
        label: str,
        variable: tk.StringVar,
        width: int,
        values: list[str] | None = None,
    ) -> tk.Frame:
        frame = tk.Frame(parent, bg=COLORS["paper"])
        tk.Label(
            frame,
            text=label,
            bg=COLORS["paper"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8, "bold"),
        ).pack(anchor="w", pady=(0, 3))
        box = ttk.Combobox(
            frame,
            textvariable=variable,
            values=values or [],
            width=width,
            state="readonly",
            style="Study.TCombobox",
        )
        box.pack(fill="x")
        frame.combobox = box  # type: ignore[attr-defined]
        return frame

    def _button(
        self,
        parent: tk.Widget,
        text: str,
        command: Callable[[], None],
        background: str,
        foreground: str,
        *,
        border: str | None = None,
    ) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=background,
            fg=foreground,
            activebackground=background,
            activeforeground=foreground,
            relief="solid" if border else "flat",
            bd=1 if border else 0,
            highlightthickness=0,
            padx=13,
            pady=8,
            cursor="hand2",
            font=("Microsoft YaHei UI", 9, "bold"),
        )

    def _link_button(
        self, parent: tk.Widget, text: str, command: Callable[[], None]
    ) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=COLORS["paper"],
            fg=COLORS["red"],
            activebackground=COLORS["paper"],
            activeforeground=COLORS["red"],
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Microsoft YaHei UI", 8, "bold"),
        )

    def _load_monitors(self) -> None:
        self._run_worker(list_monitors, self._monitors_loaded, self._show_error)

    def _monitors_loaded(self, monitors: list[dict[str, int | str]]) -> None:
        self.monitors = monitors
        names = [
            f"{item['name']} · {item['width']}×{item['height']}" for item in monitors
        ]
        combo = self.monitor_control.combobox  # type: ignore[attr-defined]
        combo.configure(values=names)
        if names:
            self.monitor_var.set(names[0])
            self.refresh_preview()
        else:
            self._show_error(RuntimeError("没有检测到可用显示器。"))

    def _monitor_id(self) -> int:
        selected = self.monitor_var.get()
        for monitor in self.monitors:
            label = f"{monitor['name']} · {monitor['width']}×{monitor['height']}"
            if label == selected:
                return int(monitor["id"])
        raise RuntimeError("请先选择显示器。")

    def _selected_monitor(self) -> dict[str, int | str]:
        monitor_id = self._monitor_id()
        return next(item for item in self.monitors if int(item["id"]) == monitor_id)

    def refresh_preview(self) -> None:
        try:
            monitor_id = self._monitor_id()
        except Exception as exc:
            self._show_error(exc)
            return
        self._set_status("读取画面", "idle")
        self._run_worker(
            lambda: capture_screen(monitor_id)[0],
            self._preview_loaded,
            self._show_error,
        )

    def delayed_preview(self) -> None:
        """给单显示器用户留出切回游戏的时间，再恢复客户端显示。"""
        self._set_status("3 秒后抓取", "idle")
        self.root.iconify()
        self.root.after(3000, self._capture_delayed_preview)

    def _capture_delayed_preview(self) -> None:
        try:
            monitor_id = self._monitor_id()
        except Exception as exc:
            self.root.deiconify()
            self._show_error(exc)
            return
        self._run_worker(
            lambda: capture_screen(monitor_id)[0],
            self._delayed_preview_loaded,
            self._delayed_preview_failed,
        )

    def _delayed_preview_loaded(self, frame: Any) -> None:
        self.root.deiconify()
        self.root.lift()
        self._preview_loaded(frame)

    def _delayed_preview_failed(self, exc: Exception) -> None:
        self.root.deiconify()
        self.root.lift()
        self._show_error(exc)

    def _preview_loaded(self, frame: Any) -> None:
        self.preview_frame = frame
        self.region = None
        self.default_bottom_region()
        self._draw_preview()
        self._set_status("可框选", "idle")

    def _canvas_resized(self, _: tk.Event[Any]) -> None:
        if self.resize_id:
            self.root.after_cancel(self.resize_id)
        self.resize_id = self.root.after(80, self._draw_preview)

    def _draw_preview(self) -> None:
        if self.preview_frame is None:
            return
        try:
            from PIL import Image, ImageTk  # type: ignore[import-not-found]
        except ImportError:
            self._show_error(RuntimeError("缺少 Pillow 图像组件。"))
            return

        canvas_width = max(100, self.canvas.winfo_width() - 12)
        canvas_height = max(100, self.canvas.winfo_height() - 12)
        source_height, source_width = self.preview_frame.shape[:2]
        display_width, display_height = fit_preview_size(
            source_width, source_height, canvas_width, canvas_height
        )
        rgb = self.preview_frame[:, :, ::-1]
        image = Image.fromarray(rgb).resize(
            (display_width, display_height), Image.Resampling.LANCZOS
        )
        self.preview_photo = ImageTk.PhotoImage(image)
        left = (self.canvas.winfo_width() - display_width) // 2
        top = (self.canvas.winfo_height() - display_height) // 2
        self.preview_box = (left, top, display_width, display_height)
        self.canvas.delete("all")
        self.canvas.create_image(left, top, image=self.preview_photo, anchor="nw")
        if self.region:
            x1 = left + self.region["x"] * display_width / source_width
            y1 = top + self.region["y"] * display_height / source_height
            x2 = x1 + self.region["width"] * display_width / source_width
            y2 = y1 + self.region["height"] * display_height / source_height
            self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline=COLORS["gold"],
                width=3,
                dash=(9, 5),
                fill="",
            )
            self.canvas.create_rectangle(
                x1 + 4,
                y1 + 4,
                x2 - 4,
                y2 - 4,
                outline=COLORS["paper"],
                width=1,
            )

    def _canvas_to_source(self, x: int, y: int) -> tuple[int, int]:
        left, top, width, height = self.preview_box
        source_height, source_width = self.preview_frame.shape[:2]
        clamped_x = max(left, min(left + width, x))
        clamped_y = max(top, min(top + height, y))
        return (
            round((clamped_x - left) * source_width / max(1, width)),
            round((clamped_y - top) * source_height / max(1, height)),
        )

    def _drag_begin(self, event: tk.Event[Any]) -> None:
        if self.preview_frame is not None:
            self.drag_start = self._canvas_to_source(event.x, event.y)

    def _drag_move(self, event: tk.Event[Any]) -> None:
        if self.drag_start is None:
            return
        current = self._canvas_to_source(event.x, event.y)
        self.region = {
            "x": min(self.drag_start[0], current[0]),
            "y": min(self.drag_start[1], current[1]),
            "width": abs(current[0] - self.drag_start[0]),
            "height": abs(current[1] - self.drag_start[1]),
        }
        self._update_region_text()
        self._draw_preview()

    def _drag_end(self, _: tk.Event[Any]) -> None:
        self.drag_start = None
        if not self.region or self.region["width"] < 30 or self.region["height"] < 30:
            self.default_bottom_region()

    def default_bottom_region(self) -> None:
        if self.preview_frame is None:
            return
        height, width = self.preview_frame.shape[:2]
        self.region = {
            "x": 0,
            "y": round(height * 0.64),
            "width": width,
            "height": height - round(height * 0.64),
        }
        self._update_region_text()
        self._draw_preview()

    def _update_region_text(self) -> None:
        if not self.region:
            self.region_var.set("尚未选择")
            return
        self.region_var.set(
            f"x {self.region['x']} · y {self.region['y']} · "
            f"{self.region['width']} × {self.region['height']}"
        )

    def _expected_count(self) -> int | None:
        value = self.count_var.get()
        return None if value == "自动" else int(value)

    def _interval_ms(self) -> int:
        return {"1 秒": 1000, "1.5 秒": 1500, "2.5 秒": 2500}.get(
            self.interval_var.get(), 1500
        )

    def start_monitoring(self) -> None:
        self.running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._show_overlay()
        self.root.after(120, self._enter_overlay_mode)

    def _enter_overlay_mode(self) -> None:
        """最小化设置窗口，再把顶部提示条独立恢复到最前。"""
        if not self.running or self.overlay is None:
            return
        self.root.iconify()
        self.overlay.deiconify()
        self.overlay.lift()
        self.overlay.attributes("-topmost", True)
        self.root.after(50, self._enable_overlay_click_through)
        # 等设置窗口完全离开画面后再获取第一帧，避免截到客户端自身。
        self.root.after(450, self.analyze_once)

    def stop_monitoring(self) -> None:
        self.running = False
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self._hide_overlay()
        self._set_status("已停止", "idle")

    def analyze_once(self) -> None:
        if self.analysis_inflight:
            return
        try:
            monitor_id = self._monitor_id()
            if self.region is None:
                self.default_bottom_region()
            region = dict(self.region or {})
            expected_count = self._expected_count()
        except Exception as exc:
            self._show_error(exc)
            return
        self.analysis_inflight = True
        self._set_status("识别中", "running")
        self._run_worker(
            lambda: analyze_capture(monitor_id, region, expected_count),
            self._analysis_loaded,
            self._analysis_failed,
        )

    def _analysis_loaded(self, data: dict[str, Any]) -> None:
        self.analysis_inflight = False
        self._render_analysis(data)
        self._render_overlay(data)
        self._set_status("观察中" if self.running else "已更新", "running" if self.running else "idle")
        if self.running:
            self.timer_id = self.root.after(self._interval_ms(), self.analyze_once)

    def _analysis_failed(self, exc: Exception) -> None:
        self.analysis_inflight = False
        self._show_error(exc)
        self.overlay_main_var.set("未识别到手牌 · 请从任务栏恢复牌理镜检查框选")
        self.overlay_meta_var.set("捕获或识别需要调整")
        if self.running:
            self.timer_id = self.root.after(self._interval_ms(), self.analyze_once)

    def _render_analysis(self, data: dict[str, Any]) -> None:
        self.captured_var.set(str(data["capturedAt"]))
        self.shanten_var.set(shanten_label(int(data["shanten"])))
        confidence = float(data["minimumConfidence"])
        self.confidence_var.set(
            f"最低置信度 {confidence * 100:.0f}% · {len(data['tiles'])} 张"
        )
        if confidence < 0.62:
            self.tip_var.set("部分牌识别置信度偏低，请重新框选或明确暗牌张数。")
            self.tip_label.configure(bg=COLORS["red_pale"], fg=COLORS["red"])
        elif data["mode"] == "draw":
            draws = "、".join(
                f"{item['tile']}×{item['remaining']}"
                for item in data.get("effectiveDraws", [])
            )
            self.tip_var.set(f"有效摸牌：{draws or '—'}")
            self.tip_label.configure(bg=COLORS["green_pale"], fg=COLORS["felt"])
        else:
            self.tip_var.set("排序规则：先降低向听数，再比较理论有效进张总枚数。")
            self.tip_label.configure(bg=COLORS["green_pale"], fg=COLORS["felt"])

        self._clear_children(self.recommend_frame)
        recommendations = data.get("recommendations", [])
        if recommendations:
            for tile in recommendations:
                self._tile_label(self.recommend_frame, tile, large=True).pack(
                    side="left", padx=(0, 6)
                )
        else:
            text = "已经和牌" if data["mode"] == "agari" else "等待摸牌"
            tk.Label(
                self.recommend_frame,
                text=text,
                bg=COLORS["ivory"],
                fg=COLORS["muted"],
                font=("Microsoft YaHei UI", 10, "bold"),
            ).pack(side="left", pady=8)

        self._clear_children(self.rack_frame)
        for tile in data["tiles"]:
            self._tile_label(self.rack_frame, tile).pack(side="left", padx=(0, 4))

        self.tree.delete(*self.tree.get_children())
        best = set(recommendations)
        for candidate in data.get("candidates", []):
            effective = "、".join(
                f"{item['tile']}×{item['remaining']}"
                for item in candidate["effectiveTiles"]
            )
            self.tree.insert(
                "",
                "end",
                values=(
                    candidate["discard"],
                    shanten_label(int(candidate["shanten"])),
                    effective or "—",
                    candidate["ukeire"],
                ),
                tags=("best",) if candidate["discard"] in best else (),
            )

    def _render_overlay(self, data: dict[str, Any]) -> None:
        recommendations = data.get("recommendations", [])
        if recommendations:
            recommended = " / ".join(recommendations)
            best = data.get("candidates", [{}])[0]
            ukeire = best.get("ukeire", "—")
            self.overlay_main_var.set(
                f"{shanten_label(int(data['shanten']))}   建议切 {recommended}   有效 {ukeire} 枚"
            )
        elif data["mode"] == "draw":
            self.overlay_main_var.set(
                f"{shanten_label(int(data['shanten']))}   等待摸牌   有效 {data.get('drawUkeire', 0)} 枚"
            )
        else:
            self.overlay_main_var.set("已经和牌")
        confidence = float(data["minimumConfidence"])
        self.overlay_meta_var.set(
            f"牌理镜 · {data['capturedAt']} · 识别置信度 {confidence * 100:.0f}% · 鼠标可穿透"
        )

    def _show_overlay(self) -> None:
        if self.overlay is None:
            return
        self.overlay_main_var.set("正在识别手牌…")
        self.overlay_meta_var.set("牌理镜 · 本机分析 · 鼠标可穿透")
        self.overlay.deiconify()
        self.overlay.lift()
        self.root.after(50, self._enable_overlay_click_through)

    def _hide_overlay(self) -> None:
        if self.overlay is not None:
            self.overlay.withdraw()

    def _enable_overlay_click_through(self) -> None:
        """Windows 上让顶部提示条不拦截游戏鼠标操作。"""
        if os.name != "nt" or self.overlay is None:
            return
        try:
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            user32.GetParent.argtypes = [wintypes.HWND]
            user32.GetParent.restype = wintypes.HWND
            widget_hwnd = wintypes.HWND(self.overlay.winfo_id())
            hwnd = user32.GetParent(widget_hwnd) or widget_hwnd
            get_style = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
            set_style = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            get_style.argtypes = [wintypes.HWND, ctypes.c_int]
            get_style.restype = ctypes.c_ssize_t
            set_style.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
            set_style.restype = ctypes.c_ssize_t
            extended = get_style(hwnd, -20)
            # WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
            set_style(hwnd, -20, extended | 0x80000 | 0x20 | 0x80 | 0x08000000)
        except Exception:
            # 穿透失败不影响分析；提示条仍然只是顶部小窗口。
            self.overlay_meta_var.set("牌理镜 · 本机分析")

    def _tile_label(self, parent: tk.Widget, tile: str, *, large: bool = False) -> tk.Label:
        suit_color = {
            "m": "#A83A32",
            "p": "#244F82",
            "s": "#177046",
            "z": "#252B29",
        }.get(tile[-1], COLORS["ink"])
        return tk.Label(
            parent,
            text=tile,
            bg=COLORS["paper"],
            fg=suit_color,
            width=3 if large else 3,
            height=2 if large else 2,
            highlightbackground=COLORS["red"] if large else "#BFC8C3",
            highlightthickness=2 if large else 1,
            relief="raised",
            bd=1,
            font=("Consolas", 14 if large else 11, "bold"),
        )

    def calibrate_templates(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="选择雀魂 34 种牌总览图",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.webp *.bmp"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        self._set_status("正在校准", "running")
        self._run_worker(
            lambda: build_mahjong_soul_templates(Path(path)),
            self._calibration_done,
            self._show_error,
        )

    def _calibration_done(self, path: Path) -> None:
        self._set_status("校准完成", "idle")
        messagebox.showinfo(
            "牌面校准完成",
            f"34 种牌模板已保存。\n\n{path}\n\n之后的分析会自动使用这套模板。",
            parent=self.root,
        )

    def _set_status(self, text: str, kind: str) -> None:
        self.status_var.set(text)
        palette = {
            "idle": ("#F4EAD3", "#79623A"),
            "running": ("#DDF0E5", "#24704F"),
            "error": (COLORS["red_pale"], COLORS["red"]),
        }
        background, foreground = palette[kind]
        self.status_label.configure(bg=background, fg=foreground)

    def _show_error(self, exc: Exception) -> None:
        self._set_status("需要处理", "error")
        self.tip_var.set(friendly_error_message(exc))
        self.tip_label.configure(bg=COLORS["red_pale"], fg=COLORS["red"])

    def _run_worker(
        self,
        function: Callable[[], Any],
        success: Callable[[Any], None],
        failure: Callable[[Exception], None],
    ) -> None:
        def work() -> None:
            try:
                result = function()
            except Exception as exc:  # 后台线程必须把错误交回 UI 线程
                try:
                    self.root.after(0, lambda error=exc: failure(error))
                except (RuntimeError, tk.TclError):
                    return
            else:
                try:
                    self.root.after(0, lambda value=result: success(value))
                except (RuntimeError, tk.TclError):
                    return

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _clear_children(frame: tk.Widget) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def close(self) -> None:
        self.running = False
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
        if self.overlay is not None:
            self.overlay.destroy()
        self.root.destroy()


def run_desktop_app() -> None:
    """创建并运行原生客户端。"""
    root = tk.Tk()
    try:
        root.iconname("牌理镜")
        app = MahjongDesktopApp(root)
        if DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR.is_dir():
            app.tip_var.set("已加载本机雀魂牌面模板，可以开始框选。")
        else:
            app.tip_var.set("首次使用请点击顶部“校准雀魂牌面”，选择 34 种牌总览图。")
        root.mainloop()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


def desktop_ui_smoke_test() -> None:
    """供 Windows 打包流程验证 Tk 窗口及全部控件能完成初始化。"""
    root = tk.Tk()
    root.withdraw()
    app = MahjongDesktopApp(root)
    root.update_idletasks()
    app.close()
