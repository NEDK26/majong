# 本程序仅用于立直麻将算法学习研究，禁止用于线上游戏对局辅助。

"""Windows 顶部悬浮牌理条。

默认只显示一条置顶提示，不打开大型控制台。屏幕框选、捕获方式和牌面校准
仅在用户主动点击时出现；程序不抓包、不读取游戏进程、不控制鼠标键盘。
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any, Callable

from desktop_utils import friendly_error_message, shanten_label
from ocr_input import (
    DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR,
    build_mahjong_soul_templates,
    import_labeled_template_folder,
    prepare_ocr_templates,
)
from screen_capture import (
    analyze_capture,
    capture_screen,
    list_monitors,
    release_capture_resources,
)
from tile_utils import tile_name_to_chinese


COLORS = {
    "felt": "#103B36",
    "felt_light": "#1D5A50",
    "paper": "#FFFDF7",
    "muted": "#9BB7AE",
    "gold": "#E0B55A",
    "red": "#D96658",
    "line": "#35675F",
    "panel": "#F3F0E8",
}

BACKEND_LABELS = {
    "自动（DXGI 优先）": "auto",
    "DXGI（游戏画面）": "dxgi",
    "MSS（普通窗口）": "mss",
}
COUNT_VALUES = ["自动", "14", "13", "11", "10", "8", "7", "5", "4", "2", "1"]
INTERVAL_VALUES = ["0.5 秒", "1 秒", "1.5 秒"]


class CompactOverlayApp:
    """只常驻顶部提示条的屏幕 OCR 客户端。"""

    def __init__(self, root: tk.Tk, *, auto_start: bool = True) -> None:
        self.root = root
        self.auto_start = auto_start
        self.monitors: list[dict[str, int | str]] = []
        self.monitor_id = 1
        self.region: dict[str, int] | None = None
        self.backend = "auto"
        self.expected_count: int | None = None
        self.interval_ms = 500
        self.running = False
        self.analysis_inflight = False
        self.templates_ready = False
        self.initializing_templates = False
        self.timer_id: str | None = None
        self.settings_window: tk.Toplevel | None = None
        self.selector_window: tk.Toplevel | None = None
        self.selector_frame: Any | None = None
        self.selector_photo: tk.PhotoImage | None = None
        self.selector_start: tuple[int, int] | None = None
        self.selector_rect: int | None = None
        # DXGI/COM 对象固定在同一个工作线程中创建和使用，避免跨线程失效。
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="capture")

        self.status_var = tk.StringVar(value="正在读取显示器")
        self.main_var = tk.StringVar(value="牌理镜正在准备…")
        self.meta_var = tk.StringVar(value="仅在本机处理画面")
        self.monitor_var = tk.StringVar()
        self.backend_var = tk.StringVar(value="自动（DXGI 优先）")
        self.count_var = tk.StringVar(value="自动")
        self.interval_var = tk.StringVar(value="0.5 秒")

        self._load_preferences()
        self._build_bar()
        self._place_bar()
        self._apply_windows_window_style()
        self.root.after(80, self._load_monitors)

    def _build_bar(self) -> None:
        self.root.title("牌理镜")
        self.root.overrideredirect(True)
        self.root.configure(bg=COLORS["gold"])
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", 0.96)
        except tk.TclError:
            pass

        shell = tk.Frame(self.root, bg=COLORS["felt"], padx=13, pady=8)
        shell.pack(fill="both", expand=True, padx=1, pady=(1, 3))
        shell.grid_columnconfigure(1, weight=1)

        brand = tk.Frame(shell, bg=COLORS["felt"], width=118)
        brand.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 12))
        brand.grid_propagate(False)
        tk.Label(
            brand,
            text="牌理镜",
            bg=COLORS["felt"],
            fg=COLORS["gold"],
            font=("Microsoft YaHei UI", 14, "bold"),
        ).pack(anchor="w")
        self.status_label = tk.Label(
            brand,
            textvariable=self.status_var,
            bg=COLORS["felt"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei UI", 8),
        )
        self.status_label.pack(anchor="w", pady=(1, 0))

        tk.Label(
            shell,
            textvariable=self.main_var,
            width=32,
            bg=COLORS["felt"],
            fg="white",
            anchor="w",
            font=("Microsoft YaHei UI", 13, "bold"),
        ).grid(row=0, column=1, sticky="ew")
        tk.Label(
            shell,
            textvariable=self.meta_var,
            width=48,
            bg=COLORS["felt"],
            fg=COLORS["muted"],
            anchor="w",
            font=("Microsoft YaHei UI", 8),
        ).grid(row=1, column=1, sticky="ew", pady=(2, 0))

        actions = tk.Frame(shell, bg=COLORS["felt"])
        actions.grid(row=0, column=2, rowspan=2, sticky="e", padx=(12, 0))
        self.select_button = self._bar_button(actions, "框选", self.select_region)
        self.select_button.pack(side="left", padx=(0, 5))
        self.pause_button = self._bar_button(actions, "暂停", self.toggle_running)
        self.pause_button.pack(side="left", padx=(0, 5))
        self._bar_button(actions, "设置", self.toggle_settings).pack(
            side="left", padx=(0, 5)
        )
        self._bar_button(actions, "×", self.close, danger=True, width=3).pack(
            side="left"
        )

        for widget in (shell, brand):
            widget.bind("<ButtonPress-1>", self._drag_start)
            widget.bind("<B1-Motion>", self._drag_move)

    def _bar_button(
        self,
        parent: tk.Widget,
        text: str,
        command: Callable[[], None],
        *,
        danger: bool = False,
        width: int = 5,
    ) -> tk.Button:
        color = COLORS["red"] if danger else COLORS["felt_light"]
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            bg=color,
            fg="white",
            activebackground=COLORS["red"] if danger else COLORS["line"],
            activeforeground="white",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=5,
            pady=7,
            font=("Microsoft YaHei UI", 8, "bold"),
        )

    def _place_bar(self) -> None:
        self.root.update_idletasks()
        width = min(820, max(620, self.root.winfo_screenwidth() - 80))
        left = max(8, (self.root.winfo_screenwidth() - width) // 2)
        self.root.geometry(f"{width}x72+{left}+12")

    def _apply_windows_window_style(self) -> None:
        """Windows 上隐藏任务栏图标并避免启动时抢走游戏焦点。"""
        if os.name != "nt":
            return
        self.root.update_idletasks()
        try:
            hwnd = self.root.winfo_id()
            get_long = ctypes.windll.user32.GetWindowLongW
            set_long = ctypes.windll.user32.SetWindowLongW
            ex_style = get_long(hwnd, -20)
            # WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
            set_long(hwnd, -20, ex_style | 0x00000080 | 0x08000000)
        except Exception:
            pass

    def _drag_start(self, event: tk.Event[Any]) -> None:
        self._drag_offset = (
            event.x_root - self.root.winfo_x(),
            event.y_root - self.root.winfo_y(),
        )

    def _drag_move(self, event: tk.Event[Any]) -> None:
        offset = getattr(self, "_drag_offset", (0, 0))
        self.root.geometry(f"+{event.x_root - offset[0]}+{event.y_root - offset[1]}")

    def _load_monitors(self) -> None:
        self._run_worker(list_monitors, self._monitors_loaded, self._show_error)

    def _monitors_loaded(self, monitors: list[dict[str, int | str]]) -> None:
        if not monitors:
            self._show_error(RuntimeError("没有检测到可用显示器。"))
            return
        self.monitors = monitors
        preferred_monitor_id = self.monitor_id
        selected_monitor = next(
            (
                item
                for item in monitors
                if int(item["id"]) == self.monitor_id
            ),
            monitors[0],
        )
        self.monitor_id = int(selected_monitor["id"])
        self.monitor_var.set(self._monitor_label(selected_monitor))
        if self.monitor_id != preferred_monitor_id:
            self.region = None
        if self.region is None:
            self._set_default_region()
        self.status_var.set("准备完成")
        self.main_var.set("默认读取屏幕底部手牌区")
        self.meta_var.set("识别不准时点击“框选”，只圈住自己的暗牌")
        if self.auto_start:
            self.root.after(500, self.start)

    @staticmethod
    def _monitor_label(monitor: dict[str, int | str]) -> str:
        return f"{monitor['name']} · {monitor['width']}×{monitor['height']}"

    def _current_monitor(self) -> dict[str, int | str]:
        for monitor in self.monitors:
            if int(monitor["id"]) == self.monitor_id:
                return monitor
        raise RuntimeError("当前显示器不可用，请在设置中重新选择。")

    def _set_default_region(self) -> None:
        monitor = self._current_monitor()
        height = int(monitor["height"])
        width = int(monitor["width"])
        top = round(height * 0.62)
        self.region = {"x": 0, "y": top, "width": width, "height": height - top}

    @staticmethod
    def _preferences_path() -> Path:
        if getattr(sys, "frozen", False):
            base = Path(
                os.environ.get("LOCALAPPDATA")
                or os.environ.get("APPDATA")
                or Path.home()
            )
            return base / "MahjongStudyAnalyzer" / "settings.json"
        return Path(__file__).resolve().parent / "local_settings.json"

    def _load_preferences(self) -> None:
        try:
            data = json.loads(self._preferences_path().read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        try:
            self.monitor_id = max(1, int(data.get("monitor_id", 1)))
            backend = str(data.get("backend", "auto"))
            self.backend = backend if backend in BACKEND_LABELS.values() else "auto"
            self.backend_var.set(
                next(
                    label
                    for label, value in BACKEND_LABELS.items()
                    if value == self.backend
                )
            )
            count = data.get("expected_count")
            self.expected_count = int(count) if count is not None else None
            self.count_var.set(
                "自动" if self.expected_count is None else str(self.expected_count)
            )
            settings_version = int(data.get("settings_version", 1))
            self.interval_ms = int(data.get("interval_ms", 500))
            # 旧版默认 1.5 秒，容易错过摸牌后的短暂 14 张状态。
            if settings_version < 2:
                self.interval_ms = 500
            self.interval_ms = min(1500, max(500, self.interval_ms))
            self.interval_var.set(
                {500: "0.5 秒", 1000: "1 秒", 1500: "1.5 秒"}.get(
                    self.interval_ms, "0.5 秒"
                )
            )
            region = data.get("region")
            if isinstance(region, dict):
                self.region = {
                    key: max(0 if key in ("x", "y") else 1, int(region[key]))
                    for key in ("x", "y", "width", "height")
                }
        except (KeyError, StopIteration, TypeError, ValueError):
            self.region = None

    def _save_preferences(self) -> None:
        path = self._preferences_path()
        data = {
            "settings_version": 2,
            "monitor_id": self.monitor_id,
            "backend": self.backend,
            "expected_count": self.expected_count,
            "interval_ms": self.interval_ms,
            "region": self.region,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def start(self) -> None:
        if not DEFAULT_MAHJONG_SOUL_TEMPLATE_DIR.is_dir():
            self.running = False
            self.pause_button.configure(text="开始")
            self.status_var.set("等待校准")
            self.main_var.set("首次使用：设置 → 导入拆分好的 34 张牌")
            self.meta_var.set("也可以选择 34 种牌总览图，由程序自动拆分")
            return
        if not self.templates_ready:
            if self.initializing_templates:
                return
            self.initializing_templates = True
            self.running = False
            self.pause_button.configure(text="准备中")
            self.status_var.set("OCR 初始化")
            self.main_var.set("正在预加载 34 种牌…")
            self.meta_var.set("每种牌会提前生成正、倒、左横、右横四方向特征")
            self._run_worker(
                lambda: prepare_ocr_templates(
                    progress=self._template_initialization_progress
                ),
                self._template_initialization_done,
                self._template_initialization_failed,
            )
            return
        self.running = True
        self.pause_button.configure(text="暂停")
        self._schedule_analysis(80)

    def _template_initialization_progress(
        self, tile: str, current: int, total: int
    ) -> None:
        try:
            self.root.after(
                0,
                lambda: self._show_template_initialization_progress(
                    tile, current, total
                ),
            )
        except (RuntimeError, tk.TclError):
            pass

    def _show_template_initialization_progress(
        self, tile: str, current: int, total: int
    ) -> None:
        self.status_var.set(f"初始化 {current}/{total}")
        self.main_var.set(
            f"{tile_name_to_chinese(tile)}已识别（{current}/{total}）"
        )
        self.meta_var.set("正在生成四方向特征并写入本机缓存…")

    def _template_initialization_done(self, result: tuple[int, int]) -> None:
        tile_count, descriptor_count = result
        self.initializing_templates = False
        self.templates_ready = True
        self.status_var.set("初始化完成")
        self.main_var.set(f"{tile_count} 种牌已准备完成")
        self.meta_var.set(f"已缓存 {descriptor_count} 组方向特征，开始读取画面")
        self.start()

    def _template_initialization_failed(self, exc: Exception) -> None:
        self.initializing_templates = False
        self.templates_ready = False
        self.pause_button.configure(text="重试")
        self._show_error(exc)

    def stop(self) -> None:
        self.running = False
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None
        self.pause_button.configure(text="继续")
        self.status_var.set("已暂停")

    def toggle_running(self) -> None:
        if self.running:
            self.stop()
        else:
            self.start()

    def _schedule_analysis(self, delay: int | None = None) -> None:
        if not self.running:
            return
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
        self.timer_id = self.root.after(
            self.interval_ms if delay is None else delay, self.analyze_once
        )

    def analyze_once(self) -> None:
        self.timer_id = None
        if not self.running or self.analysis_inflight:
            return
        self.analysis_inflight = True
        self.status_var.set("识别中")
        monitor_id = self.monitor_id
        region = dict(self.region or {})
        expected_count = self.expected_count
        backend = self.backend
        monitor_geometry = dict(self._current_monitor())
        self._run_worker(
            lambda: analyze_capture(
                monitor_id,
                region,
                expected_count,
                backend=backend,
                monitor_geometry=monitor_geometry,
            ),
            self._analysis_loaded,
            self._analysis_failed,
        )

    def _analysis_loaded(self, data: dict[str, Any]) -> None:
        self.analysis_inflight = False
        capture = data.get("captureRegion", {})
        backend = capture.get("backend", "屏幕")
        confidence = float(data.get("minimumConfidence", 0.0))
        recognized_count = len(data.get("tiles", []))
        corrected_count = int(data.get("correctedTileCount", 0))
        correction_label = f" · 修正{corrected_count}张" if corrected_count else ""
        self.status_var.set(
            f"{backend} · {recognized_count}张 · {confidence * 100:.0f}%"
            f"{correction_label}"
        )

        recommendations = list(data.get("recommendations", []))
        candidates = list(data.get("candidates", []))
        if recommendations and candidates:
            best = candidates[0]
            discard = " / ".join(tile_name_to_chinese(tile) for tile in recommendations)
            self.main_var.set(
                f"建议打：{discard}　·　{shanten_label(int(best['shanten']))}　·　"
                f"有效牌共 {best['ukeire']} 张"
            )
            effective = "、".join(
                f"{tile_name_to_chinese(item['tile'])}×{item['remaining']}"
                for item in best.get("effectiveTiles", [])
            )
            self.meta_var.set(f"有效牌：{effective or '无'}")
        elif data.get("mode") == "draw":
            effective = list(data.get("effectiveDraws", []))
            self.main_var.set(
                f"识别到 {recognized_count} 张，等待摸牌　·　"
                f"当前 {shanten_label(int(data['shanten']))}"
            )
            self.meta_var.set(
                "摸到后有效牌："
                + "、".join(
                    f"{tile_name_to_chinese(item['tile'])}×{item['remaining']}"
                    for item in effective
                )
            )
        elif data.get("mode") == "agari":
            self.main_var.set("当前手牌已经和牌")
            self.meta_var.set("牌理镜仍会继续观察下一次手牌变化")
        else:
            self.main_var.set("已识别手牌，等待出现可舍牌状态")
            self.meta_var.set(
                "手牌 "
                + " ".join(
                    tile_name_to_chinese(tile) for tile in data.get("tiles", [])
                )
            )
        self._schedule_analysis()

    def _analysis_failed(self, exc: Exception) -> None:
        self.analysis_inflight = False
        self.status_var.set("需要调整")
        self.main_var.set(friendly_error_message(exc))
        self.meta_var.set("若能看到游戏但识别失败，请点击“框选”并只圈暗牌")
        self._schedule_analysis(max(self.interval_ms, 2500))

    def toggle_settings(self) -> None:
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.destroy()
            self.settings_window = None
            return
        self._open_settings()

    def _open_settings(self) -> None:
        window = tk.Toplevel(self.root)
        self.settings_window = window
        window.title("牌理镜设置")
        window.configure(bg=COLORS["paper"])
        window.resizable(False, False)
        window.attributes("-topmost", True)
        window.protocol("WM_DELETE_WINDOW", self.toggle_settings)
        width, height = 360, 372
        left = max(8, self.root.winfo_x() + self.root.winfo_width() - width)
        top = self.root.winfo_y() + self.root.winfo_height() + 6
        window.geometry(f"{width}x{height}+{left}+{top}")

        body = tk.Frame(window, bg=COLORS["paper"], padx=18, pady=16)
        body.pack(fill="both", expand=True)
        tk.Label(
            body,
            text="捕获设置",
            bg=COLORS["paper"],
            fg=COLORS["felt"],
            font=("Microsoft YaHei UI", 15, "bold"),
        ).pack(anchor="w")
        tk.Label(
            body,
            text="只在画面读取有问题时修改；自动模式会先尝试 DXGI。",
            bg=COLORS["paper"],
            fg="#62746F",
            font=("Microsoft YaHei UI", 8),
        ).pack(anchor="w", pady=(2, 12))

        self._setting_combo(
            body,
            "显示器",
            self.monitor_var,
            [self._monitor_label(item) for item in self.monitors],
            self._settings_changed,
        )
        self._setting_combo(
            body,
            "捕获方式",
            self.backend_var,
            list(BACKEND_LABELS),
            self._settings_changed,
        )

        row = tk.Frame(body, bg=COLORS["paper"])
        row.pack(fill="x", pady=(0, 12))
        self._setting_combo(
            row,
            "暗牌张数",
            self.count_var,
            COUNT_VALUES,
            self._settings_changed,
            side="left",
        )
        self._setting_combo(
            row,
            "刷新间隔",
            self.interval_var,
            INTERVAL_VALUES,
            self._settings_changed,
            side="right",
        )

        actions = tk.Frame(body, bg=COLORS["paper"])
        actions.pack(fill="x", pady=(3, 0))
        self._settings_button(actions, "重新框选手牌", self.select_region).pack(
            side="left"
        )
        self._settings_button(
            actions, "校准雀魂牌面", self.calibrate_templates, secondary=True
        ).pack(side="right")

        self._settings_button(
            body,
            "导入拆分好的牌 / 实战样本",
            self.import_labeled_samples,
            secondary=True,
        ).pack(fill="x", pady=(10, 0))

        tk.Label(
            body,
            text="提示：游戏建议使用窗口化或无边框窗口；如果游戏以管理员身份运行，"
            "牌理镜也要使用相同权限。",
            wraplength=320,
            justify="left",
            bg=COLORS["paper"],
            fg="#7B6650",
            font=("Microsoft YaHei UI", 8),
        ).pack(anchor="w", pady=(14, 0))

    def _setting_combo(
        self,
        parent: tk.Widget,
        label: str,
        variable: tk.StringVar,
        values: list[str],
        callback: Callable[[Any], None],
        *,
        side: str | None = None,
    ) -> None:
        frame = tk.Frame(parent, bg=COLORS["paper"])
        if side:
            padding = (0, 6) if side == "left" else (6, 0)
            frame.pack(side=side, fill="x", expand=True, padx=padding)
        else:
            frame.pack(fill="x", pady=(0, 10))
        tk.Label(
            frame,
            text=label,
            bg=COLORS["paper"],
            fg="#62746F",
            font=("Microsoft YaHei UI", 8, "bold"),
        ).pack(anchor="w", pady=(0, 3))
        combo = ttk.Combobox(
            frame, textvariable=variable, values=values, state="readonly", width=19
        )
        combo.pack(fill="x")
        combo.bind("<<ComboboxSelected>>", callback)

    def _settings_button(
        self,
        parent: tk.Widget,
        text: str,
        command: Callable[[], None],
        *,
        secondary: bool = False,
    ) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=COLORS["panel"] if secondary else COLORS["felt"],
            fg=COLORS["felt"] if secondary else "white",
            relief="flat",
            bd=0,
            padx=10,
            pady=7,
            cursor="hand2",
            font=("Microsoft YaHei UI", 8, "bold"),
        )

    def _settings_changed(self, _: Any = None) -> None:
        selected = self.monitor_var.get()
        for monitor in self.monitors:
            if self._monitor_label(monitor) == selected:
                changed_monitor = self.monitor_id != int(monitor["id"])
                self.monitor_id = int(monitor["id"])
                if changed_monitor:
                    self._set_default_region()
                break
        self.backend = BACKEND_LABELS.get(self.backend_var.get(), "auto")
        count = self.count_var.get()
        self.expected_count = None if count == "自动" else int(count)
        self.interval_ms = {"0.5 秒": 500, "1 秒": 1000, "1.5 秒": 1500}.get(
            self.interval_var.get(), 500
        )
        self._save_preferences()
        if self.running:
            self._schedule_analysis(100)

    def calibrate_templates(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.settings_window or self.root,
            title="选择雀魂 34 种牌总览图",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.webp"), ("所有文件", "*.*")],
        )
        if not path:
            return
        self.status_var.set("正在校准")
        self.main_var.set("正在从总览图提取 34 种牌…")
        self._run_worker(
            lambda: build_mahjong_soul_templates(path),
            self._calibration_done,
            self._show_error,
        )

    def _calibration_done(self, _: Any) -> None:
        self.templates_ready = False
        self.status_var.set("校准完成")
        self.main_var.set("牌面模板已保存，可以开始读取游戏画面")
        self.meta_var.set("下一步点击“框选”，只圈住自己的暗牌")
        self.start()

    def import_labeled_samples(self) -> None:
        path = filedialog.askdirectory(
            parent=self.settings_window or self.root,
            title="选择拆分牌文件夹（例如：一万.png、八筒.png、北.png）",
        )
        if not path:
            return
        self.status_var.set("正在导入")
        self.main_var.set("正在导入逐张实战牌面样本…")
        self._run_worker(
            lambda: import_labeled_template_folder(path),
            self._sample_import_done,
            self._show_error,
        )

    def _sample_import_done(self, result: tuple[Path, int, int]) -> None:
        self.templates_ready = False
        _, sample_count, tile_count = result
        self.status_var.set("样本已导入")
        self.main_var.set(
            f"已导入 {sample_count} 张样本，覆盖 {tile_count} 种牌"
        )
        self.meta_var.set("样本只保存在本机，不会上传；正在重新识别")
        self.start()

    def select_region(self) -> None:
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.destroy()
            self.settings_window = None
        was_running = self.running
        self.running = False
        self.root.withdraw()
        self.root.after(300, lambda: self._capture_for_selector(was_running))

    def _capture_for_selector(self, resume_after: bool) -> None:
        monitor_geometry = dict(self._current_monitor())
        self._run_worker(
            lambda: capture_screen(
                self.monitor_id,
                backend=self.backend,
                monitor_geometry=monitor_geometry,
            )[0],
            lambda frame: self._open_selector(frame, resume_after),
            lambda exc: self._selector_failed(exc, resume_after),
        )

    def _open_selector(self, frame: Any, resume_after: bool) -> None:
        self.selector_frame = frame
        monitor = self._current_monitor()
        height, width = frame.shape[:2]
        try:
            import cv2  # type: ignore[import-not-found]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            self._selector_failed(exc, resume_after)
            return
        ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes()

        selector = tk.Toplevel(self.root)
        self.selector_window = selector
        selector.overrideredirect(True)
        selector.attributes("-topmost", True)
        selector.geometry(
            f"{int(monitor['width'])}x{int(monitor['height'])}"
            f"+{int(monitor['left'])}+{int(monitor['top'])}"
        )
        canvas = tk.Canvas(selector, width=width, height=height, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        self.selector_photo = tk.PhotoImage(data=ppm, format="PPM")
        canvas.create_image(0, 0, image=self.selector_photo, anchor="nw")
        canvas.create_rectangle(0, 0, width, 54, fill="#102E2B", outline="")
        canvas.create_text(
            22,
            27,
            text="拖动框住自己的暗牌　·　Esc 取消",
            anchor="w",
            fill="white",
            font=("Microsoft YaHei UI", 14, "bold"),
        )
        canvas.bind("<ButtonPress-1>", lambda event: self._selector_press(canvas, event))
        canvas.bind("<B1-Motion>", lambda event: self._selector_drag(canvas, event))
        canvas.bind(
            "<ButtonRelease-1>",
            lambda event: self._selector_release(canvas, event, resume_after),
        )
        selector.bind("<Escape>", lambda _: self._close_selector(resume_after))
        selector.focus_force()

    def _selector_press(self, canvas: tk.Canvas, event: tk.Event[Any]) -> None:
        self.selector_start = (event.x, event.y)
        if self.selector_rect:
            canvas.delete(self.selector_rect)
        self.selector_rect = canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline=COLORS["gold"], width=4
        )

    def _selector_drag(self, canvas: tk.Canvas, event: tk.Event[Any]) -> None:
        if self.selector_start and self.selector_rect:
            canvas.coords(
                self.selector_rect,
                self.selector_start[0],
                self.selector_start[1],
                event.x,
                event.y,
            )

    def _selector_release(
        self, canvas: tk.Canvas, event: tk.Event[Any], resume_after: bool
    ) -> None:
        if not self.selector_start:
            return
        x1, y1 = self.selector_start
        x2, y2 = event.x, event.y
        left, top = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)
        if width < 80 or height < 45:
            if self.selector_rect:
                canvas.itemconfigure(self.selector_rect, outline=COLORS["red"])
            return
        self.region = {"x": left, "y": top, "width": width, "height": height}
        self._save_preferences()
        self._close_selector(False)
        self.status_var.set("框选完成")
        self.main_var.set(f"手牌区域 {width}×{height}，正在重新识别")
        self.start()

    def _close_selector(self, resume_after: bool) -> None:
        if self.selector_window is not None:
            self.selector_window.destroy()
            self.selector_window = None
        self.selector_photo = None
        self.selector_frame = None
        self.selector_start = None
        self.selector_rect = None
        self.root.deiconify()
        self.root.lift()
        if resume_after:
            self.start()

    def _selector_failed(self, exc: Exception, resume_after: bool) -> None:
        self.root.deiconify()
        self._show_error(exc)
        if resume_after:
            self.start()

    def _show_error(self, exc: Exception) -> None:
        self.analysis_inflight = False
        self.status_var.set("操作失败")
        self.main_var.set(friendly_error_message(exc))
        self.meta_var.set("点击“设置”切换捕获方式，或确认游戏与本程序权限一致")

    def _run_worker(
        self,
        function: Callable[[], Any],
        success: Callable[[Any], None],
        failure: Callable[[Exception], None],
    ) -> None:
        def work() -> None:
            try:
                result = function()
            except Exception as exc:
                try:
                    self.root.after(0, lambda error=exc: failure(error))
                except (RuntimeError, tk.TclError):
                    return
            else:
                try:
                    self.root.after(0, lambda value=result: success(value))
                except (RuntimeError, tk.TclError):
                    return

        self.worker.submit(work)

    def close(self) -> None:
        self.running = False
        self._save_preferences()
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
        if self.settings_window is not None:
            self.settings_window.destroy()
        if self.selector_window is not None:
            self.selector_window.destroy()
        self.worker.submit(release_capture_resources)
        self.worker.shutdown(wait=False, cancel_futures=False)
        self.root.destroy()


def run_desktop_app() -> None:
    root = tk.Tk()
    try:
        CompactOverlayApp(root)
        root.mainloop()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


def desktop_ui_smoke_test() -> None:
    """验证顶部条、设置弹层与 Tk 原生 PPM 图像路径。"""
    root = tk.Tk()
    root.withdraw()
    app = CompactOverlayApp(root, auto_start=False)
    preview = tk.PhotoImage(data=b"P6\n1 1\n255\n\x12\x3e\x39", format="PPM")
    if preview.width() != 1 or preview.height() != 1:
        raise RuntimeError("原生预览图像组件初始化失败。")
    root.update_idletasks()
    app.close()
