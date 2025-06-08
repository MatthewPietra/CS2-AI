import win32gui
import win32con
import win32api
import win32ui
import ctypes
import threading
import time
import sys
from ctypes import wintypes
import json
import os
from tkinter import Tk, filedialog, messagebox
 
# --- Configurable Menu State ---
config = {
    'show_menu': False,
    'confidence': 0.5,
    'fov_modifier': 150,
    'fov_color': (0, 255, 0),  # Green
    'show_player_boxes': True,
    'tracking_speed': 0.5,
    'humaniser': 0.2,
    'show_fov_visualiser': True,
}
 
# --- Constants ---
MENU_WIDTH = 320
MENU_HEIGHT = 320
SLIDER_WIDTH = 180
SLIDER_HEIGHT = 12
SLIDER_X = 120
SLIDER_STEP = 0.01
CIRCLE_THICKNESS = 2
 
# --- Win32 Setup ---
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
 
# For hotkey
MOD_NOREPEAT = 0x4000
VK_INSERT = 0x2D
 
# --- Helper Functions ---
def rgb_to_colorref(rgb):
    r, g, b = rgb
    return r | (g << 8) | (b << 16)
 
def point_in_rect(x, y, rect):
    left, top, right, bottom = rect
    return left <= x <= right and top <= y <= bottom
 
# --- Overlay Window Class ---
class OverlayWindow:
    def __init__(self):
        self.hInstance = win32api.GetModuleHandle()
        self.className = 'OverlayWindowMenu'
        self.hwnd = None
        self.running = True
        self.menu_rect = (40, 40, 40 + MENU_WIDTH, 40 + MENU_HEIGHT)
        self.dragging = False
        self.drag_offset = (0, 0)
        self.active_slider = None
        self.mouse_down = False
        self.last_mouse_pos = (0, 0)
        self.register_class()
        self.create_window()
        self.set_click_through(True)
        self.register_hotkey()
        self.menu_items = self.build_menu_items()
        self.status_message = ''
        self.status_message_time = 0
        self.show_config_dropdown = False
        self.config_files = []
        self.dropdown_selected = -1
        self.menu_page = 0
        self.items_per_page = 5
        self.config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
 
    def register_class(self):
        wndClass = win32gui.WNDCLASS()
        wndClass.lpfnWndProc = self.wnd_proc
        wndClass.hInstance = self.hInstance
        wndClass.lpszClassName = self.className
        win32gui.RegisterClass(wndClass)
 
    def create_window(self):
        style = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST
        self.hwnd = win32gui.CreateWindowEx(
            style,
            self.className,
            None,
            win32con.WS_POPUP,
            0, 0, win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1),
            None, None, self.hInstance, None
        )
        win32gui.SetLayeredWindowAttributes(self.hwnd, 0x000000, 0, win32con.LWA_COLORKEY)
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
 
    def set_click_through(self, enable):
        ex_style = win32gui.GetWindowLong(self.hwnd, win32con.GWL_EXSTYLE)
        if enable:
            ex_style |= win32con.WS_EX_TRANSPARENT
        else:
            ex_style &= ~win32con.WS_EX_TRANSPARENT
        win32gui.SetWindowLong(self.hwnd, win32con.GWL_EXSTYLE, ex_style)
 
    def register_hotkey(self):
        if not user32.RegisterHotKey(self.hwnd, 1, MOD_NOREPEAT, VK_INSERT):
            print('Failed to register hotkey')
 
    def build_menu_items(self):
        # Each item: (label, type, config_key, min, max, color)
        items = [
            ('Show FOV Visualiser', 'toggle', 'show_fov_visualiser', None, None, None),
            ('Show Player Boxes', 'toggle', 'show_player_boxes', None, None, None),
            ('Detection Confidence', 'slider', 'confidence', 0.0, 1.0, None),
            ('FOV Modifier', 'slider', 'fov_modifier', 50, 500, None),
            ('FOV Color', 'color', 'fov_color', None, None, None),
            ('Tracking Speed', 'slider', 'tracking_speed', 0.0, 1.0, None),
            ('Humaniser', 'slider', 'humaniser', 0.0, 1.0, None),
        ]
        return items
 
    def wnd_proc(self, hwnd, msg, wparam, lparam):
        if msg == win32con.WM_PAINT:
            hdc, paintStruct = win32gui.BeginPaint(hwnd)
            self.draw_overlay(hdc)
            win32gui.EndPaint(hwnd, paintStruct)
            return 0
        elif msg == win32con.WM_DESTROY:
            win32gui.PostQuitMessage(0)
            return 0
        elif msg == win32con.WM_HOTKEY:
            if wparam == 1:
                config['show_menu'] = not config['show_menu']
                self.set_click_through(not config['show_menu'])
                win32gui.InvalidateRect(self.hwnd, None, True)
            return 0
        elif msg == win32con.WM_LBUTTONDOWN:
            x, y = win32api.LOWORD(lparam), win32api.HIWORD(lparam)
            self.mouse_down = True
            self.last_mouse_pos = (x, y)
            if config['show_menu'] and point_in_rect(x, y, self.menu_rect):
                # Check if dragging menu
                if y < self.menu_rect[1] + 30:
                    self.dragging = True
                    self.drag_offset = (x - self.menu_rect[0], y - self.menu_rect[1])
                # Check sliders/toggles/buttons
                self.handle_menu_mouse(x, y, down=True)
            return 0
        elif msg == win32con.WM_LBUTTONUP:
            self.mouse_down = False
            self.dragging = False
            self.active_slider = None
            return 0
        elif msg == win32con.WM_MOUSEMOVE:
            x, y = win32api.LOWORD(lparam), win32api.HIWORD(lparam)
            if self.dragging:
                dx, dy = self.drag_offset
                win32gui.InvalidateRect(self.hwnd, None, True)
                new_left = max(0, min(x - dx, win32api.GetSystemMetrics(0) - MENU_WIDTH))
                new_top = max(0, min(y - dy, win32api.GetSystemMetrics(1) - MENU_HEIGHT))
                self.menu_rect = (
                    new_left,
                    new_top,
                    new_left + MENU_WIDTH,
                    new_top + MENU_HEIGHT
                )
                win32gui.InvalidateRect(self.hwnd, None, True)
            elif self.mouse_down and self.active_slider is not None:
                self.handle_slider_drag(x, y)
            # List box hover logic
            if config['show_menu'] and self.show_config_dropdown:
                mx, my = x - self.menu_rect[0], y - self.menu_rect[1]
                self.config_files = self.list_valid_configs()
                listbox_x = 20
                listbox_w = MENU_WIDTH - 40
                listbox_h = min(6, len(self.config_files)) * 32 + 8
                listbox_y = MENU_HEIGHT - 50 - listbox_h - 10
                found = False
                for i, fname in enumerate(self.config_files):
                    row_rect = (listbox_x + 4, listbox_y + 4 + i * 32, listbox_x + listbox_w - 4, listbox_y + 4 + (i + 1) * 32)
                    if point_in_rect(mx, my, row_rect):
                        if self.dropdown_selected != i:
                            self.dropdown_selected = i
                            win32gui.InvalidateRect(self.hwnd, None, True)
                        found = True
                        break
                if not found and self.dropdown_selected != -1:
                    self.dropdown_selected = -1
                    win32gui.InvalidateRect(self.hwnd, None, True)
            return 0
        elif msg == win32con.WM_SETCURSOR:
            if config['show_menu']:
                win32gui.SetCursor(win32gui.LoadCursor(0, win32con.IDC_ARROW))
                return True
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)
 
    def handle_menu_mouse(self, x, y, down=False):
        mx, my = x - self.menu_rect[0], y - self.menu_rect[1]
        # Only process visible (paged) items
        item_y = 40
        total_items = len(self.menu_items)
        start_idx = self.menu_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, total_items)
        for idx in range(start_idx, end_idx):
            label, typ, key, minv, maxv, color = self.menu_items[idx]
            if typ == 'toggle':
                rect = (30, item_y, 30 + 20, item_y + 20)
                if point_in_rect(mx, my, rect) and down:
                    config[key] = not config[key]
                    win32gui.InvalidateRect(self.hwnd, None, True)
            elif typ == 'slider':
                rect = (SLIDER_X, item_y, SLIDER_X + SLIDER_WIDTH, item_y + SLIDER_HEIGHT)
                if point_in_rect(mx, my, rect):
                    self.active_slider = (key, minv, maxv, item_y)
                    self.handle_slider_drag(x, y)
            elif typ == 'color':
                rect = (SLIDER_X, item_y, SLIDER_X + 40, item_y + 20)
                if point_in_rect(mx, my, rect) and down:
                    # Cycle color for demo
                    r, g, b = config[key]
                    config[key] = (g, b, r)
                    win32gui.InvalidateRect(self.hwnd, None, True)
            item_y += 40
        # --- Paging Buttons ---
        nav_btn_w, nav_btn_h = 80, 28
        nav_btn_y = MENU_HEIGHT - 90
        prev_btn_rect = (20, nav_btn_y, 20 + nav_btn_w, nav_btn_y + nav_btn_h)
        next_btn_rect = (120, nav_btn_y, 120 + nav_btn_w, nav_btn_y + nav_btn_h)
        if point_in_rect(mx, my, prev_btn_rect) and down and self.menu_page > 0:
            self.menu_page -= 1
            win32gui.InvalidateRect(self.hwnd, None, True)
            return
        if point_in_rect(mx, my, next_btn_rect) and down and end_idx < total_items:
            self.menu_page += 1
            win32gui.InvalidateRect(self.hwnd, None, True)
            return
        # --- Save/Load Config Buttons ---
        btn_w, btn_h = 120, 28
        btn_y = MENU_HEIGHT - 50
        save_btn_rect = (20, btn_y, 20 + btn_w, btn_y + btn_h)
        load_btn_rect = (160, btn_y, 160 + btn_w, btn_y + btn_h)
        if point_in_rect(mx, my, save_btn_rect) and down:
            self.save_config()
            self.show_config_dropdown = False
        elif point_in_rect(mx, my, load_btn_rect) and down:
            self.show_config_dropdown = not self.show_config_dropdown
            self.dropdown_selected = -1
        elif self.show_config_dropdown:
            # List box logic
            self.config_files = self.list_valid_configs()
            listbox_x = 20
            listbox_w = MENU_WIDTH - 40
            listbox_h = min(6, len(self.config_files)) * 32 + 8
            listbox_y = MENU_HEIGHT - 50 - listbox_h - 10
            for i, fname in enumerate(self.config_files):
                row_rect = (listbox_x + 4, listbox_y + 4 + i * 32, listbox_x + listbox_w - 4, listbox_y + 4 + (i + 1) * 32)
                if point_in_rect(mx, my, row_rect) and down:
                    self.dropdown_selected = i
                    self.load_config_by_name(fname)
                    self.show_config_dropdown = False
                    return
            # Click outside listbox closes it
            self.show_config_dropdown = False
 
    def handle_slider_drag(self, x, y):
        key, minv, maxv, item_y = self.active_slider
        mx = x - self.menu_rect[0]
        rel = (mx - SLIDER_X) / SLIDER_WIDTH
        rel = max(0.0, min(1.0, rel))
        value = minv + (maxv - minv) * rel
        if isinstance(minv, int):
            value = int(value)
        else:
            value = round(value, 2)
        config[key] = value
        win32gui.InvalidateRect(self.hwnd, None, True)
 
    def draw_overlay(self, hdc):
        width = win32api.GetSystemMetrics(0)
        height = win32api.GetSystemMetrics(1)
        # --- Double buffering setup ---
        mem_dc = win32gui.CreateCompatibleDC(hdc)
        bmp = win32gui.CreateCompatibleBitmap(hdc, width, height)
        old_bmp = win32gui.SelectObject(mem_dc, bmp)
        # Clear the entire buffer with the transparent color key
        transparent_brush = win32gui.CreateSolidBrush(0x000000)
        win32gui.FillRect(mem_dc, (0, 0, width, height), transparent_brush)
        win32gui.DeleteObject(transparent_brush)
        # Draw FOV circle (only if enabled)
        if config.get('show_fov_visualiser', True):
            x, y = width // 2, height // 2
            color = rgb_to_colorref(config['fov_color'])
            pen = win32gui.CreatePen(win32con.PS_SOLID, CIRCLE_THICKNESS, color)
            old_pen = win32gui.SelectObject(mem_dc, pen)
            win32gui.SelectObject(mem_dc, gdi32.GetStockObject(win32con.NULL_BRUSH))
            win32gui.Ellipse(mem_dc, x - config['fov_modifier'], y - config['fov_modifier'], x + config['fov_modifier'], y + config['fov_modifier'])
            win32gui.SelectObject(mem_dc, old_pen)
            win32gui.DeleteObject(pen)
        # Draw menu if open
        if config['show_menu']:
            self.draw_menu(mem_dc)
        # --- BitBlt the buffer to the window ---
        win32gui.BitBlt(hdc, 0, 0, width, height, mem_dc, 0, 0, win32con.SRCCOPY)
        # --- Cleanup ---
        win32gui.SelectObject(mem_dc, old_bmp)
        win32gui.DeleteObject(bmp)
        win32gui.DeleteDC(mem_dc)
 
    def draw_menu(self, hdc):
        left, top, right, bottom = self.menu_rect
        # Menu background
        brush = win32gui.CreateSolidBrush(rgb_to_colorref((30, 30, 30)))
        win32gui.FillRect(hdc, (left, top, right, bottom), brush)
        win32gui.DeleteObject(brush)
        # Menu border
        pen = win32gui.CreatePen(win32con.PS_SOLID, 2, rgb_to_colorref((80, 255, 80)))
        old_pen = win32gui.SelectObject(hdc, pen)
        win32gui.SelectObject(hdc, gdi32.GetStockObject(win32con.NULL_BRUSH))
        win32gui.Rectangle(hdc, left, top, right, bottom)
        win32gui.SelectObject(hdc, old_pen)
        win32gui.DeleteObject(pen)
        # Title
        self.draw_text(hdc, left + 10, top + 8, 'Overlay Menu', 0x00FFFFFF, 18, bold=True)
        # Items
        item_y = top + 40
        # Paging logic
        total_items = len(self.menu_items)
        start_idx = self.menu_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, total_items)
        for idx in range(start_idx, end_idx):
            label, typ, key, minv, maxv, color = self.menu_items[idx]
            if typ == 'toggle':
                # Checkbox
                box_rect = (left + 30, item_y, left + 50, item_y + 20)
                if config[key]:
                    # Fill the box with green
                    green_brush = win32gui.CreateSolidBrush(rgb_to_colorref((80, 255, 80)))
                    win32gui.FillRect(hdc, box_rect, green_brush)
                    win32gui.DeleteObject(green_brush)
                    # Draw green checkmark
                    green_pen = win32gui.CreatePen(win32con.PS_SOLID, 2, rgb_to_colorref((30, 30, 30)))
                    old_pen = win32gui.SelectObject(hdc, green_pen)
                    win32gui.MoveToEx(hdc, box_rect[0], box_rect[1])
                    win32gui.LineTo(hdc, box_rect[2], box_rect[3])
                    win32gui.MoveToEx(hdc, box_rect[0], box_rect[3])
                    win32gui.LineTo(hdc, box_rect[2], box_rect[1])
                    win32gui.SelectObject(hdc, old_pen)
                    win32gui.DeleteObject(green_pen)
                    # Draw green outline
                    green_pen_outline = win32gui.CreatePen(win32con.PS_SOLID, 2, rgb_to_colorref((80, 255, 80)))
                    old_pen_outline = win32gui.SelectObject(hdc, green_pen_outline)
                    win32gui.Rectangle(hdc, *box_rect)
                    win32gui.SelectObject(hdc, old_pen_outline)
                    win32gui.DeleteObject(green_pen_outline)
                self.draw_text(hdc, left + 60, item_y, label, 0x00FFFFFF, 14)
            elif typ == 'slider':
                # Label
                self.draw_text(hdc, left + 10, item_y, label, 0x00FFFFFF, 14)
                # Slider bar
                sx, sy = left + SLIDER_X, item_y + 6
                # Draw filled green slider bar
                green_brush = win32gui.CreateSolidBrush(rgb_to_colorref((80, 255, 80)))
                win32gui.FillRect(hdc, (sx, sy, sx + SLIDER_WIDTH, sy + SLIDER_HEIGHT), green_brush)
                win32gui.DeleteObject(green_brush)
                # Draw slider outline (optional, for contrast)
                win32gui.Rectangle(hdc, sx, sy, sx + SLIDER_WIDTH, sy + SLIDER_HEIGHT)
                # Slider /
                rel = (config[key] - minv) / (maxv - minv)
                handle_x = int(sx + rel * SLIDER_WIDTH)
                # Draw filled white slider handle
                white_brush = win32gui.CreateSolidBrush(rgb_to_colorref((255, 255, 255)))
                win32gui.FillRect(hdc, (handle_x - 4, sy - 4, handle_x + 4, sy + SLIDER_HEIGHT + 4), white_brush)
                win32gui.DeleteObject(white_brush)
                win32gui.Rectangle(hdc, handle_x - 4, sy - 4, handle_x + 4, sy + SLIDER_HEIGHT + 4)
                # Value
                val_str = f'{config[key]:.2f}' if isinstance(config[key], float) else str(config[key])
                # Place value just to the right of the slider, but inside the menu
                value_bg_left = min(sx + SLIDER_WIDTH + 10, right - 40)
                value_bg_top = item_y + 2
                value_bg_right = value_bg_left + 32
                value_bg_bottom = value_bg_top + 16
                # Draw solid background for value (menu background color)
                value_bg_brush = win32gui.CreateSolidBrush(rgb_to_colorref((30, 30, 30)))
                win32gui.FillRect(hdc, (value_bg_left, value_bg_top, value_bg_right, value_bg_bottom), value_bg_brush)
                win32gui.DeleteObject(value_bg_brush)
                self.draw_text(hdc, value_bg_left + 2, item_y + 2, val_str, 0x00FFFFFF, 14)
            elif typ == 'color':
                self.draw_text(hdc, left + 10, item_y, label, 0x00FFFFFF, 14)
                cx, cy = left + SLIDER_X, item_y
                color_brush = win32gui.CreateSolidBrush(rgb_to_colorref(config[key]))
                win32gui.FillRect(hdc, (cx, cy, cx + 40, cy + 20), color_brush)
                win32gui.DeleteObject(color_brush)
            item_y += 40
        # --- Paging Buttons ---
        nav_btn_w, nav_btn_h = 80, 28
        nav_btn_y = bottom - 90
        prev_btn_rect = (left + 20, nav_btn_y, left + 20 + nav_btn_w, nav_btn_y + nav_btn_h)
        next_btn_rect = (left + 120, nav_btn_y, left + 120 + nav_btn_w, nav_btn_y + nav_btn_h)
        # Previous button
        prev_brush = win32gui.CreateSolidBrush(rgb_to_colorref((60, 60, 60)) if self.menu_page > 0 else rgb_to_colorref((30, 30, 30)))
        win32gui.FillRect(hdc, prev_btn_rect, prev_brush)
        win32gui.DeleteObject(prev_brush)
        black_brush = win32gui.CreateSolidBrush(rgb_to_colorref((0, 0, 0)))
        win32gui.FrameRect(hdc, prev_btn_rect, black_brush)
        win32gui.DeleteObject(black_brush)
        self.draw_text(hdc, prev_btn_rect[0] + 12, prev_btn_rect[1] + 5, 'Previous', 0x00FFFFFF, 14, bold=True)
        # Next button
        next_brush = win32gui.CreateSolidBrush(rgb_to_colorref((60, 60, 60)) if end_idx < total_items else rgb_to_colorref((30, 30, 30)))
        win32gui.FillRect(hdc, next_btn_rect, next_brush)
        win32gui.DeleteObject(next_brush)
        black_brush = win32gui.CreateSolidBrush(rgb_to_colorref((0, 0, 0)))
        win32gui.FrameRect(hdc, next_btn_rect, black_brush)
        win32gui.DeleteObject(black_brush)
        self.draw_text(hdc, next_btn_rect[0] + 22, next_btn_rect[1] + 5, 'Next', 0x00FFFFFF, 14, bold=True)
        # --- Save/Load Config Buttons ---
        btn_w, btn_h = 120, 28
        btn_y = bottom - 50
        save_btn_rect = (left + 20, btn_y, left + 20 + btn_w, btn_y + btn_h)
        load_btn_rect = (left + 160, btn_y, left + 160 + btn_w, btn_y + btn_h)
        # Save button
        save_brush = win32gui.CreateSolidBrush(rgb_to_colorref((60, 120, 60)))
        win32gui.FillRect(hdc, save_btn_rect, save_brush)
        win32gui.DeleteObject(save_brush)
        black_brush = win32gui.CreateSolidBrush(rgb_to_colorref((0, 0, 0)))
        win32gui.FrameRect(hdc, save_btn_rect, black_brush)
        win32gui.DeleteObject(black_brush)
        self.draw_text(hdc, save_btn_rect[0] + 18, save_btn_rect[1] + 5, 'Save Config', 0x00FFFFFF, 14, bold=True)
        # Load button
        load_brush = win32gui.CreateSolidBrush(rgb_to_colorref((60, 60, 120)))
        win32gui.FillRect(hdc, load_btn_rect, load_brush)
        win32gui.DeleteObject(load_brush)
        black_brush = win32gui.CreateSolidBrush(rgb_to_colorref((0, 0, 0)))
        win32gui.FrameRect(hdc, load_btn_rect, black_brush)
        win32gui.DeleteObject(black_brush)
        self.draw_text(hdc, load_btn_rect[0] + 18, load_btn_rect[1] + 5, 'Load Config', 0x00FFFFFF, 14, bold=True)
        # List box for configs
        if self.show_config_dropdown:
            self.config_files = self.list_valid_configs()
            listbox_x = left + 20
            listbox_w = MENU_WIDTH - 40
            listbox_h = min(6, len(self.config_files)) * 32 + 8
            listbox_y = bottom - 50 - listbox_h - 10
            listbox_rect = (listbox_x, listbox_y, listbox_x + listbox_w, listbox_y + listbox_h)
            # Draw listbox background
            listbox_brush = win32gui.CreateSolidBrush(rgb_to_colorref((40, 40, 60)))
            win32gui.FillRect(hdc, listbox_rect, listbox_brush)
            win32gui.DeleteObject(listbox_brush)
            win32gui.Rectangle(hdc, *listbox_rect)
            # Draw each config name as a row
            for i, fname in enumerate(self.config_files):
                row_rect = (listbox_x + 4, listbox_y + 4 + i * 32, listbox_x + listbox_w - 4, listbox_y + 4 + (i + 1) * 32)
                if i == self.dropdown_selected:
                    sel_brush = win32gui.CreateSolidBrush(rgb_to_colorref((80, 255, 80)))
                    win32gui.FillRect(hdc, row_rect, sel_brush)
                    win32gui.DeleteObject(sel_brush)
                win32gui.Rectangle(hdc, *row_rect)
                self.draw_text(hdc, row_rect[0] + 8, row_rect[1] + 6, fname, 0x00FFFFFF, 14)
        # Status message
        if self.status_message and time.time() < self.status_message_time:
            self.draw_text(hdc, left + 20, bottom - 20, self.status_message, 0x00FFAAAA, 14)
 
    def draw_text(self, hdc, x, y, text, color, size, bold=False):
        lf = win32gui.LOGFONT()
        lf.lfHeight = -size
        lf.lfWeight = 700 if bold else 400
        lf.lfFaceName = 'Segoe UI'
        font = win32gui.CreateFontIndirect(lf)
        old_font = win32gui.SelectObject(hdc, font)
        win32gui.SetTextColor(hdc, color)
        win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
        win32gui.ExtTextOut(hdc, x, y, 0, None, text, None)
        win32gui.SelectObject(hdc, old_font)
        win32gui.DeleteObject(font)
 
    def run(self):
        threading.Thread(target=self.redraw_loop, daemon=True).start()
        win32gui.PumpMessages()
        self.running = False
 
    def redraw_loop(self):
        while self.running:
            win32gui.InvalidateRect(self.hwnd, None, True)
            time.sleep(1/144)  # 144 FPS
 
    def save_config(self):
        config_to_save = dict(config)
        config_to_save['overlay_menu_config'] = True
        try:
            Tk().withdraw()
            file_path = filedialog.asksaveasfilename(
                defaultextension='.json',
                filetypes=[('JSON Files', '*.json')],
                title='Save Config',
                initialdir=self.config_dir
            )
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(config_to_save, f, indent=2)
                self.set_status_message('Config saved!')
        except Exception as e:
            self.set_status_message(f'Error saving: {e}')
 
    def list_valid_configs(self):
        files = []
        for fname in os.listdir(self.config_dir):
            if fname.endswith('.json'):
                try:
                    with open(os.path.join(self.config_dir, fname), 'r') as f:
                        loaded = json.load(f)
                    if loaded.get('overlay_menu_config'):
                        files.append(fname)
                except Exception:
                    continue
        return files
 
    def load_config_by_name(self, fname):
        try:
            file_path = os.path.join(self.config_dir, fname)
            with open(file_path, 'r') as f:
                loaded = json.load(f)
            if loaded.get('overlay_menu_config'):
                for k in config:
                    if k in loaded:
                        config[k] = loaded[k]
                self.set_status_message('Config loaded!')
                win32gui.InvalidateRect(self.hwnd, None, True)
            else:
                self.set_status_message('Invalid config file!')
        except Exception as e:
            self.set_status_message(f'Error loading: {e}')
 
    def set_status_message(self, msg, duration=2):
        self.status_message = msg
        self.status_message_time = time.time() + duration
 
if __name__ == "__main__":
    overlay = OverlayWindow()
    overlay.run()
