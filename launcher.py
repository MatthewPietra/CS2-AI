import win32gui
import win32con
import win32api
import win32ui
import ctypes
import threading
import time
import sys
import traceback
from ctypes import wintypes
import json
import os
from tkinter import Tk, filedialog, messagebox, colorchooser
import torch
import numpy as np
import cv2
from PIL import ImageGrab, Image
from ultralytics import YOLO
from ultralytics.trackers import BYTETracker
import win32process
import psutil
import queue
import collections
import torch_directml
from argparse import Namespace
import GPUtil
import subprocess
import platform
import wmi
import logging
from datetime import datetime
import math
import pyautogui
import mss

# --- Configurable Menu State ---
config = {
    'show_menu': False,
    'confidence': 0.5,
    'fov_modifier': 150,
    'fov_size': 150,  # Add fov_size to match fov_modifier
    'fov_color': (0, 255, 0),  # Green
    'show_player_boxes': True,
    'tracking_speed': 0.5,
    'humaniser': 0.2,
    'show_fov_visualiser': True,
    'tracking_key': 'F2',  # Default tracking key
    'gpu_info': None,  # Will store GPU info
    'show_fps_counter': False,  # FPS counter toggle
    'smoothing_xy': 0.5,
    'optimizations_enabled': False,
    'gpu_utilization_threshold': 80,
    'optimized_fps_target': 60,
    'default_target_fps': 144,
    'aim_assist_enabled': False,  # Add aim assist toggle
    'trigger_bot_enabled': False,  # Add trigger bot toggle
    'box_color': (255, 0, 0),  # Red
    'text_color': (255, 255, 255),  # White
    'menu_background_color': (15, 15, 15),
    'menu_border_color': (0, 255, 0),
    'menu_tab_color': (20, 20, 20),
    'menu_active_tab_color': (30, 30, 30),
    'menu_text_color': (255, 255, 255),
    'menu_highlight_color': (0, 255, 0),
    'slider_color': (40, 40, 40),
    'slider_active_color': (0, 255, 0),
    'button_color': (40, 40, 40),
    'button_hover_color': (60, 60, 60),
    'button_active_color': (0, 255, 0),
    'status_font_size': 14,
    'fps_display_style': 0,  # 0 = show labels, 1 = just numbers
    'menu_key': 'INSERT',
    'trigger_key': 'F3',
    'fov_toggle_key': 'F4',
    'fps_toggle_key': 'F5',
    'use_gpu_capture': True,  # Added option to use GPU for capture processing
    'capture_size': 480,      # Configurable capture size
    'target_size': 160,       # Configurable target size for model input
    'max_boxes': 3,           # Maximum number of boxes to process
}

# Add this after INFO_DESCRIPTIONS
INFO_ICON_EXCLUDE = set([
    'gpu_info', 'save_config', 'load_config', 'reset_settings',
    'tracking_key', 'menu_key', 'trigger_key', 'fov_toggle_key', 'fps_toggle_key',
])

INFO_DESCRIPTIONS = {
    'optimizations_enabled': 'Enable performance optimizations based on GPU usage.',
    'use_directx_capture': 'Use DirectX for faster GPU-based screen capture.',
    'gpu_utilization_threshold': 'Max GPU usage (%) before optimizations activate.',
    'optimized_fps_target': 'FPS to target when optimizations are active.',
    'default_target_fps': 'Default FPS target for detection/processing.',
    'show_fov_visualiser': 'Show the field of view (FOV) circle.',
    'show_player_boxes': 'Draw boxes around detected players.',
    'confidence': 'Detection confidence threshold for object detection.',
    'tracking_speed': 'Speed at which the aim assist tracks targets.',
    'humaniser': 'Adds randomness to aim movement for realism.',
    'fov_modifier': 'Adjusts the size of the field of view (FOV) circle.',
    'anti_lag_value': 'Compensates for system lag in aim assist.',
    'custom_bone_position': 'Custom offset for aim target (advanced).',
    'smoothing_xy': 'Smoothness of aim movement in X/Y directions.',
    'trigger_bot_enabled': 'Automatically fires when a target is detected.',
    'trigger_delay': 'Delay (seconds) before trigger fires.',
    'trigger_random': 'Randomness added to trigger timing.',
    'trigger_hold': 'How long to hold the trigger (seconds).',
    'trigger_release': 'Delay (seconds) after trigger fires.',
    'show_fps_counter': 'Display the current frames per second (FPS).',
    'fps_display_style': 'Choose between detailed or simple FPS display.',
    'fov_color': 'Color of the field of view (FOV) circle.',
    'box_color': 'Color of the player detection boxes.',
    'menu_background_color': 'Background color of the menu.',
    'optimizations_enabled': 'Enable performance optimizations based on GPU usage.',
    'use_directx_capture': 'Use DirectX for faster GPU-based screen capture.',
    # Add more as needed for all other options...
}
 
# --- Constants ---
SCREEN_WIDTH = win32api.GetSystemMetrics(0)
SCREEN_HEIGHT = win32api.GetSystemMetrics(1)

# Define fixed menu dimensions for better control
FIXED_MENU_WIDTH = 500
FIXED_MENU_HEIGHT = 320

# Calculate menu position to be in top-left corner
MENU_START_X = 20
MENU_START_Y = 20

FULL_MENU_RECT = (
    MENU_START_X,
    MENU_START_Y,
    MENU_START_X + FIXED_MENU_WIDTH,
    MENU_START_Y + FIXED_MENU_HEIGHT
)

TOP_BAR_HEIGHT = 30
SIDEBAR_WIDTH = 100
BOTTOM_BAR_HEIGHT = 20

SLIDER_WIDTH = 140
SLIDER_HEIGHT = 8
SLIDER_X = SIDEBAR_WIDTH + 15
SLIDER_STEP = 0.01
CIRCLE_THICKNESS = 2

# Add new constants for scaled elements
TAB_HEIGHT = 28
TEXT_SIZE = 14
BUTTON_HEIGHT = 24
BUTTON_PADDING = 8
TOGGLE_SIZE = 20
LABEL_WIDTH = 170
INFO_ICON_SIZE = 18
INFO_ICON_OFFSET = 6
 
# --- Win32 Setup ---
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
 
# For hotkey
MOD_NOREPEAT = 0x4000
VK_INSERT = 0x2D
VK_F2 = 0x71  # F2 key for toggling aim
VK_F4 = 0x73  # F4 key for toggling FOV
VK_F5 = 0x74  # F5 key for toggling FPS

# Mouse movement constants
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000

# Add key mapping constants
VK_F1 = 0x70
VK_F3 = 0x72
VK_F6 = 0x75
VK_F7 = 0x76
VK_F8 = 0x77
VK_F9 = 0x78
VK_F10 = 0x79
VK_F11 = 0x7A
VK_F12 = 0x7B

# Key mapping dictionary
KEY_MAPPING = {
    'F1': VK_F1,
    'F2': VK_F2,
    'F3': VK_F3,
    'F4': VK_F4,
    'F5': VK_F5,
    'F6': VK_F6,
    'F7': VK_F7,
    'F8': VK_F8,
    'F9': VK_F9,
    'F10': VK_F10,
    'F11': VK_F11,
    'F12': VK_F12,
    'INSERT': VK_INSERT,
}

# Reverse key mapping for getting key names
REVERSE_KEY_MAPPING = {v: k for k, v in KEY_MAPPING.items()}

def get_key_name(vk_code):
    """Convert virtual key code to key name."""
    if vk_code in REVERSE_KEY_MAPPING:
        return REVERSE_KEY_MAPPING[vk_code]
    return f"Key {vk_code}"

def move_mouse(dx, dy):
    """Move the mouse cursor by the specified delta using SendInput (safer than mouse_event)."""
    import ctypes
    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]
    class INPUT(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_ulong),
            ("mi", MOUSEINPUT),
        ]
    extra = ctypes.c_ulong(0)
    ii_ = INPUT(type=0, mi=MOUSEINPUT(int(dx), int(dy), 0, 0x0001, 0, ctypes.pointer(extra)))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(ii_), ctypes.sizeof(ii_))

def simulate_click():
    """Simulate a left mouse click using SendInput (safer than mouse_event)."""
    import ctypes
    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]
    class INPUT(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_ulong),
            ("mi", MOUSEINPUT),
        ]
    extra = ctypes.c_ulong(0)
    # Mouse left down
    ii_down = INPUT(type=0, mi=MOUSEINPUT(0, 0, 0, 0x0002, 0, ctypes.pointer(extra)))
    # Mouse left up
    ii_up = INPUT(type=0, mi=MOUSEINPUT(0, 0, 0, 0x0004, 0, ctypes.pointer(extra)))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(ii_down), ctypes.sizeof(ii_down))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(ii_up), ctypes.sizeof(ii_up))

# --- Helper Functions ---
def rgb_to_colorref(rgb):
    r, g, b = rgb
    return r | (g << 8) | (b << 16)
 
def point_in_rect(x, y, rect):
    left, top, right, bottom = rect
    return left <= x <= right and top <= y <= bottom
 
# --- Overlay Window Class ---
class ConfigManager:
    """Manages configuration with default values and type checking."""
    
    @staticmethod
    def get_default_config():
        """Returns a dictionary of all possible configuration options with their default values."""
        return {
            # --- Display Settings ---
            'show_menu': False,
            'show_fps_counter': True,
            'show_player_boxes': True,
            'show_fov_visualiser': True,
            'fov_enabled': True,
            'box_enabled': True,
            'text_enabled': True,
            'menu_enabled': True,
            'fps_display_style': 'simple',  # 'simple' or 'detailed'
            'show_detection_confidence': True,
            'show_track_ids': True,
            'show_target_info': True,
            
            # --- Colors ---
            'box_color': (255, 0, 0),  # Red
            'text_color': (255, 255, 255),  # White
            'fov_color': (0, 255, 0),  # Green
            'menu_color': (0, 0, 0, 128),
            'menu_text_color': (255, 255, 255),
            'menu_highlight_color': (0, 255, 0),
            'menu_background_color': (15, 15, 15),
            'menu_border_color': (0, 255, 0),
            'menu_tab_color': (20, 20, 20),
            'menu_active_tab_color': (30, 30, 30),
            'slider_color': (40, 40, 40),
            'slider_active_color': (0, 255, 0),
            'button_color': (40, 40, 40),
            'button_hover_color': (60, 60, 60),
            'button_active_color': (0, 255, 0),
            
            # --- Sizes and Dimensions ---
            'fov_size': 200,
            'aim_fov': 200,
            'menu_width': 460,
            'menu_height': 290,
            'menu_x': 10,
            'menu_y': 10,
            'menu_padding': 10,
            'menu_spacing': 5,
            'menu_max_items': 20,
            'menu_item_height': 25,
            'menu_font_size': 12,
            'top_bar_height': 23,
            'sidebar_width': 92,
            'bottom_bar_height': 17,
            'slider_width': 115,
            'slider_height': 7,
            'button_height': 21,
            'button_padding': 6,
            'tab_height': 23,
            'circle_thickness': 2,
            
            # --- Performance Settings ---
            'confidence': 0.5,
            'aim_smoothness': 0.5,
            'aim_speed': 1.0,
            'tracking_speed': 0.5,
            'humaniser': 0.2,
            'anti_lag_value': 5.0,
            'custom_bone_position': 0.0,
            'smoothing_xy': 0.5,
            'optimizations_enabled': False,
            'gpu_utilization_threshold': 80,
            'optimized_fps_target': 144,
            'default_target_fps': 230,
            'max_detection_fps': 144,
            'min_detection_fps': 30,
            'frame_queue_size': 2,
            'result_queue_size': 2,
            'fps_history_size': 30,
            
            # --- Timing Settings ---
            'trigger_delay': 0.1,
            'trigger_random': 0.05,
            'trigger_hold': 0.05,
            'trigger_release': 0.1,
            'aim_delay': 0.0,
            'aim_random': 0.0,
            'menu_update_interval': 0.016,  # ~60 FPS
            'status_message_duration': 3.0,
            
            # --- Feature Toggles ---
            'trigger_enabled': True,
            'trigger_bot_enabled': False,
            'aim_enabled': True,
            'tracking_enabled': True,
            'use_directx_capture': True,
            'use_gpu_capture': True,
            'use_optimized_capture': False,
            'use_anti_aliasing': True,
            'use_vsync': False,
            'use_fullscreen': True,
            'use_borderless': False,
            'use_click_through': True,
            'use_topmost': True,
            
            # --- Key Bindings ---
            'menu_key': 'insert',
            'exit_key': 'end',
            'reset_key': 'home',
            'save_key': 'f5',
            'load_key': 'f6',
            'tracking_key': 'f2',
            'fov_toggle_key': 'f4',
            'fps_toggle_key': 'f5',
            'aim_key': 'shift',
            'trigger_key': 'f3',
            'is_mouse_key': False,  # Track if aim key is a mouse button
            
            # --- Menu State ---
            'menu_visible': False,
            'menu_active': False,
            'menu_selected': 0,
            'menu_scroll': 0,
            'active_tab': 'Home',
            'mouse_down': False,
            'last_mouse_pos': (0, 0),
            'active_slider': None,
            'waiting_for_key': None,
            'active_input_field': None,
            'current_input_text': "",
            
            # --- Font Settings ---
            'menu_font': 'Arial',
            'menu_bold': True,
            'menu_italic': False,
            'menu_underline': False,
            'menu_strikeout': False,
            'menu_charset': 0,
            'menu_quality': 0,
            'menu_pitch': 0,
            'menu_family': 0,
            
            # --- Detection Settings ---
            'detection_confidence': 0.5,
            'detection_iou_threshold': 0.45,
            'detection_max_detections': 100,
            'detection_min_size': 20,
            'detection_max_size': 1000,
            'detection_aspect_ratio': 1.0,
            'detection_scale_factor': 1.0,
            
            # --- Tracking Settings ---
            'track_thresh': 0.5,
            'track_buffer': 30,
            'match_thresh': 0.8,
            'frame_rate': 144,
            'track_min_hits': 3,
            'track_max_age': 30,
            'track_min_steps': 3,
            'track_max_steps': 30,
            
            # --- Aim Settings ---
            'aim_fov_radius': 200,
            'aim_smooth_factor': 0.5,
            'aim_prediction_factor': 0.5,
            'aim_reaction_time': 0.1,
            'aim_max_angle': 180,
            'aim_min_angle': 0,
            'aim_max_distance': 1000,
            'aim_min_distance': 50,
            'aim_target_priority': 'distance',  # 'distance', 'confidence', 'size'
            'aim_target_bone': 'head',  # 'head', 'chest', 'pelvis'
            'aim_target_offset': 0.0,
            
            # --- Trigger Settings ---
            'trigger_delay_min': 0.05,
            'trigger_delay_max': 0.15,
            'trigger_hold_min': 0.05,
            'trigger_hold_max': 0.15,
            'trigger_release_min': 0.05,
            'trigger_release_max': 0.15,
            'trigger_random_factor': 0.1,
            'trigger_confidence_threshold': 0.5,
            'trigger_max_distance': 1000,
            'trigger_min_distance': 50,
            'trigger_target_bone': 'head',  # 'head', 'chest', 'pelvis'
            'trigger_target_offset': 0.0,
            
            # --- System Settings ---
            'config_dir': 'config',
            'model_path': 'best.pt',
            'log_level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
            'save_screenshots': False,
            'screenshot_dir': 'screenshots',
            'auto_update': False,
            'check_updates': True,
            'update_channel': 'stable',  # 'stable', 'beta', 'alpha'
            
            # --- Debug Settings ---
            'debug_mode': False,
            'debug_draw_fps': True,
            'debug_draw_boxes': True,
            'debug_draw_tracks': True,
            'debug_draw_targets': True,
            'debug_draw_angles': True,
            'debug_draw_distances': True,
            'debug_log_detections': False,
            'debug_log_tracking': False,
            'debug_log_aiming': False,
            'debug_log_trigger': False,
        }
    
    @staticmethod
    def validate_config(config):
        """Validates and ensures all required configuration options exist."""
        default_config = ConfigManager.get_default_config()
        validated_config = default_config.copy()
        
        # Update with any existing values
        for key, value in config.items():
            if key in default_config:
                validated_config[key] = value
        
        return validated_config
    
    @staticmethod
    def get_category_config(category):
        """Returns configuration options for a specific category."""
        all_config = ConfigManager.get_default_config()
        categories = {
            'display': [k for k in all_config.keys() if k.startswith('show_') or k.endswith('_enabled')],
            'colors': [k for k in all_config.keys() if k.endswith('_color')],
            'sizes': [k for k in all_config.keys() if k.endswith('_size') or k.endswith('_width') or k.endswith('_height')],
            'performance': [k for k in all_config.keys() if 'fps' in k or 'speed' in k or 'threshold' in k],
            'timing': [k for k in all_config.keys() if 'delay' in k or 'time' in k or 'interval' in k],
            'features': [k for k in all_config.keys() if k.startswith('use_') or k.endswith('_enabled')],
            'keys': [k for k in all_config.keys() if k.endswith('_key')],
            'menu': [k for k in all_config.keys() if k.startswith('menu_')],
            'fonts': [k for k in all_config.keys() if k.startswith('menu_font')],
            'detection': [k for k in all_config.keys() if k.startswith('detection_')],
            'tracking': [k for k in all_config.keys() if k.startswith('track_')],
            'aim': [k for k in all_config.keys() if k.startswith('aim_')],
            'trigger': [k for k in all_config.keys() if k.startswith('trigger_')],
            'system': [k for k in all_config.keys() if k in ['config_dir', 'model_path', 'log_level', 'save_screenshots', 'screenshot_dir', 'auto_update', 'check_updates', 'update_channel']],
            'debug': [k for k in all_config.keys() if k.startswith('debug_')],
        }
        return {k: all_config[k] for k in categories.get(category, [])}

def detect_gpu():
    """Detect available GPU and return device type and name."""
    try:
        import torch
        import wmi
        
        # Try to get GPU name using WMI
        try:
            w = wmi.WMI()
            gpu_info = w.Win32_VideoController()
            if gpu_info:
                for gpu in gpu_info:
                    if gpu.Name:
                        if "NVIDIA" in gpu.Name:
                            if torch.cuda.is_available():
                                return 'NVIDIA', gpu.Name
                        elif "AMD" in gpu.Name or "Radeon" in gpu.Name:
                            try:
                                import torch_directml
                                if torch_directml.is_available():
                                    return 'AMD', gpu.Name
                            except ImportError:
                                print("torch-directml not installed. To use AMD GPU, install it with: pip install torch-directml")
        except:
            pass
            
        # Check for NVIDIA GPU using CUDA
        if torch.cuda.is_available():
            return 'NVIDIA', torch.cuda.get_device_name(0)
            
        # Check for AMD GPU using DirectML
        try:
            import torch_directml
            if torch_directml.is_available():
                return 'AMD', 'AMD GPU (DirectML)'
        except ImportError:
            print("torch-directml not installed. To use AMD GPU, install it with: pip install torch-directml")
            
        # If no GPU found, return CPU info
        import platform
        cpu_info = platform.processor()
        return 'CPU', cpu_info

    except Exception as e:
        print(f"Error detecting GPU: {e}")
        return 'CPU', 'Unknown CPU'

class YOLODetector:
    def __init__(self, config):
        """Initialize the YOLODetector with configuration."""
        self.config = config
        self.model_path = config.get('model_path', 'best.pt')
        self.gpu_type = None
        self.gpu_name = None
        self.device = None
        self.dml_device = None
        self.model = None
        self.aiming_enabled = False
        self.last_fps = 0
        self.current_fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # Add back necessary attributes for detection thread
        self.running = False
        self.detection_thread = None
        self.fps_history = []
        self.processing_fps_history = []
        self.current_boxes = None
        self.current_confidences = None
        self.current_class_ids = None
        self.class_names = ['item']  # Define class names
        self.last_box_state = False  # Track if boxes were present last frame
        
        # Frame skipping parameters
        self.process_every_n = 1  # Start by processing every frame
        self.skip_counter = 0
        self.frame_counter = 0  # Counter for frame skipping
        self.last_detections = []  # Cache for frame skipping
        
        # Initialize tracker
        from argparse import Namespace
        tracker_args = Namespace(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30
        )
        self.tracker = BYTETracker(tracker_args)
        
        # Initialize GPU detection
        self._detect_gpu()
        
        # Load the model
        if not self.load_model():
            raise Exception("Failed to load YOLOv8 model")
            
        # Check for MSS library
        try:
            import mss
            print("MSS screen capture library found")
        except ImportError:
            print("MSS library not found. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "mss"])
            print("MSS installed successfully")

    def _detect_gpu(self):
        """Detect available GPU and set appropriate device."""
        try:
            import torch
            import torch_directml
            
            # Check for AMD GPU first
            if torch_directml.is_available():
                self.gpu_type = 'AMD'
                self.gpu_name = torch_directml.device_name(0)
                self.dml_device = torch_directml.device(0)
                print(f"Detected GPU Type: {self.gpu_type}")
                print(f"GPU Name: {self.gpu_name}")
                print("DirectML initialized successfully. Using AMD GPU:", self.gpu_name)
                
                # Force DirectML to initialize the GPU
                try:
                    # Create a large tensor to force GPU memory allocation
                    warm_tensor = torch.rand(1000, 1000, device=self.dml_device)
                    # Perform some operations to ensure GPU is active
                    result = warm_tensor @ warm_tensor.t()
                    # Force synchronization
                    _ = result.cpu().numpy()
                    print("GPU memory allocation test successful - GPU should be active now")
                except Exception as e:
                    print(f"Warning: GPU warm-up failed: {e}")
                
                return
            
            # Check for NVIDIA GPU
            if torch.cuda.is_available():
                self.gpu_type = 'NVIDIA'
                self.gpu_name = torch.cuda.get_device_name(0)
                self.device = torch.device('cuda')
                print(f"Detected GPU Type: {self.gpu_type}")
                print(f"GPU Name: {self.gpu_name}")
                return
            
            # No GPU detected
            print("ERROR: No compatible GPU detected. This application requires a GPU.")
            raise Exception("No GPU detected")
            
        except Exception as e:
            print(f"ERROR: Failed to initialize GPU: {e}")
            raise Exception(f"GPU initialization failed: {e}")

    def update_fps(self):
        """Update FPS counter."""
        current_time = time.time()
        self.frame_count += 1
        
        # Update FPS every second
        if current_time - self.last_time >= 1.0:
            self.last_fps = self.current_fps
            self.current_fps = self.frame_count
            self.frame_count = 0
            self.last_time = current_time
            print(f"Current FPS: {self.current_fps}")

    def load_model(self):
        """Load the YOLOv8 model on GPU with performance optimizations."""
        try:
            print(f"Loading model from: {self.model_path}")
            
            # Load model with task-specific optimizations
            self.model = YOLO(self.model_path, task='detect')
            
            # Apply global performance optimizations
            torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms
            
            if self.gpu_type == 'NVIDIA':
                # Move model to CUDA
                self.model.to(self.device)
                print(f"Model moved to CUDA device: {self.device}")
                
                # Force model to CUDA
                self.model.model = self.model.model.to(self.device)
                print("Model forced to CUDA device")
                
                # Set model to evaluation mode
                self.model.model.eval()
                print("Model set to evaluation mode")
                
                # Enable CUDA graph for faster inference
                try:
                    torch.cuda.synchronize()
                    print("CUDA synchronization successful")
                except Exception as e:
                    print(f"CUDA synchronization warning: {e}")
            
            elif self.gpu_type == 'AMD':
                # Move model to DirectML
                self.model.to(self.dml_device)
                print(f"Model moved to DirectML device: {self.dml_device}")
                
                # Set model to evaluation mode
                self.model.model.eval()
                print("Model set to evaluation mode")
                
                # Force model to DirectML device
                self.model.model = self.model.model.to(self.dml_device)
                print("Model forced to DirectML device")
                
                # Force all parameters to DirectML device
                for param in self.model.model.parameters():
                    param.data = param.data.to(self.dml_device)
                print("All model parameters forced to DirectML device")
                
                # Use smaller test tensor for better performance
                test_tensor = torch.randn(1, 3, 320, 320, device=self.dml_device)
                print(f"Test tensor created on DirectML device: {test_tensor.device}")
                
                # Test forward pass with explicit computation
                with torch.no_grad():
                    test_output = self.model.model(test_tensor)
                    # Force computation by accessing the output
                    if isinstance(test_output, (list, tuple)):
                        _ = test_output[0].cpu()
                    else:
                        _ = test_output.cpu()
                print("Test forward pass successful with explicit computation")
                
                # Initialize predictor with DirectML device
                self.model.predictor = self.model.model
                print("Predictor initialized with DirectML device")
                
                # Verify model is on DirectML device
                model_device = next(self.model.model.parameters()).device
                print(f"Model device: {model_device}")
                print(f"Expected device: {self.dml_device}")
                
                if model_device == self.dml_device:
                    print("Model successfully moved to AMD GPU using DirectML")
                else:
                    print("WARNING: Model not on DirectML device")
                    raise Exception("Failed to move model to GPU")
                
                # Set model parameters for better performance
                self.model.conf = self.config.get('confidence', 0.5)
                self.model.iou = self.config.get('iou_threshold', 0.45)
                print(f"Model confidence threshold set to: {self.model.conf}")
                print(f"Model IoU threshold set to: {self.model.iou}")
                
                # Set class names
                self.class_names = ['item']
                print(f"Model class names set to: {self.class_names}")
                
                # Force another computation to ensure GPU is active (use smaller tensor)
                try:
                    warm_tensor = torch.rand(500, 500, device=self.dml_device)
                    result = warm_tensor @ warm_tensor.t()
                    _ = result.cpu().numpy()
                    print("Final GPU warm-up successful - GPU should be active now")
                except Exception as e:
                    print(f"Warning: Final GPU warm-up failed: {e}")
            
            # Apply additional performance optimizations
            try:
                # Optimize model for inference
                if hasattr(self.model.model, 'fuse'):
                    self.model.model.fuse()
                    print("Model layers fused for better performance")
                
                # Set half-precision for better performance if supported
                if self.gpu_type == 'NVIDIA':
                    self.model.model = self.model.model.half()
                    print("Model converted to half-precision for better performance")
            except Exception as e:
                print(f"Warning: Some optimizations couldn't be applied: {e}")
            
            print(f"YOLOv8 model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load model on GPU: {e}")
            return False

    def process_frame(self, frame):
        """Extreme optimization for CS2 aiming for 150+ FPS with maximum GPU utilization."""
        try:
            # Use configurable target size for maximum speed
            target_size = config.get('target_size', 160)
            capture_size = config.get('capture_size', 480)
            max_boxes = config.get('max_boxes', 3)
            
            # Skip processing every other frame if we're behind
            self.skip_counter += 1
            if self.skip_counter % 2 != 0 and hasattr(self, 'last_detections'):
                # Return cached results to maintain responsiveness
                return self.last_detections
            else:
                self.skip_counter = 0
            
            # Initialize GPU processing
            if not hasattr(self, 'gpu_initialized'):
                self._initialize_gpu_processing(target_size)
                self.gpu_initialized = True
            
            if self.gpu_type == 'AMD':
                # AMD DirectML path - fully GPU accelerated
                try:
                    # Convert frame directly to GPU tensor if possible
                    if isinstance(frame, torch.Tensor) and frame.device == self.dml_device:
                        # Frame is already a GPU tensor
                        gpu_frame = frame
                    else:
                        # Convert numpy array to GPU tensor
                        gpu_frame = torch.from_numpy(frame).to(self.dml_device)
                    
                    # Resize directly on GPU using interpolate
                    if gpu_frame.shape[0] != target_size or gpu_frame.shape[1] != target_size:
                        # Reshape for interpolate (N,C,H,W)
                        if len(gpu_frame.shape) == 3:  # (H,W,C)
                            gpu_frame = gpu_frame.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
                        
                        # Resize on GPU
                        gpu_frame = torch.nn.functional.interpolate(
                            gpu_frame.float(), 
                            size=(target_size, target_size),
                            mode='bilinear',
                            align_corners=False
                        )
                    else:
                        # Just reshape for model input
                        if len(gpu_frame.shape) == 3:  # (H,W,C)
                            gpu_frame = gpu_frame.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
                    
                    # Normalize on GPU (in-place when possible)
                    if gpu_frame.dtype != torch.float32:
                        gpu_frame = gpu_frame.float()
                    gpu_frame.div_(255.0)  # In-place division
                    
                    # Run inference directly with GPU tensor
                    with torch.no_grad():
                        # Use cached tensor for maximum speed
                        self.input_tensor_buffer.copy_(gpu_frame, non_blocking=True)
                        results = self.model.model(self.input_tensor_buffer)
                
                        # Synchronize GPU to make sure operations are complete
                        if self.gpu_type == 'NVIDIA' and hasattr(torch, 'cuda'):
                            torch.cuda.synchronize()
                        elif self.gpu_type == 'AMD' and hasattr(torch, 'dml'):
                            torch.dml.synchronize()
                        elif self.gpu_type == 'AMD':
                            # DirectML doesn't have synchronize, use device barrier instead
                            try:
                                # Try to use device barrier if available
                                if hasattr(self.model, 'model') and hasattr(self.model.model, 'to'):
                                    dummy = torch.zeros(1, device=self.dml_device)
                                    _ = dummy + 1  # Force execution
                            except Exception as e:
                                pass  # Silently fail if barrier doesn't work
                        
                        # Ultra-minimal result processing
                        if isinstance(results, (list, tuple)):
                            results = results[0] if results else None
                
                except Exception as e:
                    print(f"GPU processing error (AMD): {e}, falling back to CPU path")
                    # Fall back to CPU path
                    if not hasattr(self, 'resize_buffer'):
                        self.resize_buffer = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                    
                    # Resize on CPU
                    frame_resized = cv2.resize(frame, (target_size, target_size), 
                                             dst=self.resize_buffer,
                                             interpolation=cv2.INTER_AREA)
                    
                    # Standard CPU to GPU transfer
                    np_tensor = frame_resized.transpose(2, 0, 1)  # HWC to CHW
                    self.cpu_tensor[0].copy_(torch.from_numpy(np_tensor))
                    self.cpu_tensor.div_(255.0)  # In-place division
                    
                    # Copy to device
                    self.input_tensor_buffer.copy_(self.cpu_tensor.to(self.dml_device))
                
                    # Run inference
                    with torch.no_grad():
                        results = self.model.model(self.input_tensor_buffer)
                        if isinstance(results, (list, tuple)):
                            results = results[0] if results else None
            elif self.gpu_type == 'NVIDIA':
                # NVIDIA CUDA path - fully GPU accelerated
                try:
                    # Convert frame directly to GPU tensor if possible
                    if isinstance(frame, torch.Tensor) and frame.device == self.device:
                        # Frame is already a GPU tensor
                        gpu_frame = frame
                    else:
                        # Convert numpy array to GPU tensor
                        gpu_frame = torch.from_numpy(frame).to(self.device)
                    
                    # Resize directly on GPU using interpolate
                    if gpu_frame.shape[0] != target_size or gpu_frame.shape[1] != target_size:
                        # Reshape for interpolate (N,C,H,W)
                        if len(gpu_frame.shape) == 3:  # (H,W,C)
                            gpu_frame = gpu_frame.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
                        
                        # Resize on GPU
                        gpu_frame = torch.nn.functional.interpolate(
                            gpu_frame.float(), 
                            size=(target_size, target_size),
                            mode='bilinear',
                            align_corners=False
                        )
                    else:
                        # Just reshape for model input
                        if len(gpu_frame.shape) == 3:  # (H,W,C)
                            gpu_frame = gpu_frame.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
                    
                    # Normalize on GPU (in-place when possible)
                    if gpu_frame.dtype != torch.float16:
                        gpu_frame = gpu_frame.half()  # Use half precision for NVIDIA
                    gpu_frame.div_(255.0)  # In-place division
                    
                    # Run inference directly with GPU tensor
                    with torch.no_grad():
                        # Use cached tensor for maximum speed
                        self.input_tensor_buffer.copy_(gpu_frame, non_blocking=True)
                        results = self.model.model(self.input_tensor_buffer)
                        
                        # Synchronize GPU to make sure operations are complete
                        if self.gpu_type == 'NVIDIA' and hasattr(torch, 'cuda'):
                            torch.cuda.synchronize()
                        elif self.gpu_type == 'AMD' and hasattr(torch, 'dml'):
                            torch.dml.synchronize()
                        elif self.gpu_type == 'AMD':
                            # DirectML doesn't have synchronize, use device barrier instead
                            try:
                                # Try to use device barrier if available
                                if hasattr(self.model, 'model') and hasattr(self.model.model, 'to'):
                                    dummy = torch.zeros(1, device=self.dml_device)
                                    _ = dummy + 1  # Force execution
                            except Exception as e:
                                pass  # Silently fail if barrier doesn't work
                        
                        # Ultra-minimal result processing
                        if isinstance(results, (list, tuple)):
                            results = results[0] if results else None
                
                except Exception as e:
                    print(f"GPU processing error (NVIDIA): {e}, falling back to CPU path")
                    # Fall back to CPU path
                    if not hasattr(self, 'resize_buffer'):
                        self.resize_buffer = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                    
                    # Resize on CPU
                    frame_resized = cv2.resize(frame, (target_size, target_size), 
                                             dst=self.resize_buffer,
                                             interpolation=cv2.INTER_AREA)
                    
                    # Standard CPU to GPU transfer
                    np_tensor = frame_resized.transpose(2, 0, 1)  # HWC to CHW
                    self.cpu_tensor[0].copy_(torch.from_numpy(np_tensor))
                    self.cpu_tensor.div_(255.0)  # In-place division
                    
                    # Copy to device
                    self.input_tensor_buffer.copy_(self.cpu_tensor.to(self.device))
                    
                    # Run inference
                    with torch.no_grad():
                        results = self.model.model(self.input_tensor_buffer)
                        if isinstance(results, (list, tuple)):
                            results = results[0] if results else None
            
            else:
                # CPU fallback path
                if not hasattr(self, 'resize_buffer'):
                    self.resize_buffer = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                
                # Skip resize if frame is already the right size
                if frame.shape[0] == target_size and frame.shape[1] == target_size:
                    frame_resized = frame
                else:
                    frame_resized = cv2.resize(frame, (target_size, target_size), 
                                             dst=self.resize_buffer,
                                             interpolation=cv2.INTER_AREA)
                
                # Run inference on CPU (slow)
                results = self.model(frame_resized)
            
            # Process results (absolute minimal processing)
            detections = []
            if results is not None and hasattr(results, 'boxes'):
                boxes = results.boxes
                # Process results on GPU when possible
                try:
                    if self.gpu_type in ['AMD', 'NVIDIA'] and hasattr(boxes, 'xyxy') and hasattr(boxes, 'conf'):
                        # Get all boxes at once
                        all_boxes = boxes.xyxy[:max_boxes]  # Limit to max_boxes
                        all_confs = boxes.conf[:max_boxes]
                        all_cls = boxes.cls[:max_boxes]
                        
                        # Filter by confidence on GPU
                        conf_threshold = self.config.get('confidence', 0.5)
                        mask = all_confs > conf_threshold
                        
                        # Apply filter
                        filtered_boxes = all_boxes[mask]
                        filtered_confs = all_confs[mask]
                        filtered_cls = all_cls[mask]
                        
                        # Scale coordinates on GPU
                        scale_factor = capture_size / target_size
                        filtered_boxes *= scale_factor
                        
                        # Transfer to CPU only once
                        cpu_boxes = filtered_boxes.cpu().numpy()
                        cpu_confs = filtered_confs.cpu().numpy()
                        cpu_cls = filtered_cls.cpu().numpy()
                        
                        # Create detection objects
                        for i in range(len(cpu_boxes)):
                            x1, y1, x2, y2 = cpu_boxes[i]
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(cpu_confs[i]),
                                'class': int(cpu_cls[i])
                            })
                    else:
                        # CPU fallback for processing boxes
                        for i in range(min(max_boxes, len(boxes))):
                            box = boxes[i]
                            if box.conf > self.config.get('confidence', 0.5):
                                try:
                                    # Get coordinates directly
                                    xyxy = box.xyxy[0].cpu().numpy()
                        
                                    # Scale back to original frame size
                                    scale_factor = capture_size / target_size
                                    x1, y1, x2, y2 = xyxy * scale_factor
                        
                                    detections.append({
                                        'bbox': [x1, y1, x2, y2],
                                        'confidence': float(box.conf),
                                        'class': int(box.cls)
                                    })
                                except:
                                    continue
                except Exception as e:
                    print(f"Error processing detection results: {e}")
                    # Fallback to basic processing
                    for i in range(min(max_boxes, len(boxes))):
                        box = boxes[i]
                        if box.conf > self.config.get('confidence', 0.5):
                            try:
                                # Get coordinates directly
                                xyxy = box.xyxy[0].cpu().numpy()
                                
                                # Scale back to original frame size
                                scale_factor = capture_size / target_size
                                x1, y1, x2, y2 = xyxy * scale_factor
                                
                                detections.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': float(box.conf),
                                    'class': int(box.cls)
                                })
                            except:
                                continue
            
            # Cache detections for frame skipping
            self.last_detections = detections
            return detections
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return []
            
    def _initialize_gpu_processing(self, target_size):
        """Initialize GPU resources for processing."""
        try:
            if self.gpu_type == 'AMD':
                # Pre-allocate tensor for zero-copy
                self.input_tensor_buffer = torch.zeros((1, 3, target_size, target_size), 
                                                     device=self.dml_device,
                                                     dtype=torch.float32)
                
                # Pre-allocate CPU tensor to avoid repeated allocations
                self.cpu_tensor = torch.zeros((1, 3, target_size, target_size), 
                                           dtype=torch.float32)
                
                # Create utility tensors for GPU operations
                self.gpu_resize_tensor = torch.zeros((1, 3, target_size, target_size), 
                                                   device=self.dml_device,
                                                   dtype=torch.float32)
                
                print("Initialized AMD GPU resources for accelerated processing")
                
            elif self.gpu_type == 'NVIDIA':
                # Pre-allocate tensor for zero-copy with half precision
                self.input_tensor_buffer = torch.zeros((1, 3, target_size, target_size), 
                                                     device=self.device,
                                                     dtype=torch.float16)
                
                # Pre-allocate CPU tensor to avoid repeated allocations
                self.cpu_tensor = torch.zeros((1, 3, target_size, target_size), 
                                           dtype=torch.float16)
                
                # Create utility tensors for GPU operations
                self.gpu_resize_tensor = torch.zeros((1, 3, target_size, target_size), 
                                                   device=self.device,
                                                   dtype=torch.float16)
                
                # Enable CUDA graphs for faster inference if available
                if hasattr(torch.cuda, 'make_graphed_callables'):
                    try:
                        sample_input = torch.randn((1, 3, target_size, target_size), 
                                                device=self.device, 
                                                dtype=torch.float16)
                        
                        # Try to graph the model for faster inference
                        self.model.model = torch.cuda.make_graphed_callables(
                            self.model.model, (sample_input,))
                        print("CUDA graph optimization enabled for faster inference")
                    except Exception as e:
                        print(f"CUDA graph optimization failed: {e}")
                
                print("Initialized NVIDIA GPU resources for accelerated processing")
        except Exception as e:
            print(f"Failed to initialize GPU resources: {e}")
            # Reset GPU initialized flag to try again next time
            self.gpu_initialized = False

    def force_overlay_redraw(self):
        # This method should trigger a redraw of the overlay window
        try:
            if hasattr(self, 'hwnd'):
                win32gui.InvalidateRect(self.hwnd, None, True)
        except Exception as e:
            print(f"[DEBUG] Error forcing overlay redraw: {e}")

    def _detection_loop(self):
        """CS2-specific extreme performance detection loop targeting 150+ FPS using maximum GPU acceleration."""
        # Pre-allocate tensors for better performance
        import torch
        import numpy as np
        import cv2
        import threading
        import time
        import ctypes
        
        # Set thread priority to time-critical
        try:
            import win32api
            import win32process
            import win32con
            # Set this process to high priority
            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
            win32process.SetPriorityClass(handle, win32process.REALTIME_PRIORITY_CLASS)
            print("Process priority set to REALTIME")
        except Exception as e:
            print(f"Could not set process priority to REALTIME: {e}")
        
        # Initialize screen capture with direct thread
        self.screen_capture = self.ScreenCapture()
        
        # Pass GPU information to screen capture for acceleration
        if hasattr(self, 'gpu_type'):
            self.screen_capture.parent = self
            print(f"Passed {self.gpu_type} GPU reference to screen capture")
        
        # Configure sizes from config
        target_size = config.get('target_size', 160)
        
        # Create pre-allocated tensors with configurable size for maximum performance
        if self.gpu_type == 'AMD':
            # Create tensors on AMD GPU using DirectML
            self.input_tensor = torch.zeros((1, 3, target_size, target_size), device=self.dml_device)
            
            # Create a tensor for direct GPU-to-GPU transfer (avoid CPU roundtrip)
            self.gpu_frame_tensor = torch.zeros((config.get('capture_size', 480), 
                                               config.get('capture_size', 480), 
                                               3), 
                                              device=self.dml_device,
                                              dtype=torch.uint8)
            
            # Create a tensor for GPU-based preprocessing
            self.preprocess_tensor = torch.zeros((1, 3, target_size, target_size), 
                                               device=self.dml_device,
                                               dtype=torch.float32)
            
            # Create a pinned memory tensor for faster CPU-GPU transfers
            try:
                self.pinned_tensor = torch.zeros((1, 3, target_size, target_size), 
                                               dtype=torch.float32, 
                                               pin_memory=True)
            except RuntimeError:
                # Fallback if pin_memory is not available
                print("Pin memory not available, using standard tensor")
                self.pinned_tensor = torch.zeros((1, 3, target_size, target_size), 
                                               dtype=torch.float32)
            
            # Create a stream for asynchronous operations
            self.stream = None  # AMD DirectML doesn't support CUDA streams
            
        elif self.gpu_type == 'NVIDIA':
            # Create tensors on NVIDIA GPU using CUDA
            self.input_tensor = torch.zeros((1, 3, target_size, target_size), device=self.device)
            
            # Create a tensor for direct GPU-to-GPU transfer (avoid CPU roundtrip)
            self.gpu_frame_tensor = torch.zeros((config.get('capture_size', 480), 
                                               config.get('capture_size', 480), 
                                               3), 
                                              device=self.device,
                                              dtype=torch.uint8)
        
            # Create a tensor for GPU-based preprocessing
            self.preprocess_tensor = torch.zeros((1, 3, target_size, target_size), 
                                               device=self.device,
                                               dtype=torch.float16)
            
            # Create a pinned memory tensor for faster CPU-GPU transfers
            try:
                self.pinned_tensor = torch.zeros((1, 3, target_size, target_size), 
                                               dtype=torch.float16, 
                                               pin_memory=True)
            except RuntimeError:
                # Fallback if pin_memory is not available
                print("Pin memory not available, using standard tensor")
                self.pinned_tensor = torch.zeros((1, 3, target_size, target_size), 
                                               dtype=torch.float16)
            
            # Create a stream for asynchronous operations
            self.stream = torch.cuda.Stream()
            
            # Enable CUDA graph for faster inference if available
            if hasattr(torch.cuda, 'make_graphed_callables') and target_size <= 160:
                try:
                    sample_input = torch.randn((1, 3, target_size, target_size), 
                                             device=self.device, 
                                             dtype=torch.float16)
                    
                    # Try to graph the model for faster inference
                    self.model.model = torch.cuda.make_graphed_callables(
                        self.model.model, (sample_input,))
                    print("CUDA graph optimization enabled for faster inference")
                except Exception as e:
                    print(f"CUDA graph optimization failed: {e}")
        
        # Prepare for ultra-performance loop
        print(f"Starting CS2-optimized detection loop for 150+ FPS with {self.gpu_type} GPU acceleration")
        self.last_time = time.time()
        self.frame_count = 0
        self.processed_frames = 0
        
        # Set main thread to high priority
        try:
            import win32api
            import win32process
            import win32con
            
            # Set process priority
            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
            win32process.SetPriorityClass(handle, win32process.REALTIME_PRIORITY_CLASS)
            
            # Set thread priority using win32process
            tid = win32api.GetCurrentThreadId()
            thandle = win32api.OpenThread(win32con.THREAD_SET_INFORMATION, False, tid)
            win32process.SetThreadPriority(thandle, win32process.THREAD_PRIORITY_TIME_CRITICAL)
            
            print("Detection thread priority set to TIME_CRITICAL")
        except Exception as e:
            print(f"Could not set thread priority: {e}")
            
        # Pre-allocate frame to avoid memory allocations
        self.cached_frame = None
        
        # For adaptive timing
        detection_times = collections.deque(maxlen=30)
        
        # For GPU memory management
        if self.gpu_type == 'NVIDIA':
            # Schedule periodic GPU memory cleanup
            last_memory_cleanup = time.time()
            
            # Monitor GPU utilization
            try:
                import GPUtil
                last_gpu_check = time.time()
                gpu_utils = []
            except ImportError:
                print("GPUtil not available for GPU monitoring")
        
        # Main detection loop - directly process frames from screen capture with GPU acceleration
        while self.running:
            try:
                # Initialize detections with empty list as default for this iteration
                detections = []
                
                # Get latest frame directly from screen capture
                frame = self.screen_capture.get_frame()
                
                if frame is not None:
                    self.cached_frame = frame  # Cache the latest frame
                    
                    # Process frame (with optional frame skipping for stability)
                    self.frame_counter += 1
                    if self.frame_counter % self.process_every_n == 0:
                        # Time the detection process
                        start_time = time.time()
                        
                        # Process the frame with GPU acceleration
                        try:
                            # Try to use direct GPU processing if the frame is already on GPU
                            if isinstance(frame, torch.Tensor) and (
                                (self.gpu_type == 'AMD' and frame.device == self.dml_device) or
                                (self.gpu_type == 'NVIDIA' and frame.device == self.device)
                            ):
                                # Frame is already on GPU - use direct GPU-to-GPU processing
                                detections = self.process_frame(frame)
                            else:
                                # Standard processing path
                                detections = self.process_frame(frame)
                                
                        except Exception as e:
                            print(f"Error in GPU-accelerated processing: {e}")
                            # Fall back to standard processing
                            try:
                                detections = self.process_frame(frame)
                            except Exception as e2:
                                print(f"Error in fallback processing: {e2}")
                                # detections remains as empty list
                        
                        # Update detection timing
                        end_time = time.time()
                        detection_time = end_time - start_time
                        detection_times.append(detection_time)
                
                # Update current boxes
                if detections:
                    self.current_boxes = [d['bbox'] for d in detections]
                    self.current_confidences = [d['confidence'] for d in detections]
                    self.current_class_ids = [d['class'] for d in detections]
                    self.last_box_state = True
                else:
                    if not hasattr(self, 'last_box_state') or not self.last_box_state:
                        self.current_boxes = None
                        self.current_confidences = None
                        self.current_class_ids = None
                        self.last_box_state = False
                
                        # Adaptive frame skipping based on detection time
                        if len(detection_times) >= 5:
                            avg_detection_time = sum(detection_times) / len(detection_times)
                            
                            # Adjust frame skipping based on detection time
                            if avg_detection_time > 0.015:  # If detection takes > 15ms
                                self.process_every_n = min(3, self.process_every_n + 1)  # Skip more frames
                            elif avg_detection_time < 0.008:  # If detection is fast (< 8ms)
                                self.process_every_n = max(1, self.process_every_n - 1)  # Process more frames
                
                # Update FPS more frequently for better feedback
                self.frame_count += 1
                if self.frame_count % 30 == 0:  # Every 30 frames
                    current_time = time.time()
                    elapsed = current_time - self.last_time
                    
                    if elapsed > 0:
                        # Calculate processing FPS
                        self.current_fps = int(30 / elapsed)
                        self.last_fps = self.current_fps
                        
                        # Calculate detection FPS (accounting for frame skipping)
                        if hasattr(self, 'screen_capture') and hasattr(self.screen_capture, 'fps'):
                            capture_fps = self.screen_capture.fps
                        else:
                            capture_fps = 0
                            
                        # Print detailed performance metrics
                        if len(detection_times) > 0:
                            avg_detection_ms = sum(detection_times) / len(detection_times) * 1000
                            print(f"FPS: {self.current_fps}, Capture: {capture_fps}, " +
                                  f"Detection: {avg_detection_ms:.1f}ms, Skip: {self.process_every_n}")
                        else:
                            print(f"FPS: {self.current_fps}, Capture: {capture_fps}, Skip: {self.process_every_n}")
                        
                        # Reset counters
                        self.last_time = current_time
                
                        # GPU memory management for NVIDIA
                        if self.gpu_type == 'NVIDIA' and hasattr(torch, 'cuda'):
                            current_time = time.time()
                            
                            # Check GPU utilization periodically
                            if hasattr(GPUtil, 'getGPUs') and current_time - last_gpu_check > 5.0:
                                try:
                                    gpus = GPUtil.getGPUs()
                                    if gpus:
                                        gpu_util = gpus[0].load * 100
                                        gpu_utils.append(gpu_util)
                                        if len(gpu_utils) > 5:
                                            avg_util = sum(gpu_utils) / len(gpu_utils)
                                            print(f"GPU Utilization: {avg_util:.1f}%")
                                            gpu_utils = []
                                    last_gpu_check = current_time
                                except Exception as e:
                                    print(f"Error checking GPU: {e}")
                            
                            # Periodic memory cleanup
                            if current_time - last_memory_cleanup > 30.0:
                                try:
                                    # Empty CUDA cache to free up memory
                                    torch.cuda.empty_cache()
                                    print("CUDA memory cache cleared")
                                    last_memory_cleanup = current_time
                                except Exception as e:
                                    print(f"Error cleaning GPU memory: {e}")
                
                # Ultra-short sleep to prevent CPU overload while maintaining high FPS
                # Skip sleep entirely if FPS is below target
                if self.current_fps < 120:
                    continue
                else:
                    time.sleep(0.0001)
                
            except Exception as e:
                print(f"Error in main detection loop: {e}")
                time.sleep(0.001)

    def start_detection(self):
        """Start the detection thread."""
        if not self.running:
            self.running = True
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            print("Detection thread started")

    def stop_detection(self):
        """Stop the detection thread."""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join()
            print("Detection thread stopped")

    @property
    def fps(self):
        """Get current FPS."""
        if not self.fps_history:
            return 0
        return sum(self.fps_history) / len(self.fps_history)

    @property
    def processing_fps(self):
        """Get current processing FPS."""
        if not self.processing_fps_history:
            return 0
        return sum(self.processing_fps_history) / len(self.processing_fps_history)

    # Import required modules at class level
    import numpy as np
    import cv2
    import win32gui
    import win32ui
    import win32con
    import win32api
    import psutil
    import threading
    
    class ScreenCapture:
        def __init__(self):
            # Capture settings
            self.capture_region = None
            self.frame_buffer = None
            self.running = True
            self.lock = threading.Lock()
            self.current_frame = None
            self.frame_count = 0
            self.last_time = time.time()
            self.fps = 0
            
            # Thread management
            self.capture_thread = None
            self.capture_event = threading.Event()
            self.sleep_time = 0.0001  # Ultra-short sleep time (0.1ms)
            
            # GDI resources
            self.hwin = None
            self.hwindc = None
            self.srcdc = None
            self.memdc = None
            self.bmp = None
            
            # GPU resources
            self.gpu_type = None
            self.gpu_device = None
            self._setup_gpu()
            
            # Initialize GDI resources
            self._init_capture()
            
            # Set thread priority
            self._set_process_priority()
            
            # Start the capture thread
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            
            # Create a second thread for FPS calculation to avoid slowing down capture
            self.fps_thread = threading.Thread(target=self.fps_loop, daemon=True)
            self.fps_thread.start()
            
        def _setup_gpu(self):
            """Setup GPU for image processing acceleration."""
            try:
                # Check if we're already in a YOLODetector instance with GPU setup
                if hasattr(self, 'parent') and hasattr(self.parent, 'gpu_type'):
                    self.gpu_type = self.parent.gpu_type
                    if self.gpu_type == 'AMD':
                        self.gpu_device = self.parent.dml_device
                    elif self.gpu_type == 'NVIDIA':
                        self.gpu_device = self.parent.device
                else:
                    # Try to detect GPU independently
                    try:
                        import torch
                        import torch_directml
                        
                        if torch_directml.is_available():
                            self.gpu_type = 'AMD'
                            self.gpu_device = torch_directml.device()
                            print("Screen capture will use AMD GPU acceleration")
                        elif torch.cuda.is_available():
                            self.gpu_type = 'NVIDIA'
                            self.gpu_device = torch.device('cuda')
                            print("Screen capture will use NVIDIA GPU acceleration")
                    except ImportError:
                        self.gpu_type = None
                        self.gpu_device = None
            except Exception as e:
                print(f"GPU setup for screen capture failed: {e}")
                self.gpu_type = None
                self.gpu_device = None

        def _init_capture(self):
            """Initialize capture resources."""
            try:
                # Get screen dimensions
                screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
                screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
                
                # Create capture region for CS2 (exactly 480x480 at center)
                center_x = screen_width // 2
                center_y = screen_height // 2
                capture_size = 480
                
                # Define capture region centered on screen
                self.capture_region = {
                    'left': max(0, center_x - capture_size // 2),
                    'top': max(0, center_y - capture_size // 2),
                    'width': capture_size,
                    'height': capture_size
                }
                print(f"Screen capture initialized: {screen_width}x{screen_height}, region: {capture_size}x{capture_size}")
                
                # Pre-allocate memory for better performance
                self.frame_buffer = np.zeros((capture_size, capture_size, 3), dtype=np.uint8)
                
                # Create device contexts and bitmap for ultra-fast GDI capture
                self._create_gdi_resources(capture_size)
                
                # Note: Win32 Python API doesn't support direct buffer passing for GetBitmapBits
            except Exception as e:
                print(f"Error initializing capture: {e}")
                import traceback
                traceback.print_exc()
                
        def _create_gdi_resources(self, capture_size):
            """Create or recreate GDI resources for screen capture."""
            try:
                # Clean up existing resources if they exist
                self._clean_gdi_resources()
                
                # Create fresh GDI resources
                self.hwin = win32gui.GetDesktopWindow()
                self.hwindc = win32gui.GetWindowDC(self.hwin)
                self.srcdc = win32ui.CreateDCFromHandle(self.hwindc)
                self.memdc = self.srcdc.CreateCompatibleDC()
                self.bmp = win32ui.CreateBitmap()
                self.bmp.CreateCompatibleBitmap(self.srcdc, capture_size, capture_size)
                self.memdc.SelectObject(self.bmp)
                print("GDI resources created successfully")
                return True
            except Exception as e:
                print(f"Error creating GDI resources: {e}")
                return False
                
        def _clean_gdi_resources(self):
            """Clean up GDI resources properly."""
            try:
                if hasattr(self, 'memdc') and self.memdc:
                    self.memdc.DeleteDC()
                    self.memdc = None
                if hasattr(self, 'srcdc') and self.srcdc:
                    self.srcdc.DeleteDC()
                    self.srcdc = None
                if hasattr(self, 'hwindc') and self.hwin and self.hwindc:
                    win32gui.ReleaseDC(self.hwin, self.hwindc)
                    self.hwindc = None
                if hasattr(self, 'bmp') and self.bmp:
                    win32gui.DeleteObject(self.bmp.GetHandle())
                    self.bmp = None
                print("GDI resources cleaned up")
            except Exception as e:
                print(f"Error cleaning up GDI resources: {e}")
                
        def _set_process_priority(self):
            """Set process and thread priority to maximum."""
            try:
                # Set process priority
                p = psutil.Process()
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                
                # Set thread priority using Windows API
                try:
                    import win32process
                    import win32api
                    handle = win32api.GetCurrentThread()
                    win32process.SetThreadPriority(handle, win32process.THREAD_PRIORITY_HIGHEST)
                except:
                    pass
                    
                print("Process and thread priority set to maximum")
            except Exception as e:
                print(f"Failed to set process priority: {e}")

        def capture_loop(self):
            """Continuously capture frames in a separate thread with adaptive timing and GPU acceleration."""
            # Set thread priority to time critical
            try:
                import win32process
                import win32api
                handle = win32api.GetCurrentThread()
                win32process.SetThreadPriority(handle, win32process.THREAD_PRIORITY_TIME_CRITICAL)
                print("Capture thread priority set to TIME_CRITICAL")
            except:
                pass
                
            # Initialize for direct memory access
            left = self.capture_region['left']
            top = self.capture_region['top']
            width = self.capture_region['width']
            height = self.capture_region['height']
            
            # For adaptive timing
            frame_times = collections.deque(maxlen=10)
            
            # GPU acceleration setup
            use_gpu = self.gpu_device is not None and config.get('use_gpu_capture', True)
            if use_gpu:
                try:
                    import torch
                    # Pre-allocate GPU tensors for processing
                    if self.gpu_type == 'AMD' or self.gpu_type == 'NVIDIA':
                        # Create tensor for frame processing
                        self.gpu_frame = torch.zeros((height, width, 3), 
                                                   device=self.gpu_device, 
                                                   dtype=torch.uint8)
                        print(f"GPU acceleration enabled for screen capture using {self.gpu_type}")
                except Exception as e:
                    print(f"Failed to initialize GPU acceleration for capture: {e}")
                    use_gpu = False
            
            # Track consecutive failures to trigger resource recreation
            consecutive_failures = 0
            max_failures_before_reset = 3
            
            while self.running:
                start_time = time.time()
                
                try:
                    # Check if we need to recreate GDI resources
                    if consecutive_failures >= max_failures_before_reset:
                        print(f"Detected {consecutive_failures} consecutive BitBlt failures. Recreating GDI resources...")
                        if self._create_gdi_resources(width):
                            consecutive_failures = 0
                        else:
                            # If recreation failed, sleep a bit to avoid hammering the system
                            time.sleep(0.1)
                            continue
                    
                    # Capture screen region using GDI BitBlt (most efficient method)
                    result = self.memdc.BitBlt((0, 0), (width, height), self.srcdc, (left, top), win32con.SRCCOPY)
                    
                    # Reset failure counter on success
                    consecutive_failures = 0
                    
                    # Get bitmap bits (the Python binding doesn't support direct buffer passing)
                    signedIntsArray = self.bmp.GetBitmapBits(True)
                    
                    # Convert to numpy array
                    img = np.frombuffer(signedIntsArray, dtype=np.uint8)
                    img = img.reshape((height, width, 4))
                    
                    # Process the frame (with GPU if available)
                    if use_gpu:
                        try:
                            # Extract BGR channels using GPU
                            import torch
                            # Make a writable copy of the array before converting to tensor
                            img_copy = img[:, :, :3].copy()
                            # Convert numpy array to tensor on GPU
                            img_tensor = torch.from_numpy(img_copy).to(self.gpu_device)
                            
                            # Apply any GPU-accelerated processing here
                            # For example, we could do noise reduction, contrast enhancement, etc.
                            
                            # Get result back to CPU as numpy array
                            frame = img_tensor.cpu().numpy().copy()
                        except Exception as e:
                            # Fall back to CPU if GPU processing fails
                            print(f"GPU processing failed, falling back to CPU: {e}")
                            frame = img[:, :, :3].copy()
                    else:
                        # Extract BGR channels (fastest method, avoid cv2.cvtColor)
                        frame = img[:, :, :3].copy()  # Need copy for thread safety
                    
                    # Update current frame with minimal lock time
                    with self.lock:
                        self.current_frame = frame
                        self.frame_count += 1
                        
                except Exception as e:
                    consecutive_failures += 1
                    print(f"Error in capture loop: {e} (Failures: {consecutive_failures}/{max_failures_before_reset})")
                    
                    # If we've had too many failures, try recreating the DC
                    if consecutive_failures >= max_failures_before_reset:
                        continue
                
                # Adaptive timing to maximize FPS
                end_time = time.time()
                frame_time = end_time - start_time
                
                # Store frame time for both local calculation and FPS reporting
                frame_times.append(frame_time)
                if hasattr(self, 'frame_times'):
                    self.frame_times.append(frame_time)
                
                # Calculate optimal sleep time - start with almost no sleep for maximum FPS
                if len(frame_times) >= 5:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    
                    # Ultra-aggressive optimization for maximum FPS
                    if self.fps < 100:  # If FPS is too low, reduce sleep time aggressively
                        self.sleep_time = max(0.0, self.sleep_time * 0.5)  # Cut sleep time in half
                    elif avg_frame_time < 0.0005:  # If processing is very fast (< 0.5ms)
                        self.sleep_time = max(0.0, self.sleep_time * 0.9)  # Reduce sleep time gradually
                    else:
                        # Only add minimal sleep to prevent CPU overload
                        self.sleep_time = min(0.0001, self.sleep_time * 1.1)  # Very small maximum sleep
                
                # Ultra-short or no sleep for maximum performance
                # Skip sleep entirely if FPS is below target
                if self.fps < 120:  # Always skip sleep if below 120 FPS
                    continue
                elif self.sleep_time > 0:
                    time.sleep(self.sleep_time)

        def fps_loop(self):
            """Calculate FPS in a separate thread."""
            last_count = 0
            # For tracking frame times
            self.frame_times = collections.deque(maxlen=30)
            
            while self.running:
                time.sleep(1.0)  # Update once per second
                with self.lock:
                    current_count = self.frame_count
                    
                frames = current_count - last_count
                self.fps = frames
                
                # Calculate average frame time if we have data
                if hasattr(self, 'frame_times') and len(self.frame_times) > 0:
                    avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                    print(f"Capture FPS: {self.fps}, Sleep: {self.sleep_time*1000:.3f}ms, Frame time: {avg_frame_time*1000:.3f}ms")
                else:
                    print(f"Capture FPS: {self.fps}, Sleep: {self.sleep_time*1000:.3f}ms")
                
                last_count = current_count

        def stop_capture(self):
            """Stop the capture thread and clean up resources."""
            self.running = False
            
            if self.capture_thread:
                self.capture_thread.join(timeout=1.0)
                
            if self.fps_thread:
                self.fps_thread.join(timeout=1.0)
                
            # Clean up GDI resources
            self._clean_gdi_resources()
                
            print("Screen capture stopped and resources cleaned up")

        def get_frame(self):
            """Get the latest captured frame."""
            with self.lock:
                if self.current_frame is None:
                    return None
                return self.current_frame.copy()  # Return a copy for thread safety
    
    def get_current_frame(self):
        """Get current frame from screen capture thread."""
        # Initialize screen capture if not already done
        if not hasattr(self, 'screen_capture'):
            self.screen_capture = self.ScreenCapture()
            # Wait a moment for first frame to be captured
            import time
            time.sleep(0.1)
        
        # Get the latest frame from the capture thread
        return self.screen_capture.get_frame()

class OverlayWindow:
    def __init__(self):
        """Initialize the overlay window."""
        try:
            logging.info("Initializing OverlayWindow")
            
            # Initialize window properties
            self.width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            self.height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            logging.info(f"Screen dimensions: {self.width}x{self.height}")
            
            self.running = True
            self.hwnd = None
            
            # Initialize config
            self.config = config.copy()  # Create a copy of the global config
            
            # Initialize menu properties
            self.menu_active = False
            self.active_tab = 'Home'
            self.tabs = ['Home', 'Aim-Assist', 'Visuals', 'Trigger Bot', 'Keybinds', 'Settings']
            self.menu_items = []
            self.mouse_down = False
            self.active_slider = None
            self.waiting_for_key = None
            self.active_input_field = None
            self.current_input_text = ""
            self.last_mouse_pos = (0, 0)
            self.status_message = ""
            self.status_message_time = 0
            self.menu_rect = FULL_MENU_RECT
            self.dragging = False
            self.drag_offset = (0, 0)
            self.show_config_dropdown = False
            self.config_files = []
            self.dropdown_selected = -1
            self.menu_page = 0
            self.items_per_page = 5
            self.config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
            if not os.path.exists(self.config_dir):
                os.makedirs(self.config_dir)
            
            # Initialize detector with config
            self.detector = YOLODetector(config=self.config)
            logging.info("YOLO detector initialized")
            
            # Register window class
            wc = win32gui.WNDCLASS()
            wc.lpfnWndProc = self.wnd_proc
            wc.lpszClassName = "OverlayWindow"
            wc.hbrBackground = win32gui.GetStockObject(win32con.NULL_BRUSH)
            win32gui.RegisterClass(wc)
            logging.info("Window class registered")
            
            # Create window with extended styles for transparency and click-through
            ex_style = (win32con.WS_EX_TOPMOST | 
                       win32con.WS_EX_LAYERED |
                       win32con.WS_EX_TOOLWINDOW)
            
            # Create window with minimal styles
            self.hwnd = win32gui.CreateWindowEx(
                ex_style,
                wc.lpszClassName,
                "Overlay",
                win32con.WS_POPUP,
                0, 0, self.width, self.height,
                0, 0, 0, None
            )
            logging.info("Window created")
            
            # Set window properties for transparency
            win32gui.SetLayeredWindowAttributes(
                self.hwnd,
                0,  # Color key (0 = black for transparent)
                255,  # Alpha (255 for fully opaque)
                win32con.LWA_COLORKEY  # Use color key transparency
            )
            logging.info("Window transparency set")
            
            # Make window click-through initially
            self.set_click_through(True)
            
            # Set window to be always on top
            win32gui.SetWindowPos(
                self.hwnd,
                win32con.HWND_TOPMOST,
                0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW
            )
            logging.info("Window made click-through")
            
            # Show window
            win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
            logging.info("Window shown")
            
            # Build menu items
            self.menu_items = self.build_menu_items()
            logging.info("Menu items built")
            
            # Register hotkeys
            self.register_hotkeys()
            logging.info("Hotkeys registered")
            
            # Start aim assist thread
            self.aim_thread = threading.Thread(target=self.aim_assist_loop, daemon=True)
            self.aim_thread.start()
            
            # Remove frame skipping for box drawing
            if hasattr(self, 'draw_frame_counter'):
                del self.draw_frame_counter
            if hasattr(self, 'draw_every_n_frames'):
                del self.draw_every_n_frames
            
        except Exception as e:
            logging.error(f"Error in OverlayWindow initialization: {e}")
            logging.error(traceback.format_exc())
            raise

    def aim_assist_loop(self):
        last_target = None
        while self.running:
            # Always use latest current_boxes, regardless of aim assist state
            boxes = self.detector.current_boxes
            if self.detector.aiming_enabled and boxes and len(boxes) > 0:
                screen_cx = self.width // 2
                screen_cy = self.height // 2
                min_dist = float('inf')
                best_box = None
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    cy = y1 + 0.25 * (y2 - y1)
                    dist = math.hypot(cx - screen_cx, cy - screen_cy)
                    if dist < min_dist:
                        min_dist = dist
                        best_box = (cx, cy)
                if best_box is not None:
                    dx = best_box[0] - screen_cx
                    dy = best_box[1] - screen_cy
                    smoothing = max(0.01, min(self.config.get('smoothing_xy', 0.1), 0.2))
                    move_x = int(dx * smoothing)
                    move_y = int(dy * smoothing)
                    if abs(move_x) > 1 or abs(move_y) > 1:
                        move_mouse(move_x, move_y)
                    last_target = best_box
            else:
                if last_target is not None:
                    print("[DEBUG] Aim assist state reset (no boxes or disabled)")
                last_target = None
            time.sleep(0.01)

    def build_menu_items(self):
        """Build the menu items list based on active tab."""
        # Base items that appear in all tabs
        base_items = []
        
        # Tab-specific items
        tab_items = {
            'Home': [
                ('GPU: ' + str(self.config.get('gpu_info', 'Unknown GPU')), 'text', None, None, None, None),
                ('Show FOV Visualiser', 'toggle', 'show_fov_visualiser', None, None, None),
                ('Show Player Boxes', 'toggle', 'show_player_boxes', None, None, None),
                ('Detection Confidence', 'slider', 'confidence', 0.0, 1.0, None),
                ('Save Config', 'button', 'save_config', None, None, None),
                ('Load Config', 'button', 'load_config', None, None, None),
            ],
            'Aim-Assist': [
                ('Tracking Speed', 'slider', 'tracking_speed', 0.0, 1.0, None),
                ('Humaniser', 'slider', 'humaniser', 0.0, 1.0, None),
                ('FOV Modifier', 'slider', 'fov_modifier', 50, 500, None),
                ('Anti-lag', 'slider', 'anti_lag_value', 0.0, 10.0, None),
                ('Custom Bone Position', 'slider', 'custom_bone_position', -1.0, 1.0, None),
                ('Smoothing X/Y', 'slider', 'smoothing_xy', 0.0, 1.0, None),
                ('Enable Aim Assist', 'toggle', 'aim_assist_enabled', None, None, None),
            ],
            'Visuals': [
                ('Show FPS Counter', 'toggle', 'show_fps_counter', None, None, None),
                ('FPS Display Style', 'toggle', 'fps_display_style', None, None, None),
                ('FOV Circle Color', 'color_box', 'fov_color', None, None, None),
                ('Box Color', 'color_box', 'box_color', None, None, None),
                ('Show FOV Circle', 'toggle', 'show_fov_visualiser', None, None, None),
            ],
            'Trigger Bot': [
                ('Enable Trigger Bot', 'toggle', 'trigger_bot_enabled', None, None, None),
                ('Trigger Delay', 'slider', 'trigger_delay', 0.0, 1.0, None),
                ('Trigger Random', 'slider', 'trigger_random', 0.0, 1.0, None),
                ('Trigger Hold', 'slider', 'trigger_hold', 0.0, 1.0, None),
                ('Trigger Release', 'slider', 'trigger_release', 0.0, 1.0, None),
            ],
            'Keybinds': [
                ('Aim Key', 'key', 'tracking_key', None, None, None),
                ('Menu Key', 'key', 'menu_key', None, None, None),
                ('Trigger Key', 'key', 'trigger_key', None, None, None),
                ('FOV Toggle Key', 'key', 'fov_toggle_key', None, None, None),
                ('FPS Toggle Key', 'key', 'fps_toggle_key', None, None, None),
            ],
            'Settings': [
                ('Optimizations', 'toggle', 'optimizations_enabled', None, None, None),
                ('Use GPU Capture (DirectX)', 'toggle', 'use_directx_capture', None, None, None),
                ('Reset Settings', 'button', 'reset_settings', None, None, None),
            ]
        }
        
        # Add Optimization settings if enabled
        if self.config['optimizations_enabled']:
            tab_items['Settings'].extend([
                ('GPU Usage Threshold', 'number_input', 'gpu_utilization_threshold', 0, 100, None),
                ('Optimized FPS', 'number_input', 'optimized_fps_target', 30, 250, None),
            ])
        
        # Add default target FPS slider
        tab_items['Settings'].append(('Target FPS', 'number_input', 'default_target_fps', 30, 250, None))
        
        # Get items for current tab
        current_items = tab_items.get(self.active_tab, [])
        
        # Update menu items
        self.menu_items = base_items + current_items
        return self.menu_items

    def handle_menu_mouse(self, mx, my, down=False):
        """Handle mouse interactions with menu items."""
        if not self.config['show_menu']:
            return
            
        # Convert screen coordinates to menu-relative coordinates
        menu_x = mx - self.menu_rect[0]
        menu_y = my - self.menu_rect[1]
        
        # Check close button
        close_button_rect = (
            self.menu_rect[2] - 35,
            self.menu_rect[1] + 5,
            self.menu_rect[2] - 10,
            self.menu_rect[1] + TOP_BAR_HEIGHT - 5
        )
        if point_in_rect(mx, my, close_button_rect) and down:
            self.running = False
            win32gui.PostQuitMessage(0)
            return

        # Check sidebar tabs
        tab_start_y = self.menu_rect[1] + TOP_BAR_HEIGHT + 5
        for i, tab in enumerate(self.tabs):
            tab_rect = (
                self.menu_rect[0] + 5,
                tab_start_y + (i * (TAB_HEIGHT + 2)),
                self.menu_rect[0] + SIDEBAR_WIDTH - 5,
                tab_start_y + ((i + 1) * (TAB_HEIGHT + 2)) - 2
            )
            if point_in_rect(mx, my, tab_rect) and down:
                self.active_tab = tab
                self.menu_items = self.build_menu_items()
                self.active_slider = None
                win32gui.InvalidateRect(self.hwnd, None, True)
                win32gui.UpdateWindow(self.hwnd)
                return

        # Handle menu items based on active tab
        content_left = self.menu_rect[0] + SIDEBAR_WIDTH + 20
        y_offset = self.menu_rect[1] + TOP_BAR_HEIGHT + 20
        
        for item in self.menu_items:
            item_rect = (
                content_left,
                y_offset,
                content_left + FIXED_MENU_WIDTH - SIDEBAR_WIDTH - 40,
                y_offset + BUTTON_HEIGHT
            )
            
            if point_in_rect(mx, my, item_rect):
                if item[1] == 'toggle':
                    if down:
                        current_value = self.config.get(item[2], False)
                        self.config[item[2]] = not current_value
                        self.status_message = f"{item[0]} {'enabled' if not current_value else 'disabled'}"
                        self.status_message_time = time.time()
                        # Special handling for certain toggles
                        if item[2] == 'optimizations_enabled':
                            self.menu_items = self.build_menu_items()
                        elif item[2] == 'trigger_bot_enabled':
                            self.detector.trigger_enabled = not current_value
                        elif item[2] == 'aim_assist_enabled':
                            self.detector.aiming_enabled = not current_value
                        win32gui.InvalidateRect(self.hwnd, None, True)
                        
                elif item[1] == 'slider':
                    slider_x = content_left + 200
                    slider_y = y_offset + (BUTTON_HEIGHT - SLIDER_HEIGHT) // 2
                    slider_rect = (slider_x, slider_y, slider_x + SLIDER_WIDTH, slider_y + SLIDER_HEIGHT)
                    
                    if point_in_rect(mx, my, (slider_x - 10, slider_y - 5, slider_x + SLIDER_WIDTH + 10, slider_y + SLIDER_HEIGHT + 5)):
                        if down:
                            self.active_slider = item[2]
                            self.handle_slider_drag(mx, my)
                        elif self.active_slider == item[2]:
                            self.handle_slider_drag(mx, my)
                        
                elif item[1] == 'number_input':
                    if down:
                        input_x = content_left + 200
                        input_rect = (input_x, y_offset, input_x + 80, y_offset + BUTTON_HEIGHT)
                        
                        if point_in_rect(mx, my, input_rect):
                            self.active_input_field = item[2]
                            self.current_input_text = str(self.config.get(item[2], ""))
                            self.active_slider = None
                            win32gui.InvalidateRect(self.hwnd, None, True)
                        
                elif item[1] == 'key':
                    if down:
                        key_x = content_left + 200
                        key_rect = (key_x, y_offset, key_x + 80, y_offset + BUTTON_HEIGHT)
                        
                        if point_in_rect(mx, my, key_rect):
                            self.waiting_for_key = item[2]
                            self.status_message = f"Press a key for {item[0]}..."
                            self.status_message_time = time.time()
                            win32gui.InvalidateRect(self.hwnd, None, True)
                        
                elif item[1] == 'color_box':
                    if down:
                        color_x = content_left + 200
                        color_rect = (color_x, y_offset, color_x + 40, y_offset + BUTTON_HEIGHT)
                        
                        if point_in_rect(mx, my, color_rect):
                            color = self.show_color_picker(self.config[item[2]])
                            if color:
                                self.config[item[2]] = color
                                self.status_message = f"{item[0]} color updated"
                                self.status_message_time = time.time()
                                win32gui.InvalidateRect(self.hwnd, None, True)
                            
                elif item[1] == 'button':
                    if down:
                        button_x = content_left + 200
                        button_rect = (button_x, y_offset, button_x + 100, y_offset + BUTTON_HEIGHT)
                        
                        if point_in_rect(mx, my, button_rect):
                            if item[2] == 'save_config':
                                self.save_config()
                            elif item[2] == 'load_config':
                                self.load_config()
                            elif item[2] == 'reset_settings':
                                self.reset_settings()
                            elif item[2] == 'exit':
                                self.running = False
                
            y_offset += BUTTON_HEIGHT + 5

    def draw_menu(self, hdc):
        INFO_ICON_SIZE = 18
        INFO_ICON_OFFSET = 6
        LABEL_WIDTH = 170
        """Draw the menu interface."""
        try:
            # Create font for menu using win32gui.LOGFONT with better quality
            logfont = win32gui.LOGFONT()
            logfont.lfHeight = -TEXT_SIZE  # Negative for better quality
            logfont.lfWeight = win32con.FW_NORMAL
            logfont.lfFaceName = 'Segoe UI'  # Better font
            logfont.lfQuality = win32con.ANTIALIASED_QUALITY  # Anti-aliasing
            font = win32gui.CreateFontIndirect(logfont)
            old_font = win32gui.SelectObject(hdc, font)
            
            # Enable better text rendering
            win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
            
            # Calculate menu position
            self.menu_rect = (
                MENU_START_X,
                MENU_START_Y,
                MENU_START_X + FIXED_MENU_WIDTH,
                MENU_START_Y + FIXED_MENU_HEIGHT
            )
            left, top, right, bottom = self.menu_rect
            menu_width = right - left
            menu_height = bottom - top

            # --- Main Menu Background with rounded corners effect ---
            main_bg_brush = win32gui.CreateSolidBrush(rgb_to_colorref((20, 20, 20)))
            win32gui.FillRect(hdc, (left, top, right, bottom), main_bg_brush)
            win32gui.DeleteObject(main_bg_brush)

            # --- Main Border (Subtle) ---
            main_border_pen = win32gui.CreatePen(win32con.PS_SOLID, 1, rgb_to_colorref((40, 40, 40)))
            old_pen = win32gui.SelectObject(hdc, main_border_pen)
            win32gui.SelectObject(hdc, win32gui.GetStockObject(win32con.NULL_BRUSH))
            win32gui.Rectangle(hdc, left, top, right, bottom)
            win32gui.SelectObject(hdc, old_pen)
            win32gui.DeleteObject(main_border_pen)

            # --- Top Bar with gradient effect ---
            top_bar_rect = (left, top, right, top + TOP_BAR_HEIGHT)
            top_bar_brush = win32gui.CreateSolidBrush(rgb_to_colorref((30, 30, 30)))
            win32gui.FillRect(hdc, top_bar_rect, top_bar_brush)
            win32gui.DeleteObject(top_bar_brush)
            
            # Top bar bottom border
            border_pen = win32gui.CreatePen(win32con.PS_SOLID, 1, rgb_to_colorref((50, 50, 50)))
            old_pen = win32gui.SelectObject(hdc, border_pen)
            win32gui.MoveToEx(hdc, left, top + TOP_BAR_HEIGHT)
            win32gui.LineTo(hdc, right, top + TOP_BAR_HEIGHT)
            win32gui.SelectObject(hdc, old_pen)
            win32gui.DeleteObject(border_pen)

            # Draw title with better font
            title_font = win32gui.LOGFONT()
            title_font.lfHeight = -16
            title_font.lfWeight = win32con.FW_BOLD
            title_font.lfFaceName = 'Segoe UI'
            title_font.lfQuality = win32con.ANTIALIASED_QUALITY
            title_hfont = win32gui.CreateFontIndirect(title_font)
            old_title_font = win32gui.SelectObject(hdc, title_hfont)
            
            win32gui.SetTextColor(hdc, rgb_to_colorref(self.config['menu_highlight_color']))
            title_rect = (left + 15, top + 5, left + 200, top + TOP_BAR_HEIGHT - 5)
            win32gui.DrawText(hdc, "FuryAI", -1, title_rect, win32con.DT_LEFT | win32con.DT_VCENTER | win32con.DT_SINGLELINE)
            
            win32gui.SelectObject(hdc, old_title_font)
            win32gui.DeleteObject(title_hfont)

            # Draw close button with hover effect
            close_button_rect = (right - 35, top + 5, right - 10, top + TOP_BAR_HEIGHT - 5)
            close_brush = win32gui.CreateSolidBrush(rgb_to_colorref((60, 60, 60)))
            win32gui.FillRect(hdc, close_button_rect, close_brush)
            win32gui.DeleteObject(close_brush)
            
            # Close button X
            close_font = win32gui.LOGFONT()
            close_font.lfHeight = -14
            close_font.lfWeight = win32con.FW_NORMAL
            close_font.lfFaceName = 'Segoe UI'
            close_font.lfQuality = win32con.ANTIALIASED_QUALITY
            close_hfont = win32gui.CreateFontIndirect(close_font)
            old_close_font = win32gui.SelectObject(hdc, close_hfont)
            
            win32gui.SetTextColor(hdc, rgb_to_colorref((200, 200, 200)))
            win32gui.DrawText(hdc, "✕", -1, close_button_rect, win32con.DT_CENTER | win32con.DT_VCENTER | win32con.DT_SINGLELINE)
            
            win32gui.SelectObject(hdc, old_close_font)
            win32gui.DeleteObject(close_hfont)

            # --- Sidebar with better styling ---
            sidebar_rect = (left, top + TOP_BAR_HEIGHT, left + SIDEBAR_WIDTH, bottom)
            sidebar_brush = win32gui.CreateSolidBrush(rgb_to_colorref((25, 25, 25)))
            win32gui.FillRect(hdc, sidebar_rect, sidebar_brush)
            win32gui.DeleteObject(sidebar_brush)
            
            # Sidebar right border
            sidebar_border_pen = win32gui.CreatePen(win32con.PS_SOLID, 1, rgb_to_colorref((40, 40, 40)))
            old_pen = win32gui.SelectObject(hdc, sidebar_border_pen)
            win32gui.MoveToEx(hdc, left + SIDEBAR_WIDTH, top + TOP_BAR_HEIGHT)
            win32gui.LineTo(hdc, left + SIDEBAR_WIDTH, bottom)
            win32gui.SelectObject(hdc, old_pen)
            win32gui.DeleteObject(sidebar_border_pen)

            # Draw tabs with better styling
            tab_start_y = top + TOP_BAR_HEIGHT + 5
            for i, tab_name in enumerate(self.tabs):
                tab_rect = (
                    left + 5,
                    tab_start_y + (i * (TAB_HEIGHT + 2)),
                    left + SIDEBAR_WIDTH - 5,
                    tab_start_y + ((i + 1) * (TAB_HEIGHT + 2)) - 2
                )
                
                # Draw tab background with rounded corners effect
                if tab_name == self.active_tab:
                    # Active tab background
                    tab_brush = win32gui.CreateSolidBrush(rgb_to_colorref((35, 35, 35)))
                    win32gui.FillRect(hdc, tab_rect, tab_brush)
                    win32gui.DeleteObject(tab_brush)
                    
                    # Active tab highlight
                    highlight_rect = (tab_rect[0], tab_rect[1], tab_rect[0] + 3, tab_rect[3])
                    highlight_brush = win32gui.CreateSolidBrush(rgb_to_colorref(self.config['menu_highlight_color']))
                    win32gui.FillRect(hdc, highlight_rect, highlight_brush)
                    win32gui.DeleteObject(highlight_brush)
                
                # Draw tab text with proper font
                text_color = self.config['menu_highlight_color'] if tab_name == self.active_tab else (150, 150, 150)
                win32gui.SetTextColor(hdc, rgb_to_colorref(text_color))
                
                # Center text in tab
                text_rect = (
                    tab_rect[0] + 10,
                    tab_rect[1],
                    tab_rect[2],
                    tab_rect[3]
                )
                win32gui.DrawText(hdc, tab_name, -1, text_rect, win32con.DT_LEFT | win32con.DT_VCENTER | win32con.DT_SINGLELINE)

            # --- Content Area ---
            content_left = left + SIDEBAR_WIDTH + 20
            content_top = top + TOP_BAR_HEIGHT
            content_right = right - 20
            content_bottom = bottom - BOTTOM_BAR_HEIGHT

            # Draw menu items with better spacing and styling
            y_offset = content_top + 20
            for item in self.menu_items:
                if y_offset + BUTTON_HEIGHT > content_bottom:
                    break

                # Only draw info icon if not excluded
                show_info_icon = True
                # Exclude by key or by tab (Keybinds)
                if (item[2] in INFO_ICON_EXCLUDE) or (self.active_tab == 'Keybinds') or (item[1] == 'button' and item[2] == 'reset_settings') or (item[1] == 'text'):
                    show_info_icon = False

                if show_info_icon:
                    # Draw info icon using Unicode symbol ℹ️
                    info_x = content_left
                    info_y = y_offset + (BUTTON_HEIGHT - INFO_ICON_SIZE) // 2
                    info_rect = (info_x, info_y, info_x + INFO_ICON_SIZE, info_y + INFO_ICON_SIZE)
                    # Draw the info symbol (ℹ️) with a larger, anti-aliased font
                    info_font = win32gui.LOGFONT()
                    info_font.lfHeight = -INFO_ICON_SIZE
                    info_font.lfWeight = win32con.FW_BOLD
                    info_font.lfFaceName = 'Segoe UI Symbol'
                    info_font.lfQuality = win32con.ANTIALIASED_QUALITY
                    hfont = win32gui.CreateFontIndirect(info_font)
                    old_font = win32gui.SelectObject(hdc, hfont)
                    win32gui.SetTextColor(hdc, rgb_to_colorref((100, 200, 255)))
                    win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
                    win32gui.DrawText(hdc, 'ℹ️', -1, info_rect, win32con.DT_CENTER | win32con.DT_VCENTER | win32con.DT_SINGLELINE)
                    win32gui.SelectObject(hdc, old_font)
                    win32gui.DeleteObject(hfont)
                    label_left = content_left + INFO_ICON_SIZE + INFO_ICON_OFFSET

                    # Tooltip logic (expand hitbox for easier hover)
                    if hasattr(self, 'last_mouse_pos'):
                        rel_mouse_x, rel_mouse_y = self.last_mouse_pos
                        expanded_info_rect = (info_rect[0] - 4, info_rect[1] - 4, info_rect[2] + 4, info_rect[3] + 4)
                        if (expanded_info_rect[0] <= rel_mouse_x <= expanded_info_rect[2]) and (expanded_info_rect[1] <= rel_mouse_y <= expanded_info_rect[3]):
                            desc = INFO_DESCRIPTIONS.get(item[2], None)
                            if not desc:
                                desc = 'No description available.'
                            self.draw_tooltip(hdc, rel_mouse_x + 20, rel_mouse_y, desc)
                else:
                    label_left = content_left

                # Draw label with more space
                win32gui.SetTextColor(hdc, rgb_to_colorref((200, 200, 200)))
                label_rect = (label_left, y_offset, content_left + LABEL_WIDTH, y_offset + BUTTON_HEIGHT)
                win32gui.DrawText(hdc, item[0], -1, label_rect, win32con.DT_LEFT | win32con.DT_VCENTER | win32con.DT_SINGLELINE)

                # Controls alignment
                control_x = content_left + LABEL_WIDTH + 10

                if item[1] == 'toggle':
                    toggle_x = control_x
                    toggle_y = y_offset + (BUTTON_HEIGHT - TOGGLE_SIZE) // 2
                    toggle_rect = (toggle_x, toggle_y, toggle_x + TOGGLE_SIZE * 2, toggle_y + TOGGLE_SIZE)
                    
                    # Toggle background
                    is_on = self.config.get(item[2], False)
                    toggle_color = self.config['button_active_color'] if is_on else (60, 60, 60)
                    toggle_brush = win32gui.CreateSolidBrush(rgb_to_colorref(toggle_color))
                    win32gui.FillRect(hdc, toggle_rect, toggle_brush)
                    win32gui.DeleteObject(toggle_brush)
                    
                    # Toggle circle
                    circle_x = toggle_x + (TOGGLE_SIZE if is_on else 2)
                    circle_rect = (circle_x, toggle_y + 2, circle_x + TOGGLE_SIZE - 4, toggle_y + TOGGLE_SIZE - 2)
                    circle_brush = win32gui.CreateSolidBrush(rgb_to_colorref((255, 255, 255)))
                    win32gui.FillRect(hdc, circle_rect, circle_brush)
                    win32gui.DeleteObject(circle_brush)

                elif item[1] == 'slider':
                    slider_x = control_x
                    slider_y = y_offset + (BUTTON_HEIGHT - SLIDER_HEIGHT) // 2
                    slider_rect = (slider_x, slider_y, slider_x + SLIDER_WIDTH, slider_y + SLIDER_HEIGHT)
                    
                    # Slider track
                    track_brush = win32gui.CreateSolidBrush(rgb_to_colorref((50, 50, 50)))
                    win32gui.FillRect(hdc, slider_rect, track_brush)
                    win32gui.DeleteObject(track_brush)
                    
                    # Slider fill
                    value = self.config.get(item[2], 0.0)
                    min_val, max_val = item[3], item[4]
                    fill_width = int(((value - min_val) / (max_val - min_val)) * SLIDER_WIDTH)
                    if fill_width > 0:
                        fill_rect = (slider_x, slider_y, slider_x + fill_width, slider_y + SLIDER_HEIGHT)
                        fill_brush = win32gui.CreateSolidBrush(rgb_to_colorref(self.config['slider_active_color']))
                        win32gui.FillRect(hdc, fill_rect, fill_brush)
                        win32gui.DeleteObject(fill_brush)
                    
                    # Slider handle
                    handle_x = slider_x + fill_width - 4
                    handle_rect = (handle_x, slider_y - 2, handle_x + 8, slider_y + SLIDER_HEIGHT + 2)
                    handle_brush = win32gui.CreateSolidBrush(rgb_to_colorref((255, 255, 255)))
                    win32gui.FillRect(hdc, handle_rect, handle_brush)
                    win32gui.DeleteObject(handle_brush)
                    
                    # Value text
                    value_text = f"{value:.2f}" if isinstance(value, float) else str(value)
                    win32gui.SetTextColor(hdc, rgb_to_colorref((150, 150, 150)))
                    value_rect = (slider_x + SLIDER_WIDTH + 10, y_offset, slider_x + SLIDER_WIDTH + 60, y_offset + BUTTON_HEIGHT)
                    win32gui.DrawText(hdc, value_text, -1, value_rect, win32con.DT_LEFT | win32con.DT_VCENTER | win32con.DT_SINGLELINE)

                elif item[1] == 'number_input':
                    # Draw number input field
                    input_x = control_x
                    input_rect = (input_x, y_offset, input_x + 80, y_offset + BUTTON_HEIGHT)
                    
                    # Input background
                    input_brush = win32gui.CreateSolidBrush(rgb_to_colorref((40, 40, 40)))
                    win32gui.FillRect(hdc, input_rect, input_brush)
                    win32gui.DeleteObject(input_brush)
                    
                    # Input border
                    border_pen = win32gui.CreatePen(win32con.PS_SOLID, 1, rgb_to_colorref((60, 60, 60)))
                    old_pen = win32gui.SelectObject(hdc, border_pen)
                    win32gui.SelectObject(hdc, win32gui.GetStockObject(win32con.NULL_BRUSH))
                    win32gui.Rectangle(hdc, input_rect[0], input_rect[1], input_rect[2], input_rect[3])
                    win32gui.SelectObject(hdc, old_pen)
                    win32gui.DeleteObject(border_pen)
                    
                    # Draw input text
                    value = str(self.config.get(item[2], ""))
                    if self.active_input_field == item[2]:
                        value = self.current_input_text
                    win32gui.SetTextColor(hdc, rgb_to_colorref((200, 200, 200)))
                    text_rect = (input_rect[0] + 5, input_rect[1], input_rect[2] - 5, input_rect[3])
                    win32gui.DrawText(hdc, value, -1, text_rect, win32con.DT_LEFT | win32con.DT_VCENTER | win32con.DT_SINGLELINE)

                elif item[1] == 'key':
                    # Draw modern key binding box
                    key_x = control_x
                    key_rect = (key_x, y_offset, key_x + 80, y_offset + BUTTON_HEIGHT)
                    key_brush = win32gui.CreateSolidBrush(rgb_to_colorref((50, 50, 50)))
                    win32gui.FillRect(hdc, key_rect, key_brush)
                    win32gui.DeleteObject(key_brush)
                    
                    # Key border
                    key_border_pen = win32gui.CreatePen(win32con.PS_SOLID, 1, rgb_to_colorref((70, 70, 70)))
                    old_pen = win32gui.SelectObject(hdc, key_border_pen)
                    win32gui.SelectObject(hdc, win32gui.GetStockObject(win32con.NULL_BRUSH))
                    win32gui.Rectangle(hdc, key_rect[0], key_rect[1], key_rect[2], key_rect[3])
                    win32gui.SelectObject(hdc, old_pen)
                    win32gui.DeleteObject(key_border_pen)
                    
                    # Key text
                    win32gui.SetTextColor(hdc, rgb_to_colorref((200, 200, 200)))
                    win32gui.DrawText(hdc, self.config.get(item[2], ''), -1, key_rect, win32con.DT_CENTER | win32con.DT_VCENTER | win32con.DT_SINGLELINE)

                elif item[1] == 'color_box':
                    # Draw color box with border
                    color_x = control_x
                    color_rect = (color_x, y_offset, color_x + 40, y_offset + BUTTON_HEIGHT)
                    color_brush = win32gui.CreateSolidBrush(rgb_to_colorref(self.config.get(item[2], (0, 0, 0))))
                    win32gui.FillRect(hdc, color_rect, color_brush)
                    win32gui.DeleteObject(color_brush)
                    
                    # Color box border
                    color_border_pen = win32gui.CreatePen(win32con.PS_SOLID, 1, rgb_to_colorref((70, 70, 70)))
                    old_pen = win32gui.SelectObject(hdc, color_border_pen)
                    win32gui.SelectObject(hdc, win32gui.GetStockObject(win32con.NULL_BRUSH))
                    win32gui.Rectangle(hdc, color_rect[0], color_rect[1], color_rect[2], color_rect[3])
                    win32gui.SelectObject(hdc, old_pen)
                    win32gui.DeleteObject(color_border_pen)

                elif item[1] == 'button':
                    # Draw modern button
                    button_x = control_x
                    button_rect = (button_x, y_offset, button_x + 100, y_offset + BUTTON_HEIGHT)
                    button_brush = win32gui.CreateSolidBrush(rgb_to_colorref((50, 50, 50)))
                    win32gui.FillRect(hdc, button_rect, button_brush)
                    win32gui.DeleteObject(button_brush)
                    
                    # Button border
                    button_border_pen = win32gui.CreatePen(win32con.PS_SOLID, 1, rgb_to_colorref((70, 70, 70)))
                    old_pen = win32gui.SelectObject(hdc, button_border_pen)
                    win32gui.SelectObject(hdc, win32gui.GetStockObject(win32con.NULL_BRUSH))
                    win32gui.Rectangle(hdc, button_rect[0], button_rect[1], button_rect[2], button_rect[3])
                    win32gui.SelectObject(hdc, old_pen)
                    win32gui.DeleteObject(button_border_pen)
                    
                    # Button text
                    win32gui.SetTextColor(hdc, rgb_to_colorref((200, 200, 200)))
                    win32gui.DrawText(hdc, item[0], -1, button_rect, win32con.DT_CENTER | win32con.DT_VCENTER | win32con.DT_SINGLELINE)

                y_offset += BUTTON_HEIGHT + BUTTON_PADDING

            # Clean up
            win32gui.SelectObject(hdc, old_font)
            win32gui.DeleteObject(font)
            
        except Exception as e:
            logging.error(f"Error drawing menu: {e}")
            traceback.print_exc()

    def draw_text(self, hdc, x, y, text, color, size=TEXT_SIZE):
        """Draw text on the overlay."""
        try:
            # Create font using win32gui.LOGFONT with anti-aliasing
            logfont = win32gui.LOGFONT()
            logfont.lfHeight = -size  # Negative for better quality
            logfont.lfWeight = win32con.FW_NORMAL
            logfont.lfFaceName = 'Segoe UI'
            logfont.lfQuality = win32con.ANTIALIASED_QUALITY
            font = win32gui.CreateFontIndirect(logfont)
            old_font = win32gui.SelectObject(hdc, font)
            
            # Set text color
            if isinstance(color, tuple):
                color = rgb_to_colorref(color)
            win32gui.SetTextColor(hdc, color)
            
            # Enable better text rendering
            win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
            
            # Draw text using DrawText
            text_rect = (x, y, x + 1000, y + size + 5)  # Wide enough for any text
            win32gui.DrawText(hdc, text, -1, text_rect, win32con.DT_LEFT | win32con.DT_VCENTER | win32con.DT_SINGLELINE)
            
            # Clean up
            win32gui.SelectObject(hdc, old_font)
            win32gui.DeleteObject(font)
            
        except Exception as e:
            logging.error(f"Error in draw_text: {e}")
            traceback.print_exc()

    def draw_text_centered(self, hdc, rect, text, color, size=TEXT_SIZE):
        """Draw text centered within a rectangle."""
        try:
            # Calculate text size
            text_size = win32gui.GetTextExtentPoint32(hdc, text)
            text_width = text_size[0]
            text_height = text_size[1]
            
            # Calculate position (centered within the rectangle)
            x = rect[0] + (rect[2] - rect[0] - text_width) // 2
            y = rect[1] + (rect[3] - rect[1] - text_height) // 2
            
            # Draw text
            self.draw_text(hdc, x, y, text, color, size)
        except Exception as e:
            logging.error(f"Error in draw_text_centered: {e}")
            traceback.print_exc()

    def set_click_through(self, enable):
        """Enable or disable click-through for the window."""
        try:
            ex_style = win32gui.GetWindowLong(self.hwnd, win32con.GWL_EXSTYLE)
            if enable:
                ex_style |= win32con.WS_EX_TRANSPARENT
            else:
                ex_style &= ~win32con.WS_EX_TRANSPARENT
            win32gui.SetWindowLong(self.hwnd, win32con.GWL_EXSTYLE, ex_style)
            
            # Force window update
            win32gui.SetWindowPos(
                self.hwnd,
                win32con.HWND_TOPMOST,
                0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_FRAMECHANGED
            )
        except Exception as e:
            logging.error(f"Error setting click-through: {e}")
            traceback.print_exc()

    def register_hotkeys(self):
        """Register all hotkeys based on current config values."""
        try:
            # Unregister any existing hotkeys first
            for i in range(1, 10):
                try:
                    user32.UnregisterHotKey(self.hwnd, i)
                except:
                    pass

            # Register menu key
            menu_vk = KEY_MAPPING.get(self.config.get('menu_key', 'INSERT').upper(), VK_INSERT)
            result = user32.RegisterHotKey(self.hwnd, 1, 0, menu_vk)
            # Register tracking key
            tracking_vk = KEY_MAPPING.get(self.config.get('tracking_key', 'F2').upper(), VK_F2)
            result = user32.RegisterHotKey(self.hwnd, 2, 0, tracking_vk)
            # Register trigger key
            trigger_vk = KEY_MAPPING.get(self.config.get('trigger_key', 'F3').upper(), VK_F3)
            result = user32.RegisterHotKey(self.hwnd, 3, 0, trigger_vk)
            # Register FOV toggle key
            fov_vk = KEY_MAPPING.get(self.config.get('fov_toggle_key', 'F4').upper(), VK_F4)
            result = user32.RegisterHotKey(self.hwnd, 4, 0, fov_vk)
            # Register FPS toggle key
            fps_vk = KEY_MAPPING.get(self.config.get('fps_toggle_key', 'F5').upper(), VK_F5)
            result = user32.RegisterHotKey(self.hwnd, 5, 0, fps_vk)
        except Exception as e:
            print(f"Error registering hotkeys: {e}")

    def wnd_proc(self, hwnd, msg, wparam, lparam):
        """Window procedure for handling window messages."""
        if msg == win32con.WM_DESTROY:
            self.running = False
            win32gui.PostQuitMessage(0)
            return 0
            
        elif msg == win32con.WM_PAINT:
            hdc, ps = win32gui.BeginPaint(hwnd)
            self.draw_overlay(hdc)
            win32gui.EndPaint(hwnd, ps)
            return 0
            
        elif msg == win32con.WM_LBUTTONDOWN:
            mx, my = win32gui.GetCursorPos()
            self.handle_menu_mouse(mx, my, True)
            return 0
            
        elif msg == win32con.WM_MOUSEMOVE:
            x = win32api.LOWORD(lparam)
            y = win32api.HIWORD(lparam)
            self.last_mouse_pos = (x, y)
            self.handle_menu_mouse(x, y, False)
            win32gui.InvalidateRect(hwnd, None, True)
            return 0
            
        elif msg == win32con.WM_LBUTTONUP:
            self.active_slider = None
            return 0
            
        elif msg == win32con.WM_KEYDOWN:
            if self.active_input_field:
                if wparam == win32con.VK_RETURN:  # Enter key
                    try:
                        value = int(self.current_input_text)
                        min_val = next(item[3] for item in self.menu_items if item[2] == self.active_input_field)
                        max_val = next(item[4] for item in self.menu_items if item[2] == self.active_input_field)
                        value = max(min_val, min(max_val, value))
                        self.config[self.active_input_field] = value
                        self.status_message = f"Value updated to {value}"
                        self.status_message_time = time.time()
                    except ValueError:
                        self.status_message = "Invalid number"
                        self.status_message_time = time.time()
                    self.active_input_field = None
                    self.current_input_text = ""
                    win32gui.InvalidateRect(hwnd, None, True)
                elif wparam == win32con.VK_ESCAPE:  # Escape key
                    self.active_input_field = None
                    self.current_input_text = ""
                    win32gui.InvalidateRect(hwnd, None, True)
                elif wparam == win32con.VK_BACK:  # Backspace
                    self.current_input_text = self.current_input_text[:-1]
                    win32gui.InvalidateRect(hwnd, None, True)
                elif wparam >= 0x30 and wparam <= 0x39:  # Number keys
                    self.current_input_text += chr(wparam)
                    win32gui.InvalidateRect(hwnd, None, True)
                return 0
            elif self.waiting_for_key:
                key_name = get_key_name(wparam)
                if key_name:
                    self.config[self.waiting_for_key] = key_name
                    self.waiting_for_key = None
                    self.status_message = f"Key bound to {key_name}"
                    self.status_message_time = time.time()
                    self.register_hotkeys()  # Re-register hotkeys with new key
                    win32gui.InvalidateRect(hwnd, None, True)
                return 0
                
        elif msg == win32con.WM_HOTKEY:
            if wparam == 1:  # Menu hotkey ID
                self.config['show_menu'] = not self.config['show_menu']
                self.set_click_through(not self.config['show_menu'])
                win32gui.InvalidateRect(hwnd, None, True)
                return 0
            elif wparam == 2:  # Tracking key
                self.config['aim_assist_enabled'] = not self.config['aim_assist_enabled']
                self.detector.aiming_enabled = not self.config['aim_assist_enabled']
                return 0
            elif wparam == 3:  # Trigger key
                self.config['trigger_bot_enabled'] = not self.config['trigger_bot_enabled']
                return 0
            elif wparam == 4:  # FOV toggle key
                self.config['show_fov_visualiser'] = not self.config['show_fov_visualiser']
                return 0
            elif wparam == 5:  # FPS toggle key
                self.config['show_fps_counter'] = not self.config['show_fps_counter']
                return 0
        
        # Default handling for any unhandled messages
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def draw_overlay(self, hdc):
        try:
            mem_dc = win32gui.CreateCompatibleDC(hdc)
            mem_bitmap = win32gui.CreateCompatibleBitmap(hdc, self.width, self.height)
            win32gui.SelectObject(mem_dc, mem_bitmap)
            win32gui.SetBkMode(mem_dc, win32con.TRANSPARENT)
            brush = win32gui.CreateSolidBrush(0)
            win32gui.FillRect(mem_dc, (0, 0, self.width, self.height), brush)
            win32gui.DeleteObject(brush)
            # Draw FOV circle if enabled
            if self.config['show_fov_visualiser']:
                try:
                    center_x = self.width // 2
                    center_y = self.height // 2
                    radius = self.config['fov_modifier'] // 2
                    pen = win32gui.CreatePen(win32con.PS_SOLID, 2, rgb_to_colorref(self.config['fov_color']))
                    old_pen = win32gui.SelectObject(mem_dc, pen)
                    old_brush = win32gui.SelectObject(mem_dc, win32gui.GetStockObject(win32con.NULL_BRUSH))
                    win32gui.Ellipse(mem_dc, center_x - radius, center_y - radius, center_x + radius, center_y + radius)
                    win32gui.SelectObject(mem_dc, old_brush)
                    win32gui.SelectObject(mem_dc, old_pen)
                    win32gui.DeleteObject(pen)
                except Exception as e:
                    logging.error(f"Error drawing FOV circle: {e}")
            # Draw detection boxes only if show_player_boxes is True
            if self.config.get('show_player_boxes', True) and self.detector and self.detector.current_boxes is not None and len(self.detector.current_boxes) > 0:
                boxes = self.detector.current_boxes
                confidences = self.detector.current_confidences
                class_ids = self.detector.current_class_ids
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box
                    color = self.config.get('box_color', (0, 255, 0))
                    win32gui.SelectObject(mem_dc, win32gui.CreatePen(win32con.PS_SOLID, 2, rgb_to_colorref(color)))
                    win32gui.SelectObject(mem_dc, win32gui.GetStockObject(win32con.NULL_BRUSH))
                    win32gui.Rectangle(mem_dc, x1, y1, x2, y2)
                    label = f"{self.detector.class_names[cls_id]}: {conf:.2f}"
                    self.draw_text(mem_dc, x1, y1 - 18, label, color, size=14)
            # Draw FPS counter if enabled
            if self.config['show_fps_counter']:
                try:
                    # Get FPS values from detector
                    if self.detector:
                        # Use the current_fps directly from the detector
                        fps = self.detector.current_fps if hasattr(self.detector, 'current_fps') else 0
                        
                        # Get capture FPS from screen capture if available
                        capture_fps = 0
                        if hasattr(self.detector, 'screen_capture') and hasattr(self.detector.screen_capture, 'fps'):
                            capture_fps = self.detector.screen_capture.fps
                        
                        # Get frame skipping info if available
                        skip_info = ""
                        if hasattr(self.detector, 'process_every_n'):
                            skip_info = f" | Skip: {self.detector.process_every_n}"
                        
                        # Format based on display style
                        if self.config.get('fps_display_style', 0) == 0:  # Detailed style
                            fps_text = f"FPS: {fps} | Capture: {capture_fps}{skip_info}"
                        else:  # Simple style
                            fps_text = f"{fps}/{capture_fps}{skip_info}"
                    else:
                        fps_text = "FPS: 0"
                    
                    self.draw_text(mem_dc, 10, 10, fps_text, (255, 255, 0), size=16)
                except Exception as e:
                    logging.error(f"Error drawing FPS counter: {e}")
            # Draw menu if enabled
            if self.config['show_menu']:
                try:
                    self.draw_menu(mem_dc)
                except Exception as e:
                    logging.error(f"Error drawing menu: {e}")
            # Draw status message if active
            if time.time() - self.status_message_time < 2.0:
                try:
                    self.draw_status_message(mem_dc)
                except Exception as e:
                    logging.error(f"Error drawing status message: {e}")
            win32gui.BitBlt(hdc, 0, 0, self.width, self.height, mem_dc, 0, 0, win32con.SRCCOPY)
            win32gui.DeleteObject(mem_bitmap)
            win32gui.DeleteDC(mem_dc)
        except Exception as e:
            logging.error(f"Error in draw_overlay: {e}")
            logging.error(traceback.format_exc())
            try:
                if 'mem_bitmap' in locals():
                    win32gui.DeleteObject(mem_bitmap)
                if 'mem_dc' in locals():
                    win32gui.DeleteDC(mem_dc)
            except:
                pass

    def draw_status_message(self, hdc):
        """Draw the status message."""
        try:
            if not self.status_message:
                return
                
            # Create font for status message using win32gui.LOGFONT
            logfont = win32gui.LOGFONT()
            logfont.lfHeight = self.config['status_font_size']
            logfont.lfWeight = win32con.FW_BOLD
            logfont.lfFaceName = 'Arial'
            font = win32gui.CreateFontIndirect(logfont)
            old_font = win32gui.SelectObject(hdc, font)
            
            # Set text color
            win32gui.SetTextColor(hdc, rgb_to_colorref(self.config['menu_text_color']))
            
            # Calculate text size
            text_size = win32gui.GetTextExtentPoint32(hdc, self.status_message)
            text_width = text_size[0]
            text_height = text_size[1]
            
            # Calculate position (centered at bottom)
            x = (self.width - text_width) // 2
            y = self.height - text_height - 20  # 20 pixels from bottom
            
            # Draw text with shadow for better visibility
            win32gui.SetTextColor(hdc, rgb_to_colorref((0, 0, 0)))  # Black shadow
            shadow_rect = (x + 1, y + 1, x + text_width + 1, y + text_height + 1)
            win32gui.DrawText(hdc, self.status_message, -1, shadow_rect, win32con.DT_LEFT | win32con.DT_VCENTER | win32con.DT_SINGLELINE)
            
            win32gui.SetTextColor(hdc, rgb_to_colorref((255, 255, 255)))  # White text
            text_rect = (x, y, x + text_width, y + text_height)
            win32gui.DrawText(hdc, self.status_message, -1, text_rect, win32con.DT_LEFT | win32con.DT_VCENTER | win32con.DT_SINGLELINE)
            
            # Clean up
            win32gui.SelectObject(hdc, old_font)
            win32gui.DeleteObject(font)
            
        except Exception as e:
            logging.error(f"Error in draw_status_message: {e}")
            traceback.print_exc()

    def run(self):
        """Main message loop."""
        try:
            # Register hotkeys before starting the message loop
            self.register_hotkeys()
            
            # Start detection thread
            if self.detector:
                self.detector.start_detection()
            
            # Main message loop
            while self.running:
                try:
                    msg = win32gui.GetMessage(self.hwnd, 0, 0)
                    if msg[0]:
                        win32gui.TranslateMessage(msg[1])
                        win32gui.DispatchMessage(msg[1])
                        # Force redraw every frame
                        win32gui.InvalidateRect(self.hwnd, None, True)
                        win32gui.UpdateWindow(self.hwnd)
                    else:
                        break
                except Exception as e:
                    logging.error(f"Error in message loop: {e}")
                    logging.error(traceback.format_exc())
                    break
        
        except Exception as e:
            print(f"Error in message loop: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            if self.detector:
                self.detector.stop_detection()
            # Unregister hotkeys
            for i in range(1, 10):
                try:
                    user32.UnregisterHotKey(self.hwnd, i)
                except:
                    pass

    def save_config(self):
        """Save current configuration to file."""
        try:
            config_file = os.path.join(self.config_dir, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.status_message = "Configuration saved successfully"
            self.status_message_time = time.time()
        except Exception as e:
            self.status_message = f"Error saving config: {str(e)}"
            self.status_message_time = time.time()

    def load_config(self):
        """Load configuration from file."""
        try:
            config_file = os.path.join(self.config_dir, 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                self.status_message = "Configuration loaded successfully"
                self.menu_items = self.build_menu_items()
            else:
                self.status_message = "No saved configuration found"
            self.status_message_time = time.time()
        except Exception as e:
            self.status_message = f"Error loading config: {str(e)}"
            self.status_message_time = time.time()

    def reset_settings(self):
        """Reset all settings to default values."""
        default_config = {
            'show_menu': False,
            'confidence': 0.5,
            'fov_modifier': 150,
            'fov_color': (0, 255, 0),
            'box_color': (255, 0, 0),
            'text_color': (255, 255, 255),  # White
            'show_player_boxes': True,
            'tracking_speed': 0.5,
            'humaniser': 0.2,
            'show_fov_visualiser': True,
            'tracking_key': 'F2',
            'trigger_bot_enabled': False,
            'trigger_delay': 0.1,
            'trigger_key': 'F3',
            'menu_key': 'INSERT',
            'fov_toggle_key': 'F4',  # Default FOV toggle key
            'fps_toggle_key': 'F5',  # Default FPS toggle key
            'is_mouse_key': False,  # Track if aim key is a mouse button
            'anti_lag_value': 5.0,
            'custom_bone_position': 0.0,
            'use_directx_capture': True,
            'smoothing_xy': 0.5,
            'optimizations_enabled': False,
            'gpu_utilization_threshold': 80,
            'optimized_fps_target': 60,
            'default_target_fps': 144,
            'fps_display_style': 0,  # 0 = show labels, 1 = just numbers
        }
        self.config.update(default_config)
        self.status_message = "Settings reset to default"
        self.status_message_time = time.time()
        self.menu_items = self.build_menu_items()
        self.active_input_field = None
        self.current_input_text = ""
        win32gui.InvalidateRect(self.hwnd, None, True)

    def show_color_picker(self, initial_color):
        """Show a color picker dialog."""
        try:
            root = Tk()
            root.withdraw()  # Hide the main window
            color = colorchooser.askcolor(initial_color, title="Choose Color")
            root.destroy()
            if color[0]:  # color[0] contains the RGB values
                return tuple(map(int, color[0]))
        except Exception as e:
            print(f"Error showing color picker: {e}")
            return None

    def handle_slider_drag(self, mx, my):
        """Handle slider dragging."""
        if not self.active_slider:
            return
            
        # Find the menu item for this slider
        content_left = self.menu_rect[0] + SIDEBAR_WIDTH + 20
        y_offset = self.menu_rect[1] + TOP_BAR_HEIGHT + 20
        
        for item in self.menu_items:
            if item[2] == self.active_slider:
                min_val, max_val = item[3], item[4]
                slider_x = content_left + 200
                
                # Calculate relative position
                rel_x = mx - slider_x
                rel_x = max(0, min(rel_x, SLIDER_WIDTH))
                
                # Calculate value
                value = min_val + (rel_x / SLIDER_WIDTH) * (max_val - min_val)
                
                # Round to 2 decimal places for certain sliders
                if self.active_slider in ['trigger_delay', 'aim_delay', 'tracking_speed', 'smoothing_xy', 'humaniser', 'confidence']:
                    value = round(value, 2)
                else:
                    value = round(value)
                
                # Update config
                self.config[self.active_slider] = value
                
                # Update status message
                self.status_message = f"{item[0]}: {value}"
                self.status_message_time = time.time()
                
                win32gui.InvalidateRect(self.hwnd, None, True)
                break
            
            y_offset += BUTTON_HEIGHT + BUTTON_PADDING

    def draw_tooltip(self, hdc, x, y, text):
        # Draw a simple tooltip box with text
        PADDING = 6
        logfont = win32gui.LOGFONT()
        logfont.lfHeight = -12
        logfont.lfWeight = win32con.FW_NORMAL
        logfont.lfFaceName = 'Segoe UI'
        logfont.lfQuality = win32con.ANTIALIASED_QUALITY
        font = win32gui.CreateFontIndirect(logfont)
        old_font = win32gui.SelectObject(hdc, font)
        win32gui.SetTextColor(hdc, rgb_to_colorref((255, 255, 255)))
        win32gui.SetBkMode(hdc, win32con.OPAQUE)
        win32gui.SetBkColor(hdc, rgb_to_colorref((40, 40, 40)))
        # Calculate text size
        text_size = win32gui.GetTextExtentPoint32(hdc, text)
        width, height = text_size[0] + 2 * PADDING, text_size[1] + 2 * PADDING
        rect = (x, y, x + width, y + height)
        win32gui.ExtTextOut(hdc, x + PADDING, y + PADDING, 0, rect, text, None)
        win32gui.SelectObject(hdc, old_font)
        win32gui.SetBkMode(hdc, win32con.TRANSPARENT)

def main():
    """Main function to run the overlay."""
    try:
        logging.info("Starting overlay application")
        
        # Create and show the overlay window
        overlay = OverlayWindow()
        logging.info("Overlay window created")
        
        # Start the detection thread
        overlay.detector.start_detection()
        logging.info("Detection thread started")
        
        # Main message loop
        while overlay.running:
            try:
                msg = win32gui.GetMessage(overlay.hwnd, 0, 0)
                if msg[0]:
                    win32gui.TranslateMessage(msg[1])
                    win32gui.DispatchMessage(msg[1])
            except Exception as e:
                logging.error(f"Error in message loop: {e}")
                logging.error(traceback.format_exc())
                break
        
        # Cleanup
        overlay.detector.stop_detection()
        logging.info("Detection thread stopped")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        logging.error(traceback.format_exc())
    finally:
        logging.info("Application shutting down")

if __name__ == "__main__":
    main()
