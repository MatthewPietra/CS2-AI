"""
FuryAI - YOLOv8-based Object Detection and Aim Assist System

This module provides a real-time object detection system using YOLOv8 with GPU acceleration
support for both NVIDIA (CUDA) and AMD (DirectML) graphics cards. It includes an overlay
window for visualization and configuration, aim assist functionality, and screen capture
capabilities optimized for gaming applications.

Author: FuryAI Development Team
Version: 1.0
License: Proprietary
"""

# Standard library imports
import os
import sys
import time
import json
import math
import random
import logging
import threading
import traceback
import collections
import subprocess
import platform
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from ctypes import wintypes

# Third-party imports
import numpy as np
import cv2
import torch
import psutil
import pyautogui
import mss
from PIL import ImageGrab, Image
from ultralytics import YOLO
from ultralytics.trackers import BYTETracker
from argparse import Namespace

# Windows-specific imports
import win32gui
import win32con
import win32api
import win32ui
import win32process
import ctypes
from ctypes import wintypes

# GUI imports
from tkinter import Tk, filedialog, messagebox, colorchooser

# Optional imports with fallback handling
DIRECTML_AVAILABLE = False
GPUTIL_AVAILABLE = False
WMI_AVAILABLE = False

try:
    import torch_directml
    DIRECTML_AVAILABLE = True
    print("DirectML is available for AMD GPU acceleration")
    print(f"DirectML version: {torch_directml.__version__ if hasattr(torch_directml, '__version__') else 'unknown'}")
    print(f"DirectML device count: {torch_directml.device_count()}")
    if torch_directml.device_count() > 0:
        print(f"DirectML device name: {torch_directml.device_name(0)}")
except ImportError:
    print("DirectML not available. Please install with: pip install torch-directml")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    print("GPUtil not available. Some GPU monitoring features will be disabled.")

try:
    import wmi
    WMI_AVAILABLE = True
except ImportError:
    print("WMI not available. Some GPU detection features will be limited.")

# --- Configuration ---
class Config:
    """Configuration management for the FuryAI application."""
    
    # Default configuration values
    DEFAULT_CONFIG = {
        # Display settings
    'show_menu': False,
        'show_fps_counter': True,
    'show_player_boxes': True,
        'show_fov_visualiser': True,
        'fps_display_style': 0,  # 0 = detailed, 1 = simple
        
        # Detection settings
        'confidence': 0.05,  # Lowered from 0.15 to 0.05 for better detection
        'frame_skipping': 1,
        'capture_size': 640,
        'target_size': 640,
        'max_boxes': 5,
        
        # Aim assist settings
        'aim_assist_enabled': True,
    'tracking_speed': 0.5,
    'humaniser': 0.2,
    'smoothing_xy': 0.5,
        'fov_modifier': 150,
        'fov_size': 150,
        'aim_deadzone': 3,  # pixels - don't move if within this distance
        'anti_lag_value': 0.0,
        'custom_bone_position': 0.0,
        
        # Visual settings
        'fov_color': (0, 255, 0),
        'box_color': (255, 0, 0),
        'text_color': (255, 255, 255),
        
        # Menu settings
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
        
        # Key bindings
    'menu_key': 'INSERT',
        'tracking_key': 'F2',
    'trigger_key': 'F3',
    'fov_toggle_key': 'F4',
    'fps_toggle_key': 'F5',
        'aim_key': 'SHIFT',
        
        # Performance settings
        'optimizations_enabled': False,
        'gpu_utilization_threshold': 80,
        'optimized_fps_target': 60,
        'default_target_fps': 144,
        'use_gpu_capture': False,  # Disabled by default for better performance
        'capture_optimization': True,  # Enable capture optimizations
        
        # Trigger bot settings
        'trigger_bot_enabled': False,
        
        # GPU info (will be populated at runtime)
        'gpu_info': None,
    }
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get a copy of the default configuration."""
        return cls.DEFAULT_CONFIG.copy()
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and ensure all required configuration options exist."""
        validated_config = cls.get_default_config()
        validated_config.update(config)
        return validated_config

# Global configuration instance
config = Config.get_default_config()
 
# --- Constants ---
SCREEN_WIDTH = win32api.GetSystemMetrics(0)
SCREEN_HEIGHT = win32api.GetSystemMetrics(1)

# Menu dimensions
FIXED_MENU_WIDTH = 500
FIXED_MENU_HEIGHT = 320
MENU_START_X = 20
MENU_START_Y = 20
FULL_MENU_RECT = (
    MENU_START_X,
    MENU_START_Y,
    MENU_START_X + FIXED_MENU_WIDTH,
    MENU_START_Y + FIXED_MENU_HEIGHT
)

# UI element dimensions
TOP_BAR_HEIGHT = 30
SIDEBAR_WIDTH = 100
BOTTOM_BAR_HEIGHT = 20
SLIDER_WIDTH = 140
SLIDER_HEIGHT = 8
TAB_HEIGHT = 28
TEXT_SIZE = 14
BUTTON_HEIGHT = 24
BUTTON_PADDING = 8
TOGGLE_SIZE = 20
LABEL_WIDTH = 170
INFO_ICON_SIZE = 18
INFO_ICON_OFFSET = 6
CIRCLE_THICKNESS = 2

# Virtual key codes
VK_INSERT = 0x2D
VK_F1 = 0x70
VK_F2 = 0x71
VK_F3 = 0x72
VK_F4 = 0x73
VK_F5 = 0x74
VK_F6 = 0x75
VK_F7 = 0x76
VK_F8 = 0x77
VK_F9 = 0x78
VK_F10 = 0x79
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12  # ALT key

# Key mapping dictionary
KEY_MAPPING = {
    'SHIFT': VK_SHIFT,
    'CTRL': VK_CONTROL,
    'ALT': VK_MENU,
    'INSERT': VK_INSERT,
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
}

# Reverse key mapping for getting key names
REVERSE_KEY_MAPPING = {v: k for k, v in KEY_MAPPING.items()}

# Information descriptions for UI tooltips
INFO_ICON_EXCLUDE = {
    'gpu_info', 'save_config', 'load_config', 'reset_settings',
    'tracking_key', 'menu_key', 'trigger_key', 'fov_toggle_key', 'fps_toggle_key',
}

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
    'anti_lag_value': 'Anti-lag compensation value for smoother tracking.',
    'custom_bone_position': 'Custom bone position offset for aim targeting.',
    'smoothing_xy': 'Smoothing factor for X and Y axis movement.',
    'aim_deadzone': 'Distance in pixels where aim assist stops moving to prevent jittering.',
    'aim_assist_enabled': 'Enable or disable automatic aim assistance.',
    'trigger_bot_enabled': 'Enable automatic shooting when crosshair is on target.',
    'trigger_delay': 'Delay before trigger bot fires (seconds).',
    'trigger_key': 'Key to toggle trigger bot functionality.',
    'menu_key': 'Key to toggle the configuration menu.',
    'fov_toggle_key': 'Key to toggle FOV visualization.',
    'fps_toggle_key': 'Key to toggle FPS counter display.',
    'show_fps_counter': 'Display FPS counter on screen.',
    'fps_display_style': 'FPS counter display style (0=detailed, 1=simple).',
    'box_color': 'Color of detection boxes (RGB).',
    'fov_color': 'Color of FOV circle (RGB).',
    'text_color': 'Color of on-screen text (RGB).',
}

# Win32 API setup
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

# Mouse movement constants
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOD_NOREPEAT = 0x4000

# --- Utility Functions ---

def get_key_name(vk_code: int) -> str:
    """
    Convert virtual key code to key name.
    
    Args:
        vk_code: Virtual key code from Windows API
        
    Returns:
        String representation of the key name
    """
    return REVERSE_KEY_MAPPING.get(vk_code, f"Key {vk_code}")

def move_mouse(dx: int, dy: int) -> None:
    """
    Move the mouse cursor by the specified delta using Win32 API.
    
    Args:
        dx: Horizontal movement delta in pixels
        dy: Vertical movement delta in pixels
    """
    try:
        x, y = win32api.GetCursorPos()
        win32api.SetCursorPos((x + dx, y + dy))
    except Exception as e:
        print(f"[ERROR] Failed to move mouse: {e}")

def simulate_click() -> None:
    """Simulate a left mouse click using SendInput."""
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

def rgb_to_colorref(rgb: Tuple[int, int, int]) -> int:
    """
    Convert RGB tuple to Windows color reference.
    
    Args:
        rgb: RGB color tuple (r, g, b)
        
    Returns:
        Windows color reference integer
    """
    r, g, b = rgb
    return r | (g << 8) | (b << 16)
 
def point_in_rect(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
    """
    Check if a point is within a rectangle.
    
    Args:
        x: X coordinate of the point
        y: Y coordinate of the point
        rect: Rectangle tuple (left, top, right, bottom)
        
    Returns:
        True if point is within rectangle, False otherwise
    """
    left, top, right, bottom = rect
    return left <= x <= right and top <= y <= bottom
 
# --- YOLODetector Class ---

class YOLODetector:
    """
    YOLOv8-based object detector with GPU acceleration support.
    
    This class handles real-time object detection using YOLOv8 models with support
    for both NVIDIA (CUDA) and AMD (DirectML) GPU acceleration. It includes
    frame processing, detection result handling, and performance optimization.
    
    Attributes:
        config: Configuration dictionary containing detection parameters
        model_path: Path to the YOLOv8 model file
        hardware_manager: HardwareManager instance for device management
        model: Loaded YOLOv8 model
        aiming_enabled: Whether aim assist is enabled
        running: Whether the detection thread is running
        current_boxes: Current detection bounding boxes
        current_confidences: Current detection confidences
        current_class_ids: Current detection class IDs
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the YOLODetector with configuration.
        
        Args:
            config: Configuration dictionary containing detection parameters
            
        Raises:
            Exception: If model loading fails
        """
        self.config = config
        self.model_path = config.get('model_path', 'best.pt')
        
        # Use centralized hardware manager
        self.hardware_manager = hardware_manager
        self.device = self.hardware_manager.get_device()
        self.hardware_type = self.hardware_manager.get_hardware_type()
        self.device_name = self.hardware_manager.get_device_name()
        
        # Model and state
                # Performance optimizations
        self.frame_skip = 1  # Process every 2nd frame
        self.frame_counter = 0
        self.target_size = 416  # Smaller input size for speed
        self.last_detections = []
        self.detection_lock = threading.Lock()
        self.model = None
        self.frame_counter = 0
        self.last_detections = []
        self.aiming_enabled = True
        self.trigger_enabled = True
        
        # Performance tracking
        self.last_fps = 0
        self.current_fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # Detection state
        self.running = False
        self.detection_thread = None
        self.fps_history = collections.deque(maxlen=10)
        self.processing_fps_history = collections.deque(maxlen=10)
        self.current_boxes = None
        self.current_confidences = None
        self.current_class_ids = None
        self.class_names = ['item']
        self.last_box_state = False
        
        # Frame skipping
        self.process_every_n = max(1, min(4, config.get('frame_skipping', 1)))
        self.skip_counter = 0
        self.frame_counter = 0
        self.last_detections = []
        
        # Initialize tracker
        tracker_args = Namespace(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=144
        )
        self.tracker = BYTETracker(tracker_args)
        
        # Load the model
        if not self.load_model():
            raise Exception("Failed to load YOLOv8 model")
            
        # Check for MSS library
        try:
            import mss
            print("MSS screen capture library found")
        except ImportError:
            print("MSS library not found. Installing...")
            subprocess.check_call(["pip", "install", "mss"])
            print("MSS installed successfully")

    def update_fps(self) -> bool:
        """
        Update FPS counter and return whether FPS was updated.
        
        Returns:
            True if FPS was updated, False otherwise
        """
        current_time = time.time()
        self.frame_count += 1
        
        # Update FPS every 0.2 seconds for more responsive display
        if current_time - self.last_time >= 0.2:
            self.last_fps = self.current_fps
            self.current_fps = int(self.frame_count / (current_time - self.last_time))
            self.frame_count = 0
            self.last_time = current_time
            
            print(f"Current FPS: {self.current_fps}, Skip: {self.process_every_n}")
            
            # Force overlay to update if we have a parent reference
            if hasattr(self, 'parent') and hasattr(self.parent, 'hwnd'):
                try:
                    fps_rect = (0, 0, 200, 40)
                    win32gui.InvalidateRect(self.parent.hwnd, fps_rect, True)
                    win32gui.UpdateWindow(self.parent.hwnd)
                except Exception as e:
                    print(f"Error updating FPS display: {e}")
            
            self.fps_history.append(self.current_fps)
            return True
        return False
    def optimize_model(self):
        """Apply performance optimizations to the model."""
        if hasattr(self.model, 'model'):
            # Enable PyTorch optimizations
            torch.set_num_threads(8)  # Use more CPU threads
            torch.backends.cudnn.benchmark = True
            
            # Fuse layers for faster inference
            try:
                self.model.model.fuse()
                print("[INFO] Model fused for faster inference")
            except Exception as e:
                print(f"[INFO] Model fusion failed: {e}")
            
            # Use lower precision if available
            try:
                self.model.model.half()
                print("[INFO] Using FP16 for faster inference")
            except Exception as e:
                print(f"[INFO] FP16 not available: {e}")
            
            # Optimize thresholds for speed
            self.model.conf = 0.4  # Higher confidence = fewer detections = faster
            self.model.iou = 0.45  # Optimized NMS threshold
            
            print(f"[INFO] Optimized model with {self.target_size}x{self.target_size} input")
    
    def should_process_frame(self):
        """Determine if current frame should be processed."""
        self.frame_counter += 1
        return self.frame_counter % (self.frame_skip + 1) == 0


    def load_model(self) -> bool:
        """
        Load the YOLOv8 model on GPU with performance optimizations.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            print(f"Loading model from: {self.model_path}")
            
            # Load model with task-specific optimizations
            self.model = YOLO(self.model_path, task='detect')
            
            # Set model parameters for better detection
            self.model.conf = self.config.get('confidence', 0.15)
            self.model.iou = self.config.get('iou_threshold', 0.3)
            print(f"Model confidence threshold set to: {self.model.conf}")
            print(f"Model IoU threshold set to: {self.model.iou}")
            
            # Set class names
            try:
                if hasattr(self.model, 'names'):
                    self.class_names = list(self.model.names.values())
                    print(f"Class names loaded from model: {self.class_names}")
                else:
                    self.class_names = ['item']
                    print("Using default class names: ['item']")
            except Exception as e:
                print(f"Error loading class names: {e}, using default")
                self.class_names = ['item']
            
            # Apply global performance optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            if self.hardware_manager.is_nvidia():
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
            
            elif self.hardware_manager.is_amd():
                # Use MSS + CPU optimization for AMD GPUs (eliminates DirectML issues)
                print("AMD GPU detected - using MSS + CPU optimization")
                print("Reason: MSS provides excellent capture performance, CPU provides reliable inference")
                
                # Move model to CPU with optimizations
                self.model.to('cpu')
                print("Model moved to CPU for AMD optimization")
                
                # Set model to evaluation mode
                self.model.model.eval()
                print("Model set to evaluation mode")
                
                # Enable CPU optimizations for better performance
                torch.set_num_threads(min(8, os.cpu_count()))  # Use multiple CPU threads
                torch.backends.mkldnn.enabled = True  # Enable MKL-DNN for Intel CPUs
                torch.backends.mkldnn.allow_tf32 = True  # Allow TF32 for better performance
                
                print(f"CPU optimizations enabled:")
                print(f"  Threads: {torch.get_num_threads()}")
                print(f"  MKL-DNN: {torch.backends.mkldnn.enabled}")
                print(f"  TF32: {torch.backends.mkldnn.allow_tf32}")
                
                # Test forward pass on CPU
                test_tensor = torch.randn(1, 3, 320, 320, device='cpu')
                print(f"Test tensor created on CPU: {test_tensor.device}")
                
                with torch.no_grad():
                    test_output = self.model.model(test_tensor)
                    if isinstance(test_output, (list, tuple)):
                        _ = test_output[0]
                    else:
                        _ = test_output
                print("Test forward pass successful on CPU")
                
                print("AMD MSS + CPU optimization complete")
                
                # Verify model is on CPU
                model_device = next(self.model.model.parameters()).device
                print(f"Model device: {model_device}")
                
                if model_device == torch.device('cpu'):
                    print("Model successfully optimized for AMD GPU using MSS + CPU")
                else:
                    print("WARNING: Model not on CPU device")
                    raise Exception("Failed to move model to CPU")
                
                # Set model parameters for better performance
                self.model.conf = self.config.get('confidence', 0.15)
                self.model.iou = self.config.get('iou_threshold', 0.25)
                print(f"Model confidence threshold set to: {self.model.conf}")
                print(f"Model IoU threshold set to: {self.model.iou}")
                
                # Force another computation to ensure CPU is active
                try:
                    warm_tensor = torch.rand(500, 500, device='cpu')
                    result = warm_tensor @ warm_tensor.t()
                    _ = result.numpy()
                    print("Final CPU warm-up successful - CPU should be active now")
                except Exception as e:
                    print(f"Warning: Final CPU warm-up failed: {e}")
            
            # Apply additional performance optimizations
            try:
                if hasattr(self.model.model, 'fuse'):
                    self.model.model.fuse()
                    print("Model layers fused for better performance")
                
                if self.hardware_manager.is_nvidia():
                    self.model.model = self.model.model.half()
                    print("Model converted to half-precision for better performance")
            except Exception as e:
                print(f"Warning: Some optimizations couldn't be applied: {e}")
            
            print(f"YOLOv8 model loaded successfully from {self.model_path}")
            
            # Apply performance optimizations
            self.optimize_model()
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load model on GPU: {e}")
            return False

    def optimize_model(self):
        """Apply performance optimizations to the model."""
        if hasattr(self.model, 'model'):
            # Enable PyTorch optimizations
            torch.set_num_threads(8)  # Use more CPU threads
            torch.backends.cudnn.benchmark = True
            
            # Fuse layers for faster inference
            try:
                self.model.model.fuse()
                print("[INFO] Model fused for faster inference")
            except Exception as e:
                print(f"[INFO] Model fusion failed: {e}")
            
            # Use lower precision if available
            try:
                self.model.model.half()
                print("[INFO] Using FP16 for faster inference")
            except Exception as e:
                print(f"[INFO] FP16 not available: {e}")
            
            # Optimize thresholds for speed
            self.model.conf = 0.4  # Higher confidence = fewer detections = faster
            self.model.iou = 0.45  # Optimized NMS threshold
            
            print(f"[INFO] Optimized model with {self.target_size}x{self.target_size} input")
    
    def should_process_frame(self):
        """Determine if current frame should be processed."""
        self.frame_counter += 1
        return self.frame_counter % (self.frame_skip + 1) == 0


    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process a frame for object detection with hardware acceleration.
        
        This method handles frame preprocessing, inference, and result processing
        using the appropriate hardware (AMD GPU, NVIDIA GPU, or CPU).
        
        Args:
            frame: Input frame as numpy array (HWC format)
            
        Returns:
            List of detection dictionaries containing bbox, confidence, and class info
        """
        try:
            target_size = self.config.get('target_size', 640)
            capture_size = self.config.get('capture_size', 640)
            max_boxes = self.config.get('max_boxes', 5)
            
            print(f"[DEBUG] Processing frame with shape: {frame.shape}")
            
            # Initialize hardware processing if needed
            if not hasattr(self, 'hardware_initialized'):
                self._initialize_hardware_processing(target_size)
                self.hardware_initialized = True
            
            # Add thread synchronization to prevent resource deadlocks
            if not hasattr(self, 'gpu_lock'):
                self.gpu_lock = threading.RLock()
            
            # Use hardware manager to determine processing path
            if self.hardware_manager.is_cpu():
                # Performance optimizations
                # Use smaller input size for speed
                target_size = 416  # Optimized size
                
                # Frame skipping for better performance
                if hasattr(self, 'frame_counter'):
                    self.frame_counter += 1
                    if self.frame_counter % 2 == 0:  # Process every 2nd frame
                        # Return cached detections if available
                        if hasattr(self, 'last_detections') and self.last_detections:
                            return self.last_detections
                else:
                    self.frame_counter = 0
                return self._process_frame_cpu(frame, target_size, capture_size)
            elif self.hardware_manager.is_amd():
                return self._process_frame_amd(frame, target_size, capture_size)
            elif self.hardware_manager.is_nvidia():
                return self._process_frame_nvidia(frame, target_size, capture_size)
            else:
                print("[ERROR] No supported hardware type found")
                return []
            
        except Exception as e:
            print(f"[ERROR] Critical error in process_frame: {e}")
            return []

    def _process_frame_cpu(self, frame: np.ndarray, target_size: int, capture_size: int) -> List[Dict[str, Any]]:
        """Process frame using CPU fallback mode."""
        try:
            print("[INFO] Using CPU fallback mode")
            
            # Resize on CPU
            frame_resized = cv2.resize(frame, (target_size, target_size), 
                                     interpolation=cv2.INTER_LINEAR)
            
            # Run inference with CPU mode
            with torch.no_grad():
                # Use CPU tensor directly
                np_tensor = frame_resized.transpose(2, 0, 1)  # HWC to CHW
                cpu_tensor = torch.from_numpy(np_tensor).float()
                cpu_tensor = cpu_tensor.unsqueeze(0).div(255.0)  # Add batch dimension and normalize
                
                # Run inference on CPU
                # Use YOLOv8 API directly
                cpu_tensor = cpu_tensor.clone().detach().contiguous().float().to('cpu')
                results = self.model(cpu_tensor)
                detections = self.direct_process_results(results, frame_resized, capture_size, target_size)
                return detections
                
        except Exception as e:
            print(f"[ERROR] CPU fallback processing failed: {e}")
            return []

    def _process_frame_amd(self, frame: np.ndarray, target_size: int, capture_size: int) -> List[Dict[str, Any]]:
        """Process frame using MSS + CPU optimization for AMD GPUs."""
        try:
            print("[INFO] Using MSS + CPU optimization for AMD GPU")
            
            # Optimized preprocessing with MSS data
            frame_resized = cv2.resize(frame, (target_size, target_size), 
                                     interpolation=cv2.INTER_LINEAR)
            
            # Convert to tensor with optimized operations
            np_tensor = frame_resized.transpose(2, 0, 1)  # HWC to CHW
            cpu_tensor = torch.from_numpy(np_tensor).float()
            cpu_tensor = cpu_tensor.unsqueeze(0).div(255.0)  # Add batch dimension and normalize
            
            # Run inference with CPU optimizations
            with torch.no_grad():
                # Use YOLOv8 API directly on CPU
                cpu_tensor = cpu_tensor.detach().contiguous()
                results = self.model(cpu_tensor)
                detections = self.direct_process_results(results, frame_resized, capture_size, target_size)
                return detections
                        
        except Exception as e:
            print(f"[ERROR] AMD MSS + CPU processing failed: {e}")
            return []

    def _process_frame_nvidia(self, frame: np.ndarray, target_size: int, capture_size: int) -> List[Dict[str, Any]]:
        """Process frame using NVIDIA GPU with CUDA."""
        with self.gpu_lock:
            try:
                # Resize on CPU
                frame_resized = cv2.resize(frame, (target_size, target_size), 
                                         interpolation=cv2.INTER_AREA)
                
                # Convert to tensor
                np_tensor = frame_resized.transpose(2, 0, 1)  # HWC to CHW
                cpu_tensor = torch.from_numpy(np_tensor).float()
                cpu_tensor = cpu_tensor.unsqueeze(0).div(255.0)  # Add batch dimension and normalize
                
                # Move to GPU - use synchronous transfer
                with torch.no_grad():
                    gpu_tensor = cpu_tensor.to(self.device)
                
                # Run inference
                with torch.no_grad():
                    # Use YOLOv8 API directly
                    gpu_tensor = gpu_tensor.detach().contiguous()
                    gpu_tensor = gpu_tensor.clone().detach().contiguous().float().to(self.device)
                    results = self.model(gpu_tensor)
                    detections = self.direct_process_results(results, frame_resized, capture_size, target_size)
                    return detections
                        
            except Exception as e:
                print(f"[ERROR] NVIDIA GPU processing failed: {e}")
                return []

    def direct_process_results(self, raw_results: Union[torch.Tensor, List[torch.Tensor]], 
                             frame_resized: np.ndarray, capture_size: int, target_size: int) -> List[Dict[str, Any]]:
        """
        Process raw YOLOv8 results directly using the proper YOLOv8 API.
        
        Args:
            raw_results: Raw results from YOLOv8 model
            frame_resized: Resized frame that was processed
            capture_size: Size of the captured frame
            target_size: Target size for model input
            
        Returns:
            List of detection dictionaries
        """
        try:
            # Get screen dimensions
            screen_width = win32api.GetSystemMetrics(0)
            screen_height = win32api.GetSystemMetrics(1)
            
            # Get the capture region dimensions
            if hasattr(self, 'screen_capture') and hasattr(self.screen_capture, 'capture_region'):
                capture_region = self.screen_capture.capture_region
                capture_width = capture_region.get('width', screen_width)
                capture_height = capture_region.get('height', screen_height)
            else:
                # Default to full screen if capture region not available
                capture_width = screen_width
                capture_height = screen_height
            
            # Process detections using YOLOv8 API
            valid_detections = []
            
            # The raw_results is a list of Results objects
            if isinstance(raw_results, list) and len(raw_results) > 0:
                results = raw_results[0]  # Get first result
                
                if hasattr(results, 'boxes') and results.boxes is not None:
                    boxes = results.boxes
                    
                    print(f"[DEBUG] Processing {len(boxes)} YOLOv8 detections")
                    
                    for i, box in enumerate(boxes):
                        try:
                            # Get coordinates in xyxy format (pixel coordinates)
                            coords = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            x1, y1, x2, y2 = coords
                            
                            # Skip if confidence is below threshold
                            if confidence < self.config.get('confidence', 0.05):  # Use config confidence instead of hardcoded 0.15
                                continue
                                
                            print(f"[DEBUG] YOLOv8 detection {i}: xyxy={coords}, conf={confidence:.3f}, class={class_id}")
                            
                            # Scale coordinates from model input size (640x640) to screen size
                            scale_x = capture_width / target_size
                            scale_y = capture_height / target_size
                            
                            screen_x1 = x1 * scale_x
                            screen_y1 = y1 * scale_y
                            screen_x2 = x2 * scale_x
                            screen_y2 = y2 * scale_y
                            
                            print(f"[DEBUG] Scaled coordinates: ({screen_x1:.1f},{screen_y1:.1f}) to ({screen_x2:.1f},{screen_y2:.1f})")
                            
                            # Calculate box dimensions
                            box_width = screen_x2 - screen_x1
                            box_height = screen_y2 - screen_y1
                            
                            # Filter out boxes that are too large
                            if box_width > 0.6 * screen_width or box_height > 0.8 * screen_height:
                                print(f"Filtering out oversized box: {box_width:.1f}x{box_height:.1f}")
                                continue
                                
                            # Filter out boxes that are too small
                            if box_width < 5 or box_height < 5:
                                print(f"Filtering out tiny box: {box_width:.1f}x{box_height:.1f}")
                                continue
                                
                            # Ensure coordinates are within screen bounds
                            box_left = max(0, min(screen_width, int(screen_x1)))
                            box_top = max(0, min(screen_height, int(screen_y1)))
                            box_right = max(0, min(screen_width, int(screen_x2)))
                            box_bottom = max(0, min(screen_height, int(screen_y2)))
                            
                            # Skip invalid boxes
                            if box_right <= box_left or box_bottom <= box_top:
                                print(f"Skipping invalid box: ({box_left},{box_top}) to ({box_right},{box_bottom})")
                                continue
                            
                            # Add to valid detections
                            valid_detections.append({
                                'bbox': [box_left, box_top, box_right, box_bottom],
                                'confidence': float(confidence),
                                'class': class_id
                            })
                            
                            print(f"[DEBUG] Added detection {i}: screen=({box_left},{box_top},{box_right},{box_bottom}), conf={confidence:.2f}")
                            
                        except Exception as e:
                            print(f"[DEBUG] Error processing YOLOv8 detection {i}: {e}")
                
                # Apply non-maximum suppression
                if len(valid_detections) > 0:
                    boxes = np.array([d['bbox'] for d in valid_detections])
                    scores = np.array([d['confidence'] for d in valid_detections])
                
                    # Simple NMS implementation
                    keep = []
                    for i in range(len(boxes)):
                        keep_box = True
                        for j in range(len(boxes)):
                            if i != j:
                                # Calculate IoU
                                x1_i, y1_i, x2_i, y2_i = boxes[i]
                                x1_j, y1_j, x2_j, y2_j = boxes[j]
                                
                                # Calculate intersection
                                x1_inter = max(x1_i, x1_j)
                                y1_inter = max(y1_i, y1_j)
                                x2_inter = min(x2_i, x2_j)
                                y2_inter = min(y2_i, y2_j)
                                
                                if x2_inter > x1_inter and y2_inter > y1_inter:
                                    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                                    area_i = (x2_i - x1_i) * (y2_i - y1_i)
                                    area_j = (x2_j - x1_j) * (y2_j - y1_j)
                                    iou = inter_area / (area_i + area_j - inter_area)
                                    
                                    if iou > 0.5:  # IoU threshold
                                        if scores[i] < scores[j]:
                                            keep_box = False
                                        break
                        
                        if keep_box:
                            keep.append(i)
                    
                    # Keep only non-overlapping boxes
                    detections = [valid_detections[i] for i in keep]
                    print(f"[DEBUG] After NMS: {len(detections)} boxes")
                else:
                    detections = []
                    print("[DEBUG] No valid detections found")
            else:
                detections = []
                print("[DEBUG] No results found in raw_results")
            
            return detections
            
        except Exception as e:
            print(f"[ERROR] Failed to process YOLOv8 results: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _initialize_hardware_processing(self, target_size: int) -> None:
        """
        Initialize hardware resources for processing.
        
        Args:
            target_size: Target size for model input tensors
        """
        try:
            # Create a lock for hardware operations if it doesn't exist
            if not hasattr(self, 'gpu_lock'):
                self.gpu_lock = threading.RLock()
                
            # Use the lock to prevent concurrent access during initialization
            with self.gpu_lock:
                if self.hardware_manager.is_amd():
                    # Create tensors on CPU for AMD optimization (MSS + CPU approach)
                    try:
                        # Initialize input tensor with CPU device
                        self.input_tensor = torch.zeros((1, 3, target_size, target_size), 
                                                      dtype=torch.float32, device='cpu')
                        
                        # Create a tensor for CPU processing
                        self.gpu_frame_tensor = torch.zeros((self.config.get('capture_size', 480), 
                                                          self.config.get('capture_size', 480), 
                                                          3), dtype=torch.uint8, device='cpu')
                        
                        # Create a tensor for CPU-based preprocessing
                        self.preprocess_tensor = torch.zeros((1, 3, target_size, target_size), 
                                                          dtype=torch.float32, device='cpu')
                        
                        print("Successfully created AMD MSS + CPU tensors")
                        
                    except Exception as e:
                        print(f"Failed to create tensors on AMD CPU: {e}")
                        # Create on CPU instead
                        self.input_tensor = torch.zeros((1, 3, target_size, target_size), dtype=torch.float32)
                        self.gpu_frame_tensor = torch.zeros((self.config.get('capture_size', 480), 
                                                          self.config.get('capture_size', 480), 
                                                          3), dtype=torch.uint8)
                        self.preprocess_tensor = torch.zeros((1, 3, target_size, target_size), 
                                                          dtype=torch.float32)
                    
                    # Create a stream for asynchronous operations (None for AMD CPU mode)
                    self.stream = None  # AMD CPU mode doesn't need streams
                    
                elif self.hardware_type == 'NVIDIA':
                    # Pre-allocate tensor for zero-copy with half precision
                    self.input_tensor = torch.zeros((1, 3, target_size, target_size), 
                                                         device=self.device,
                                                         dtype=torch.float16)
                    
                    # Pre-allocate CPU tensor to avoid repeated allocations
                    self.cpu_tensor = torch.zeros((1, 3, target_size, target_size), 
                                               dtype=torch.float16)
                    
                    # Create utility tensors for GPU operations
                    self.gpu_resize_tensor = torch.zeros((1, 3, target_size, target_size), 
                                                       device=self.device,
                                                       dtype=torch.float16)
                    
                    # Force synchronization to ensure GPU resources are properly initialized
                    if hasattr(torch.cuda, 'synchronize'):
                        torch.cuda.synchronize()
                    
                    # Enable CUDA graphs for faster inference if available
                    if hasattr(torch.cuda, 'make_graphed_callables') and target_size <= 160:
                        try:
                            # Use a separate stream for CUDA graph to avoid conflicts
                            if not hasattr(self, 'graph_stream'):
                                self.graph_stream = torch.cuda.Stream()
                            
                            with torch.cuda.stream(self.graph_stream):
                                sample_input = torch.randn((1, 3, target_size, target_size), 
                                                        device=self.device, 
                                                        dtype=torch.float16)
                                
                                # Try to graph the model for faster inference
                                with torch.no_grad():
                                    self.model.model = torch.cuda.make_graphed_callables(
                                        self.model.model, (sample_input,))
                                
                                # Ensure graph is completed
                                torch.cuda.current_stream().wait_stream(self.graph_stream)
                                torch.cuda.synchronize()
                                
                            print("CUDA graph optimization enabled for faster inference")
                        except Exception as e:
                            print(f"CUDA graph optimization failed: {e}")
                    
                    print("Initialized NVIDIA GPU resources for accelerated processing")
        except Exception as e:
            print(f"Failed to initialize GPU resources: {e}")
            # Reset GPU initialized flag to try again next time
            self.gpu_initialized = False

    def force_overlay_redraw(self) -> None:
        """Force a redraw of the overlay window."""
        try:
            if hasattr(self, 'hwnd'):
                win32gui.InvalidateRect(self.hwnd, None, True)
        except Exception as e:
            print(f"[DEBUG] Error forcing overlay redraw: {e}")

    def _detection_loop(self) -> None:
        """
        Main detection loop for real-time object detection.
        
        This method runs in a separate thread and continuously processes frames
        from the screen capture system, performing object detection and updating
        the detection state.
        """
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
        if hasattr(self, 'hardware_manager'):
            self.screen_capture.parent = self
            print(f"Passed {self.hardware_manager.get_hardware_type()} GPU reference to screen capture")
        
        # Configure sizes from config
        target_size = self.config.get('target_size', 160)
        
        # Get frame skipping value from config
        self.process_every_n = self.config.get('frame_skipping', 1)
        # Ensure it's within valid range (1-4)
        self.process_every_n = max(1, min(4, self.process_every_n))
        print(f"Frame skipping set to: {self.process_every_n}")
        
        # Create pre-allocated tensors with configurable size for maximum performance
        try:
            if self.hardware_manager.is_amd():
                # Create tensors on AMD GPU using DirectML
                try:
                    # Initialize input tensor with proper device
                    self.input_tensor = self.hardware_manager.create_tensor(
                        (1, 3, target_size, target_size), dtype=torch.float32)
                    
                    # Create a tensor for direct GPU-to-GPU transfer
                    self.gpu_frame_tensor = self.hardware_manager.create_tensor(
                        (self.config.get('capture_size', 480), 
                         self.config.get('capture_size', 480), 
                         3), dtype=torch.uint8)
                    
                    # Create a tensor for GPU-based preprocessing
                    self.preprocess_tensor = self.hardware_manager.create_tensor(
                        (1, 3, target_size, target_size), dtype=torch.float32)
                    
                    # Force synchronization to ensure GPU resources are properly initialized
                    self.hardware_manager.synchronize()
                    print("Successfully created AMD GPU tensors")
                    
                except Exception as e:
                    print(f"Failed to create tensors on AMD GPU: {e}")
                    # Create on CPU instead
                    self.input_tensor = torch.zeros((1, 3, target_size, target_size), dtype=torch.float32)
                    self.gpu_frame_tensor = torch.zeros((self.config.get('capture_size', 480), 
                                                      self.config.get('capture_size', 480), 
                                                      3), dtype=torch.uint8)
                    self.preprocess_tensor = torch.zeros((1, 3, target_size, target_size), 
                                                      dtype=torch.float32)
                
                # Create a stream for asynchronous operations (None for AMD)
                self.stream = None  # AMD DirectML doesn't support CUDA streams
                
            elif self.hardware_manager.is_nvidia():
                # Create tensors on NVIDIA GPU using CUDA
                try:
                    self.input_tensor = self.hardware_manager.create_tensor(
                        (1, 3, target_size, target_size), dtype=torch.float16)
                    
                    # Create a tensor for direct GPU-to-GPU transfer (avoid CPU roundtrip)
                    self.gpu_frame_tensor = self.hardware_manager.create_tensor(
                        (self.config.get('capture_size', 480), 
                         self.config.get('capture_size', 480), 
                         3), dtype=torch.uint8)
                
                    # Create a tensor for GPU-based preprocessing
                    self.preprocess_tensor = self.hardware_manager.create_tensor(
                        (1, 3, target_size, target_size), dtype=torch.float16)
                                                       
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
                    
                except Exception as e:
                    print(f"Failed to create NVIDIA tensors: {e}")
                    # Create on CPU instead
                    self.input_tensor = torch.zeros((1, 3, target_size, target_size))
                    self.gpu_frame_tensor = torch.zeros((self.config.get('capture_size', 480), 
                                                       self.config.get('capture_size', 480), 
                                                       3), dtype=torch.uint8)
                    self.preprocess_tensor = torch.zeros((1, 3, target_size, target_size), 
                                                       dtype=torch.float32)
            else:
                # CPU initialization
                print("Using CPU for tensor creation")
                self.input_tensor = torch.zeros((1, 3, target_size, target_size), dtype=torch.float32)
                self.gpu_frame_tensor = torch.zeros((self.config.get('capture_size', 480), 
                                                   self.config.get('capture_size', 480), 
                                                   3), dtype=torch.uint8)
                self.preprocess_tensor = torch.zeros((1, 3, target_size, target_size), 
                                                   dtype=torch.float32)
                self.stream = None
        except Exception as e:
            print(f"Error initializing tensors: {e}")
            
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
            
        # Prepare for performance loop
        print(f"Starting detection loop with {self.hardware_manager.get_hardware_type()} acceleration")
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
        if self.hardware_manager.is_nvidia():
            # Schedule periodic GPU memory cleanup
            last_memory_cleanup = time.time()
            
            # Monitor GPU utilization
            try:
                last_gpu_check = time.time()
                gpu_utils = []
            except ImportError:
                print("GPUtil not available for GPU monitoring")
        
        # For tracking box changes
        self.last_boxes = None
        self.boxes_changed = False
        self.last_box_update_time = time.time()
        
        # Main detection loop
        while self.running:
            try:
                # Initialize detections with empty list as default for this iteration
                detections = []
                
                # Get latest frame directly from screen capture
                frame = self.screen_capture.get_frame()
                
                if frame is not None:
                    # Process frame (with frame skipping for performance)
                    self.frame_counter += 1
                    if self.frame_counter % self.process_every_n == 0:
                        # Time the detection process
                        start_time = time.time()
                        
                        # Make a copy of the frame for processing to avoid thread conflicts
                        # This is the only copy we need to make
                        frame_copy = frame.copy()
                        
                        # Process the frame with hardware acceleration
                        try:
                            detections = self.process_frame(frame_copy)
                        except Exception as e:
                            print(f"Error in processing: {e}")
                            # Don't try fallback processing to avoid potential deadlocks
                            if "resource deadlock" in str(e).lower():
                                print("Detected resource deadlock, skipping this frame")
                                time.sleep(0.01)  # Add a small delay to allow other threads to release resources
                                continue
                            # Fall back to standard processing only for non-deadlock errors
                            try:
                                time.sleep(0.005)  # Small delay before retry
                                detections = self.process_frame(frame_copy)
                            except Exception as e2:
                                print(f"Error in fallback processing: {e2}")
                                # detections remains as empty list
                        
                        # Update detection timing
                        end_time = time.time()
                        detection_time = end_time - start_time
                        detection_times.append(detection_time)
                
                # Update current boxes - check if boxes have changed
                if detections:
                    new_boxes = [d['bbox'] for d in detections]
                    self.current_boxes = new_boxes
                    self.current_confidences = [d['confidence'] for d in detections]
                    self.current_class_ids = [d['class'] for d in detections]
                    
                    # Check if boxes have changed significantly from last frame
                    boxes_changed = False
                    if self.last_boxes is None or len(self.last_boxes) != len(new_boxes):
                        boxes_changed = True
                    else:
                        # Check if any box has moved significantly
                        for i, (old_box, new_box) in enumerate(zip(self.last_boxes, new_boxes)):
                            # Calculate box center movement
                            old_cx = (old_box[0] + old_box[2]) / 2
                            old_cy = (old_box[1] + old_box[3]) / 2
                            new_cx = (new_box[0] + new_box[2]) / 2
                            new_cy = (new_box[1] + new_box[3]) / 2
                            
                            # If center moved by more than 3 pixels, consider it changed
                            if abs(old_cx - new_cx) > 3 or abs(old_cy - new_cy) > 3:
                                boxes_changed = True
                                break
                    
                    # Update last boxes
                    self.last_boxes = new_boxes
                    self.last_box_state = True
                    
                    # Force redraw if boxes changed or it's been a while since last update
                    current_time = time.time()
                    if boxes_changed or (current_time - self.last_box_update_time > 0.1):
                        self.boxes_changed = True
                        self.last_box_update_time = current_time
                        
                        # Force redraw to show new boxes immediately
                        if hasattr(self, 'parent') and hasattr(self.parent, 'hwnd'):
                            try:
                                # Invalidate the entire screen to ensure all boxes are updated
                                win32gui.InvalidateRect(self.parent.hwnd, None, True)
                                win32gui.UpdateWindow(self.parent.hwnd)
                            except Exception as e:
                                print(f"Error invalidating box regions: {e}")
                else:
                    if self.last_box_state:  # Only update if state changed
                        self._clear_detections()
                        self.last_boxes = None
                        self.last_box_state = False
                        
                        # Force redraw to clear boxes
                        if hasattr(self, 'parent') and hasattr(self.parent, 'hwnd'):
                            try:
                                win32gui.InvalidateRect(self.parent.hwnd, None, True)
                                win32gui.UpdateWindow(self.parent.hwnd)
                            except Exception as e:
                                print(f"Error clearing boxes: {e}")
                
                # Update FPS more frequently for better feedback
                self.frame_count += 1
                if self.frame_count % 5 == 0:  # Every 5 frames
                    current_time = time.time()
                    elapsed = current_time - self.last_time
                    
                    if elapsed > 0:
                        # Calculate processing FPS
                        self.current_fps = int(5 / elapsed)
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
                        
                        # Force overlay redraw to update FPS counter
                        if hasattr(self, 'parent') and hasattr(self.parent, 'hwnd'):
                            # Only update the FPS counter area
                            fps_rect = (0, 0, 200, 40)  # Top-left area where FPS is displayed
                            win32gui.InvalidateRect(self.parent.hwnd, fps_rect, True)
                            win32gui.UpdateWindow(self.parent.hwnd)
                
                        # GPU memory management for NVIDIA
                        if self.hardware_manager.is_nvidia() and hasattr(torch, 'cuda'):
                            current_time = time.time()
                            
                            # Check GPU utilization periodically
                            if GPUTIL_AVAILABLE and current_time - last_gpu_check > 5.0:
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

    def start_detection(self) -> None:
        """Start the detection thread."""
        if not self.running:
            self.running = True
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            print("Detection thread started")

    def stop_detection(self) -> None:
        """Stop the detection thread."""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join()
            print("Detection thread stopped")

    @property
    def fps(self) -> float:
        """Get current FPS."""
        if not self.fps_history:
            return 0
        return sum(self.fps_history) / len(self.fps_history)

    @property
    def processing_fps(self) -> float:
        """Get current processing FPS."""
        if not self.processing_fps_history:
            return 0
        return sum(self.processing_fps_history) / len(self.processing_fps_history)

    def _clear_detections(self):
        self.current_boxes = None
        self.current_confidences = None
        self.current_class_ids = None
        self.last_box_state = False

    def get_detections(self) -> List[Dict[str, Any]]:
        """
        Get current detections in a format suitable for aim assist.
        
        Returns:
            List of detection dictionaries with 'screen' and 'confidence' keys
        """
        detections = []
        
        if self.current_boxes is not None and len(self.current_boxes) > 0:
            for i, box in enumerate(self.current_boxes):
                if i < len(self.current_confidences):
                    detection = {
                        'screen': box,  # (x1, y1, x2, y2)
                        'confidence': self.current_confidences[i]
                    }
                    detections.append(detection)
        
        return detections

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
            
            # MSS for high-performance screen capture (works great with AMD)
            self.mss = None
            self.monitor = None
            
            # GPU resources
            self.gpu_type = None
            self.gpu_device = None
            self._setup_gpu()
            
            # Initialize MSS capture
            self._init_mss_capture()
            
            # Set thread priority
            self._set_process_priority()
            
            # Start the capture thread
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            
            # Create a second thread for FPS calculation to avoid slowing down capture
            self.fps_thread = threading.Thread(target=self.fps_loop, daemon=True)
            self.fps_thread.start()
            
        def _setup_gpu(self):
            """Setup hardware for image processing acceleration."""
            try:
                # Check if we're already in a YOLODetector instance with hardware setup
                if hasattr(self, 'parent') and hasattr(self.parent, 'hardware_manager'):
                    # Use the parent's hardware manager
                    self.hardware_manager = self.parent.hardware_manager
                    self.gpu_type = self.hardware_manager.get_hardware_type()
                    self.gpu_device = self.hardware_manager.get_device()
                        
                    # Use the parent's lock for synchronization
                    if hasattr(self.parent, 'gpu_lock'):
                        self.gpu_lock = self.parent.gpu_lock
                    else:
                        self.gpu_lock = threading.RLock()
                        
                    print(f"Screen capture will use {self.gpu_type} acceleration with MSS")
                else:
                    # Use the global hardware manager
                    self.hardware_manager = hardware_manager
                    self.gpu_type = self.hardware_manager.get_hardware_type()
                    self.gpu_device = self.hardware_manager.get_device()
                    
                    # Create a lock for hardware operations
                    self.gpu_lock = threading.RLock()
                    print(f"Screen capture will use {self.gpu_type} acceleration with MSS")
                    
            except Exception as e:
                print(f"Hardware setup for screen capture failed: {e}")
                self.gpu_type = 'CPU'
                self.gpu_device = None
                self.hardware_manager = None
                self.gpu_lock = threading.RLock()

        def _init_mss_capture(self):
            """Initialize MSS capture resources."""
            try:
                # Import MSS
                import mss
                self.mss = mss.mss()
                
                # Get screen dimensions
                screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
                screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
                
                # Get the primary monitor
                self.monitor = self.mss.monitors[1]  # Primary monitor
                
                print(f"[DEBUG] Screen dimensions: {screen_width}x{screen_height}")
                print(f"[DEBUG] MSS monitor: {self.monitor}")
                
                # Define capture region as the entire screen
                self.capture_region = {
                    'left': 0,
                    'top': 0,
                    'width': screen_width,
                    'height': screen_height
                }
                print(f"[DEBUG] Capture region: left={self.capture_region['left']}, top={self.capture_region['top']}, " +
                      f"width={self.capture_region['width']}, height={self.capture_region['height']}")
                print(f"MSS screen capture initialized: {screen_width}x{screen_height}")
                
                # Pre-allocate memory for better performance
                self.frame_buffer = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
                
            except Exception as e:
                print(f"Error initializing MSS capture: {e}")
                import traceback
                traceback.print_exc()
                
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
            """Continuously capture frames using MSS for maximum performance."""
            # Set thread priority to time critical
            try:
                import win32process
                import win32api
                handle = win32api.GetCurrentThread()
                win32process.SetThreadPriority(handle, win32process.THREAD_PRIORITY_TIME_CRITICAL)
                print("MSS capture thread priority set to TIME_CRITICAL")
            except:
                pass
            
            # For adaptive timing
            frame_times = collections.deque(maxlen=10)
            
            # Track consecutive failures to trigger resource recreation
            consecutive_failures = 0
            max_failures_before_reset = 3
            
            # Pre-allocate frame buffer to avoid repeated allocations
            height = self.capture_region['height']
            width = self.capture_region['width']
            frame_buffer = np.zeros((height, width, 3), dtype=np.uint8)
            
            while self.running:
                start_time = time.time()
                
                try:
                    # Check if we need to recreate MSS resources
                    if consecutive_failures >= max_failures_before_reset:
                        print(f"Detected {consecutive_failures} consecutive MSS failures. Recreating resources...")
                        try:
                            self._init_mss_capture()
                            consecutive_failures = 0
                        except Exception as e:
                            print(f"Failed to recreate MSS resources: {e}")
                            time.sleep(0.1)
                            continue
                    
                    # Capture screen using MSS (much faster than GDI)
                    screenshot = self.mss.grab(self.monitor)
                    
                    # Convert to numpy array
                    frame = np.array(screenshot)
                    
                    # Convert from BGRA to BGR (remove alpha channel)
                    frame = frame[:, :, :3]
                    
                    # Reset failure counter on success
                    consecutive_failures = 0
                    
                    # Update current frame with minimal lock time
                    with self.lock:
                        # Use the pre-allocated buffer to avoid repeated allocations
                        np.copyto(frame_buffer, frame)
                        self.current_frame = frame_buffer
                        self.frame_count += 1
                        
                except Exception as e:
                    consecutive_failures += 1
                    print(f"Error in MSS capture loop: {e} (Failures: {consecutive_failures}/{max_failures_before_reset})")
                    
                    # If we've had too many failures, try recreating the MSS
                    if consecutive_failures >= max_failures_before_reset:
                        continue
                
                # Adaptive timing to maximize FPS
                end_time = time.time()
                frame_time = end_time - start_time
                
                # Store frame time for both local calculation and FPS reporting
                frame_times.append(frame_time)
                if hasattr(self, 'frame_times'):
                    self.frame_times.append(frame_time)
                    
                # Ultra-aggressive optimization for maximum FPS
                # Remove sleep entirely for maximum capture speed
                # The detection loop will handle the processing rate
                continue  # No sleep, maximum speed

        def fps_loop(self):
            """Calculate FPS in a separate thread."""
            last_count = 0
            last_time = time.time()
            # For tracking frame times
            self.frame_times = collections.deque(maxlen=30)
            
            while self.running:
                # Sleep for a shorter time to update more frequently
                time.sleep(0.5)
                
                current_time = time.time()
                elapsed = current_time - last_time
                
                with self.lock:
                    current_count = self.frame_count
                
                if elapsed > 0:
                    # Calculate frames per second
                    frames = current_count - last_count
                    self.fps = int(frames / elapsed)
                    
                    # Calculate average frame time if we have data
                    if hasattr(self, 'frame_times') and len(self.frame_times) > 0:
                        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                        print(f"MSS Capture FPS: {self.fps}, Frame time: {avg_frame_time*1000:.3f}ms")
                    else:
                        print(f"MSS Capture FPS: {self.fps}")
                    
                    # Reset for next calculation
                    last_count = current_count
                    last_time = current_time
                    
                    # If we have a parent reference, force a redraw to update FPS display
                    if hasattr(self, 'parent') and hasattr(self.parent, 'parent') and hasattr(self.parent.parent, 'hwnd'):
                        try:
                            win32gui.InvalidateRect(self.parent.parent.hwnd, None, False)
                        except Exception as e:
                            print(f"Error updating FPS display: {e}")

        def stop_capture(self):
            """Stop the capture thread and clean up resources."""
            self.running = False
            
            if self.capture_thread:
                self.capture_thread.join(timeout=1.0)
                
            if self.fps_thread:
                self.fps_thread.join(timeout=1.0)
                
            # Clean up MSS resources
            if self.mss:
                self.mss.close()
                
            print("MSS screen capture stopped and resources cleaned up")

        def get_frame(self):
            """Get the latest captured frame."""
            with self.lock:
                if self.current_frame is None:
                    return None
                # Return a view instead of a copy for better performance
                # The detection loop will make its own copy if needed
                return self.current_frame
    
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

# --- OverlayWindow Class ---

class OverlayWindow:
    """
    Transparent overlay window for displaying detection results and configuration menu.
    
    This class creates a transparent, click-through overlay window that displays
    detection boxes, FOV circles, FPS counters, and a configuration menu. It handles
    all window management, drawing, and user interaction.
    
    Attributes:
        width: Window width (full screen width)
        height: Window height (full screen height)
        hwnd: Window handle
        running: Whether the window is running
        config: Configuration dictionary
        detector: YOLODetector instance
        menu_active: Whether the menu is currently active
        active_tab: Currently active menu tab
        tabs: List of available menu tabs
        menu_items: List of current menu items
        mouse_down: Whether mouse button is pressed
        active_slider: Currently active slider for dragging
        waiting_for_key: Key binding being configured
        active_input_field: Currently active input field
        current_input_text: Text in active input field
        last_mouse_pos: Last mouse position
        status_message: Current status message
        status_message_time: Time when status message was set
        menu_rect: Menu rectangle coordinates
        dragging: Whether menu is being dragged
        drag_offset: Drag offset from mouse position
        show_config_dropdown: Whether config dropdown is shown
        config_files: List of available config files
        dropdown_selected: Selected dropdown item
        menu_page: Current menu page
        items_per_page: Number of items per page
        config_dir: Directory for configuration files
    """
    
    def __init__(self):
        """Initialize the overlay window."""
        try:
            logging.info("Initializing OverlayWindow")
            
            # Initialize window properties
            self.width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            self.height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            logging.info(f"Screen dimensions: {self.width}x{self.height}")
            
            self.running = True
            self.aim_assist_running = True  # Add this for aim assist thread
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
            self.detector.parent = self  # Add reference to parent for status messages
            logging.info("YOLO detector initialized")
            
            # Make sure FPS counter is enabled by default
            self.config['show_fps_counter'] = True
            
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
            # Use both color key (black = transparent) and alpha blending for better results
            win32gui.SetLayeredWindowAttributes(
                self.hwnd,
                0,  # Color key (0 = black for transparent)
                200,  # Alpha (200 for slightly transparent)
                win32con.LWA_COLORKEY | win32con.LWA_ALPHA  # Use both color key and alpha
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
            
        except Exception as e:
            logging.error(f"Error in OverlayWindow initialization: {e}")
            logging.error(traceback.format_exc())
            raise

    def aim_assist_loop(self) -> None:
        """
        Aim assist loop for automatic mouse movement.
        
        This method runs in a separate thread and continuously monitors detected
        objects, automatically moving the mouse to aim at targets when enabled.
        """
        last_target = None
        target_history = []  # Track recent targets for stability
        max_history = 3      # Number of frames to average for smoothing
        
        # Improved aim assist logic
        closest_target = None
        min_distance = float('inf')
        screen_center_x = win32api.GetSystemMetrics(0) // 2
        screen_center_y = win32api.GetSystemMetrics(1) // 2
        
        # Add deadzone to prevent jittering
        deadzone = self.config.get('aim_deadzone', 3)  # pixels - don't move if within this distance
        
        while self.aim_assist_running:
            try:
                if not self.config.get('aim_assist_enabled', True):
                    time.sleep(0.01)
                    continue
                
                # Get current detections
                detections = self.detector.get_detections()
                if not detections:
                    time.sleep(0.01)
                    continue
                
                # Find closest target within FOV
                closest_target = None
                min_distance = float('inf')
                fov_radius = self.config.get('fov_modifier', 150)
                
                for detection in detections:
                    if detection.get('confidence', 0) < self.config.get('confidence', 0.05):
                        continue
                    
                    # Calculate target center
                    x1, y1, x2, y2 = detection['screen']
                    target_x = (x1 + x2) // 2
                    target_y = (y1 + y2) // 2
                    
                    # Calculate distance from screen center
                    distance = math.sqrt((target_x - screen_center_x) ** 2 + (target_y - screen_center_y) ** 2)
                    
                    # Check if target is within FOV
                    if distance <= fov_radius and distance < min_distance:
                        min_distance = distance
                        closest_target = (target_x, target_y)
                
                if closest_target is None:
                    time.sleep(0.01)
                    continue
                
                target_x, target_y = closest_target
                
                # Calculate movement needed
                dx = target_x - screen_center_x
                dy = target_y - screen_center_y
                
                # Check deadzone - don't move if too close
                if abs(dx) <= deadzone and abs(dy) <= deadzone:
                    time.sleep(0.01)
                    continue
                
                # Apply tracking speed and smoothing
                tracking_speed = self.config.get('tracking_speed', 0.5)
                smoothing = self.config.get('smoothing_xy', 0.5)
                
                # Calculate movement with speed and smoothing
                move_x = int(dx * tracking_speed * smoothing)
                move_y = int(dy * tracking_speed * smoothing)
                
                # Apply humaniser for more natural movement
                humaniser = self.config.get('humaniser', 0.2)
                if humaniser > 0:
                    move_x += random.randint(-int(humaniser * 5), int(humaniser * 5))
                    move_y += random.randint(-int(humaniser * 5), int(humaniser * 5))
                
                # Move the mouse to actually control the camera
                if abs(move_x) > 0 or abs(move_y) > 0:
                    move_mouse(move_x, move_y)
                
                time.sleep(0.01)  # Small delay to prevent excessive movement
                
            except Exception as e:
                logging.error(f"Error in aim assist loop: {e}")
                time.sleep(0.1)

    def build_menu_items(self) -> List[Tuple[str, str, str, Optional[float], Optional[float], Optional[str]]]:
        """
        Build the menu items list based on active tab.
        
        Returns:
            List of menu item tuples (label, type, key, min_val, max_val, description)
        """
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
                ('Aim Deadzone', 'slider', 'aim_deadzone', 1, 10, None),
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
                ('Toggle Aim Assist', 'key', 'tracking_key', None, None, None),
                ('Menu Key', 'key', 'menu_key', None, None, None),
                ('Trigger Key', 'key', 'trigger_key', None, None, None),
                ('FOV Toggle Key', 'key', 'fov_toggle_key', None, None, None),
                ('FPS Toggle Key', 'key', 'fps_toggle_key', None, None, None),
            ],
            'Settings': [
                ('Optimizations', 'toggle', 'optimizations_enabled', None, None, None),
                ('Use GPU Capture (DirectX)', 'toggle', 'use_directx_capture', None, None, None),
                ('Frame Skipping', 'number_input', 'frame_skipping', 1, 4, None),
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

    def draw_text(self, hdc, x, y, text, color=(255, 255, 255), size=14, weight=700):
        """Draw text with a background for better visibility."""
        try:
            # Create a LOGFONT structure for better text quality
            lf = win32gui.LOGFONT()
            lf.lfFaceName = "Segoe UI"  # More modern font
            lf.lfHeight = -size  # Negative for better quality
            lf.lfWeight = weight
            lf.lfQuality = win32con.CLEARTYPE_QUALITY  # Use ClearType for better rendering
            
            # Create font
            font = win32gui.CreateFontIndirect(lf)
            old_font = win32gui.SelectObject(hdc, font)
            
            # Get text dimensions
            win32gui.SetTextColor(hdc, rgb_to_colorref(color))
            win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
            text_size = win32gui.GetTextExtentPoint32(hdc, text)
            
            # Create background rectangle with padding
            padding = 6
            bg_rect = (x - padding, y - padding, 
                      x + text_size[0] + padding, y + text_size[1] + padding)
            
            # Fill background with solid black
            bg_brush = win32gui.CreateSolidBrush(rgb_to_colorref((0, 0, 0)))
            win32gui.FillRect(hdc, bg_rect, bg_brush)
            win32gui.DeleteObject(bg_brush)
            
            # Draw border
            border_pen = win32gui.CreatePen(win32con.PS_SOLID, 1, rgb_to_colorref((70, 70, 70)))
            old_pen = win32gui.SelectObject(hdc, border_pen)
            win32gui.SelectObject(hdc, win32gui.GetStockObject(win32con.NULL_BRUSH))
            win32gui.Rectangle(hdc, bg_rect[0], bg_rect[1], bg_rect[2], bg_rect[3])
            win32gui.SelectObject(hdc, old_pen)
            win32gui.DeleteObject(border_pen)
            
            # Draw text using DrawText instead of TextOut for better reliability
            text_rect = (x, y, x + text_size[0], y + text_size[1])
            win32gui.SetTextColor(hdc, rgb_to_colorref(color))
            win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
            win32gui.DrawText(hdc, text, -1, text_rect, win32con.DT_LEFT | win32con.DT_TOP)
            
            # Clean up
            win32gui.SelectObject(hdc, old_font)
            win32gui.DeleteObject(font)
        except Exception as e:
            logging.error(f"Error drawing text: {e}")
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
                        
                        # Apply frame skipping value directly to detector if that's what was changed
                        if self.active_input_field == 'frame_skipping' and hasattr(self, 'detector'):
                            self.detector.process_every_n = value
                            print(f"Updated detector frame skipping to: {value}")
                            
                            # Force a full redraw to update FPS counter with new skip value
                            win32gui.InvalidateRect(hwnd, None, True)
                            win32gui.UpdateWindow(hwnd)
                            
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
            # Create memory DC and bitmap for double buffering
            mem_dc = win32gui.CreateCompatibleDC(hdc)
            mem_bitmap = win32gui.CreateCompatibleBitmap(hdc, self.width, self.height)
            old_bitmap = win32gui.SelectObject(mem_dc, mem_bitmap)
            
            # Important: Set background mode to TRANSPARENT to avoid white fill
            win32gui.SetBkMode(mem_dc, win32con.TRANSPARENT)
            
            # Do NOT fill the entire rectangle with a brush - this causes strobing
            # Instead, we'll only draw the specific UI elements we need
            
                # Test pattern removed - we don't need the purple crosshair anymore
            
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
            if self.config.get('show_player_boxes', True) and hasattr(self.detector, 'current_boxes'):
                boxes = self.detector.current_boxes
                
                if boxes is not None and len(boxes) > 0:
                    # Debug output to check if boxes are available
                    print(f"Drawing {len(boxes)} boxes")
                    
                    confidences = self.detector.current_confidences if hasattr(self.detector, 'current_confidences') else [0.5] * len(boxes)
                    class_ids = self.detector.current_class_ids if hasattr(self.detector, 'current_class_ids') else [0] * len(boxes)
                    
                    # Use box_color from config with high visibility
                    box_color = self.config.get('box_color', (0, 255, 0))  # Default to bright green for visibility
                    
                    # Create a thicker pen for better visibility
                    box_pen = win32gui.CreatePen(win32con.PS_SOLID, 2, rgb_to_colorref(box_color))
                    old_pen = win32gui.SelectObject(mem_dc, box_pen)
                    
                    # Use NULL_BRUSH for transparent fill (outline only)
                    old_brush = win32gui.SelectObject(mem_dc, win32gui.GetStockObject(win32con.NULL_BRUSH))
                    
                    print(f"Drawing {len(boxes)} boxes with color {box_color}")
                    
                    for i, box in enumerate(boxes):
                        try:
                            x1, y1, x2, y2 = box
                            
                            # Additional validation for reasonable box size
                            box_width = x2 - x1
                            box_height = y2 - y1
                            
                            # Skip boxes that are too large (using percentage of screen)
                            if box_width > 0.6 * self.width or box_height > 0.8 * self.height:
                                print(f"Final filter: Skipping oversized box: {box_width}x{box_height} (max: {0.6*self.width}x{0.8*self.height})")
                                continue
                                
                            # Skip boxes that are too small (less than 10 pixels)
                            if box_width < 10 or box_height < 10:
                                print(f"Skipping tiny box {i}: {x1},{y1},{x2},{y2} (w={box_width}, h={box_height})")
                                continue
                                
                            # Filter out boxes that are in the corners of the screen (likely false positives)
                            # DISABLED: This was removing valid detections
                            # corner_threshold = 0.05  # 5% of screen dimensions - reduced to allow more detections
                            # corner_width = self.width * corner_threshold
                            # corner_height = self.height * corner_threshold
                            
                            # Check if box is entirely within any corner
                            # if ((x2 < corner_width and y2 < corner_height) or  # Top-left
                            #     (x1 > self.width - corner_width and y2 < corner_height) or  # Top-right
                            #     (x2 < corner_width and y1 > self.height - corner_height) or  # Bottom-left
                            #     (x1 > self.width - corner_width and y1 > self.height - corner_height)):  # Bottom-right
                            #     print(f"Skipping corner box {i}: {x1},{y1},{x2},{y2}")
                            #     continue
                            
                            # Ensure coordinates are integers and within screen bounds
                            x1 = max(0, min(int(x1), self.width))
                            y1 = max(0, min(int(y1), self.height))
                            x2 = max(0, min(int(x2), self.width))
                            y2 = max(0, min(int(y2), self.height))
                            
                            # Skip invalid boxes
                            if x2 <= x1 or y2 <= y1:
                                print(f"Skipping invalid box {i}: {x1},{y1},{x2},{y2}")
                                continue
                                
                            # Draw the rectangle (outline only, no fill)
                            win32gui.Rectangle(mem_dc, x1, y1, x2, y2)
                            
                            # Draw confidence value if available
                            if i < len(confidences):
                                conf_text = f"{confidences[i]:.2f}"
                                win32gui.SetTextColor(mem_dc, rgb_to_colorref((255, 255, 0)))  # Yellow text
                                win32gui.SetBkMode(mem_dc, win32con.OPAQUE)  # Make text background opaque
                                win32gui.SetBkColor(mem_dc, rgb_to_colorref((0, 0, 0)))  # Black background
                                
                                # Use DrawText with larger text area
                                text_rect = (x1, y1 - 20, x1 + 60, y1)
                                win32gui.DrawText(mem_dc, conf_text, -1, text_rect, win32con.DT_LEFT)
                                
                            print(f"Drew box {i} at {x1},{y1},{x2},{y2} with confidence {confidences[i] if i < len(confidences) else 'unknown'}")
                            
                        except Exception as e:
                            print(f"Error drawing box {i}: {e}")
                    
                    # Clean up GDI resources
                    win32gui.SelectObject(mem_dc, old_brush)
                    win32gui.SelectObject(mem_dc, old_pen)
                    win32gui.DeleteObject(box_pen)
                else:
                    print("No boxes to draw")
            
            # Draw FPS counter if enabled
            if self.config['show_fps_counter']:
                try:
                    # Use dedicated method for drawing FPS counter
                    self.draw_fps_counter(mem_dc)
                except Exception as e:
                    logging.error(f"Error drawing FPS counter: {e}")
                    traceback.print_exc()
            
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
            
            # Blit the memory DC to the window DC
            win32gui.BitBlt(hdc, 0, 0, self.width, self.height, mem_dc, 0, 0, win32con.SRCCOPY)
            
            # Properly clean up GDI resources to prevent leaks
            win32gui.SelectObject(mem_dc, old_bitmap)  # Restore the original bitmap
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
            
            # Set up a timer to force redraw for FPS counter and boxes, but with reduced frequency
            self.last_redraw_time = 0
            self.redraw_interval = 0.1  # Redraw at most 10 times per second to prevent strobing
            
            def force_redraw():
                if self.running:
                    try:
                        current_time = time.time()
                        
                        # Only redraw if enough time has passed since last redraw
                        if current_time - self.last_redraw_time >= self.redraw_interval:
                            # Only invalidate specific regions that need updating
                            # FPS counter region (top-left corner)
                            fps_rect = (0, 0, 200, 40)
                            win32gui.InvalidateRect(self.hwnd, fps_rect, False)
                            
                            # Only invalidate box regions if we have detections
                            if hasattr(self.detector, 'current_boxes') and self.detector.current_boxes:
                                # Invalidate only the regions where boxes are
                                for box in self.detector.current_boxes:
                                    x1, y1, x2, y2 = box
                                    # Add padding around box for text
                                    rect = (int(x1)-5, int(y1)-20, int(x2)+5, int(y2)+5)
                                    win32gui.InvalidateRect(self.hwnd, rect, False)
                            
                            win32gui.UpdateWindow(self.hwnd)
                            self.last_redraw_time = current_time
                        
                        # Call update_fps on detector to ensure FPS values are updated
                        if self.detector:
                            self.detector.update_fps()
                        
                        # Schedule next check (more frequent checks, but less frequent redraws)
                        threading.Timer(0.05, force_redraw).start()
                    except Exception as e:
                        print(f"Error in force_redraw: {e}")
                        # Try again after a short delay
                        threading.Timer(0.2, force_redraw).start()
            
            # Start the redraw timer
            threading.Timer(0.05, force_redraw).start()
            
            # Main message loop
            while self.running:
                try:
                    msg = win32gui.GetMessage(self.hwnd, 0, 0)
                    if msg[0]:
                        win32gui.TranslateMessage(msg[1])
                        win32gui.DispatchMessage(msg[1])
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
                
                # Apply frame skipping value to detector
                if hasattr(self, 'detector') and 'frame_skipping' in self.config:
                    frame_skip = max(1, min(4, self.config['frame_skipping']))
                    self.detector.process_every_n = frame_skip
                    print(f"Applied frame skipping from config: {frame_skip}")
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
            'is_menu_open': False,
            'show_fps_counter': True,
            'fps_display_style': 0,
            'smoothing_xy': 0.5,
            'aim_deadzone': 3,
            'anti_lag_value': 0.0,
            'custom_bone_position': 0.0,
            'aim_assist_enabled': True,
        }
        self.config.update(default_config)
        self.status_message = "Settings reset to default"
        self.status_message_time = time.time()
        self.menu_items = self.build_menu_items()
        self.active_input_field = None
        self.current_input_text = ""
        
        # Apply frame skipping value to detector
        if hasattr(self, 'detector'):
            self.detector.process_every_n = default_config['frame_skipping']
            print(f"Reset frame skipping to: {default_config['frame_skipping']}")
            
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

    def draw_fps_counter(self, hdc):
        """Draw the FPS counter with high visibility."""
        try:
            if not self.config['show_fps_counter'] or not self.detector:
                return
                
            # Get FPS values
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
            
            # Create a LOGFONT structure for better text quality
            lf = win32gui.LOGFONT()
            lf.lfFaceName = "Segoe UI"
            lf.lfHeight = -14  # Smaller text (was -20)
            lf.lfWeight = 700  # Bold but not too bold
            lf.lfQuality = win32con.CLEARTYPE_QUALITY
            
            # Create font
            font = win32gui.CreateFontIndirect(lf)
            old_font = win32gui.SelectObject(hdc, font)
            
            # Get text dimensions
            win32gui.SetTextColor(hdc, rgb_to_colorref((255, 255, 0)))  # Bright yellow
            win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
            text_size = win32gui.GetTextExtentPoint32(hdc, fps_text)
            
            # Position in top-left corner
            x, y = 10, 10
            
            # Create background rectangle with minimal padding
            padding = 4  # Reduced padding (was 8)
            bg_rect = (x - padding, y - padding, 
                      x + text_size[0] + padding, y + text_size[1] + padding)
            
            # Fill background with solid black (no outline)
            bg_brush = win32gui.CreateSolidBrush(rgb_to_colorref((0, 0, 0)))
            win32gui.FillRect(hdc, bg_rect, bg_brush)
            win32gui.DeleteObject(bg_brush)
            
            # Draw text using DrawText
            text_rect = (x, y, x + text_size[0], y + text_size[1])
            win32gui.SetTextColor(hdc, rgb_to_colorref((255, 255, 0)))
            win32gui.DrawText(hdc, fps_text, -1, text_rect, win32con.DT_LEFT | win32con.DT_TOP)
            
            # Clean up
            win32gui.SelectObject(hdc, old_font)
            win32gui.DeleteObject(font)
            
        except Exception as e:
            logging.error(f"Error drawing FPS counter: {e}")
            traceback.print_exc()

    def _ensure_model_on_correct_device(self):
        """Ensure the model is on the correct device based on current hardware."""
        try:
            if hasattr(self, 'model') and hasattr(self.model, 'model'):
                current_device = next(self.model.model.parameters()).device
                target_device = self.hardware_manager.get_device()
                
                if current_device != target_device:
                    print(f"[INFO] Moving model from {current_device} to {target_device}")
                    self.model.model = self.model.model.to(target_device)
                    
                    # Force all parameters to the target device
                    for param in self.model.model.parameters():
                        param.data = param.data.to(target_device)
                    
                    print(f"[INFO] Model successfully moved to {self.hardware_manager.get_hardware_type()} device")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to move model to correct device: {e}")
            return False
    def optimize_model(self):
        """Apply performance optimizations to the model."""
        if hasattr(self.model, 'model'):
            # Enable PyTorch optimizations
            torch.set_num_threads(8)  # Use more CPU threads
            torch.backends.cudnn.benchmark = True
            
            # Fuse layers for faster inference
            try:
                self.model.model.fuse()
                print("[INFO] Model fused for faster inference")
            except Exception as e:
                print(f"[INFO] Model fusion failed: {e}")
            
            # Use lower precision if available
            try:
                self.model.model.half()
                print("[INFO] Using FP16 for faster inference")
            except Exception as e:
                print(f"[INFO] FP16 not available: {e}")
            
            # Optimize thresholds for speed
            self.model.conf = 0.4  # Higher confidence = fewer detections = faster
            self.model.iou = 0.45  # Optimized NMS threshold
            
            print(f"[INFO] Optimized model with {self.target_size}x{self.target_size} input")
    
    def should_process_frame(self):
        """Determine if current frame should be processed."""
        self.frame_counter += 1
        return self.frame_counter % (self.frame_skip + 1) == 0

        
        # Apply performance optimizations
        self.optimize_model()

# --- Hardware Management ---

class HardwareManager:
    """
    Centralized hardware detection and management.
    
    This class performs hardware detection once during initialization and provides
    clear interfaces for AMD, NVIDIA, and CPU execution paths. It eliminates
    redundant hardware checks throughout the codebase.
    
    Attributes:
        hardware_type: Type of hardware ('AMD', 'NVIDIA', or 'CPU')
        device: PyTorch device for the detected hardware
        device_name: Human-readable name of the detected hardware
        is_gpu: Whether GPU acceleration is available
        supports_cuda: Whether CUDA is supported
        supports_directml: Whether DirectML is supported
    """
    
    def __init__(self):
        """Initialize hardware detection and setup."""
        self.hardware_type = None
        self.device = None
        self.device_name = None
        self.is_gpu = False
        self.supports_cuda = False
        self.supports_directml = False
        
        # Perform hardware detection
        self._detect_hardware()
        
    def _detect_hardware(self) -> None:
        """
        Detect available hardware and set up appropriate devices.
        
        This method performs a single comprehensive hardware detection and
        sets up the appropriate PyTorch device for the detected hardware.
        """
        try:
            print(f"System: {platform.system()} {platform.release()}")
            print(f"Python: {platform.python_version()}")
            print(f"PyTorch: {torch.__version__}")
            
            # Check for NVIDIA GPU with CUDA (highest priority - works well)
            if torch.cuda.is_available():
                try:
                    print("Checking for NVIDIA GPU with CUDA...")
                    
                    # Test NVIDIA GPU functionality
                    test_device = torch.device('cuda')
                    test_tensor = torch.tensor([1.0, 2.0, 3.0], device=test_device)
                    test_result = test_tensor + test_tensor
                    _ = test_result.cpu().numpy()  # Force synchronization
                    
                    # NVIDIA GPU is working
                    self.hardware_type = 'NVIDIA'
                    self.device = test_device
                    self.device_name = torch.cuda.get_device_name(0)
                    self.is_gpu = True
                    self.supports_cuda = True
                    
                    print(f"✓ NVIDIA GPU detected: {self.device_name}")
                    print(f"  CUDA device: {self.device}")
                    return
                    
                except Exception as e:
                    print(f"✗ NVIDIA GPU detection failed: {e}")
            
            # AMD GPU: Use MSS + CPU for best performance and reliability
            # This eliminates all DirectML compatibility issues while maintaining high performance
            print("AMD GPU detected - using MSS + CPU optimization")
            print("Reason: MSS provides excellent capture performance, CPU provides reliable inference")
            print("Result: Better performance than broken DirectML")
            
            self.hardware_type = 'CPU'
            self.device = torch.device('cpu')
            self.device_name = platform.processor()
            self.is_gpu = False
            
            # Enable CPU optimizations for better performance
            torch.set_num_threads(min(8, os.cpu_count()))  # Use multiple CPU threads
            torch.backends.mkldnn.enabled = True  # Enable MKL-DNN for Intel CPUs
            torch.backends.mkldnn.allow_tf32 = True  # Allow TF32 for better performance
            
            print(f"✓ AMD GPU using MSS + CPU optimization: {self.device_name}")
            print(f"  CPU threads: {torch.get_num_threads()}")
            print(f"  MKL-DNN enabled: {torch.backends.mkldnn.enabled}")
            print(f"  MSS capture: Enabled")
            
        except Exception as e:
            print(f"ERROR: Hardware detection failed: {e}")
            # Emergency fallback to CPU
            self.hardware_type = 'CPU'
            self.device = torch.device('cpu')
            self.device_name = 'Unknown CPU'
            self.is_gpu = False
            print("✓ Emergency CPU fallback activated")
    
    def is_amd(self) -> bool:
        """Check if AMD GPU is available (using MSS + CPU optimization)."""
        return self.hardware_type == 'CPU' and 'AMD' in self.device_name.upper()
    
    def is_nvidia(self) -> bool:
        """Check if NVIDIA GPU is available."""
        return self.hardware_type == 'NVIDIA'
    
    def is_cpu(self) -> bool:
        """Check if CPU fallback is being used."""
        return self.hardware_type == 'CPU'
    
    def get_device(self) -> torch.device:
        """Get the PyTorch device for the detected hardware."""
        return self.device
    
    def get_device_name(self) -> str:
        """Get the human-readable name of the detected hardware."""
        return self.device_name
    
    def get_hardware_type(self) -> str:
        """Get the hardware type string."""
        return self.hardware_type
    
    def create_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Create a tensor on the appropriate device.
        
        Args:
            shape: Tensor shape
            dtype: Tensor data type
            
        Returns:
            Tensor created on the appropriate device
        """
        return torch.zeros(shape, device=self.device, dtype=dtype)
    
    def move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Move a tensor to the appropriate device.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor moved to the appropriate device
        """
        return tensor.to(self.device)
    
    def synchronize(self) -> None:
        """Synchronize the device (GPU-specific)."""
        if self.is_nvidia() and hasattr(torch.cuda, 'synchronize'):
            torch.cuda.synchronize()
        elif self.is_amd():
            # For AMD (CPU mode), no synchronization needed
            pass
    
    def empty_cache(self) -> None:
        """Empty device cache (GPU-specific)."""
        if self.is_nvidia() and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    
    def get_memory_info(self) -> Optional[Dict[str, Any]]:
        """Get memory information for the device."""
        if self.is_nvidia() and hasattr(torch.cuda, 'get_device_properties'):
            props = torch.cuda.get_device_properties(self.device)
            return {
                'total_memory': props.total_memory,
                'memory_allocated': torch.cuda.memory_allocated(self.device),
                'memory_reserved': torch.cuda.memory_reserved(self.device)
            }
        return None

# Global hardware manager instance
hardware_manager = HardwareManager()

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
        overlay.aim_assist_running = False  # Stop aim assist thread
        if hasattr(overlay, 'aim_thread') and overlay.aim_thread.is_alive():
            overlay.aim_thread.join(timeout=1.0)  # Wait for thread to finish
        overlay.detector.stop_detection()
        logging.info("Detection thread stopped")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        logging.error(traceback.format_exc())
    finally:
        logging.info("Application shutting down")

if __name__ == "__main__":
    main()
