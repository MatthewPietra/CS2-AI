# FuryAI Powered by YOLOv8

---

### Project Overview

FuryAi is a Python-based application leveraging a YOLOv8-trained model for real-time object detection, especially aimed at detecting players for aim assistance functionalities in Counter Strike 2. The project is designed for personal development in the aim of trying to fully understand how ai based aimbots work and then eventually how to counteract them in the futire.

The system detects targets using an accurate YOLOv8 model and integrates with Windows APIs to provide overlay visuals and assistive aim features such as aim assist toggling, trigger bot, and configurable field-of-view (FOV) visualization.

---

### Features

- **Real-time Object Detection**: Uses a YOLOv8 player-trained model (`best.pt`) to detect players smoothly.
- **GPU Acceleration**: Automatically detects and uses GPU (CUDA, DirectML for AMD, or MPS for Apple) when available; otherwise, falls back to CPU.
- **Overlay Visualization**:
  - Bounding boxes around detected players.
  - Optional FOV circle visualizer.
  - FPS counter display.
- **Configurable Aim Assist**: 
  - Adjustable tracking speed, smoothing, humanizer, and FOV modifier.
  - Toggle aim assist and trigger bot via configurable hotkeys.
- **Robust UI**:
  - Win32-based custom overlay with transparent and click-through window support.
  - Responsive, tabbed menu for configuring all settings including colors, toggles, sliders, and keybinds.
  - Color picker integration for customizing UI elements.
- **Config Management**:
  - Load and save configurations to JSON files.
  - Reset settings to smart defaults.
- **Multi-threaded Detection Loop**: Efficient frame processing and detection running asynchronously to keep UI smooth.
- **Extensive Logging & Debugging**: Logs important events and errors; configurable debug settings.
- **Compatibility**:
  - Support for Windows OS.
  - Works with multiple GPU vendors via appropriate libraries.

---

### System Requirements

- **Operating System**: Windows 10 or higher.
- **Python Version**: Python 3.8+ recommended.
- **Dependencies**:
  - PyTorch (compatible version matching CUDA or DirectML support)
  - `ultralytics` library for YOLOv8
  - `torch_directml` for AMD GPU acceleration (optional)
  - `numpy`, `opencv-python`
  - `Pillow` for frame capture and image processing
  - `pywin32` for Windows API access
  - `wmi` for GPU detection on Windows
  - `tkinter` (comes with Python standard library) for color picking dialog
- **Hardware**:
  - Recommended GPU for acceleration (NVIDIA CUDA or AMD DirectML supported)
  - CPU fallback supported but slower

---

### Installation & Setup

1. **Clone or download the project files**, including `launcher.py` and the trained YOLO model file `best.pt`.
2. **Install Python dependencies** using pip:
   ```bash
   pip install torch torchvision torchaudio ultralytics torch-directml numpy opencv-python pillow pywin32 wmi
   ```
- For AMD GPU acceleration, ensure `torch-directml` is installed.
- If using NVIDIA GPU, ensure CUDA-enabled PyTorch version is installed.   
3. **Place the YOLOv8 model file** (`best.pt`) in the same directory or configure the path in the config.
4. **Run the application** by executing:
   ```bash
   python launcher.py
   ```
5. **Control the aim assist features** via hotkeys:
   - `INSERT` - Toggle config menu visibility.
   - `F2` - Toggle aim assist.
   - `F3` - Toggle trigger bot.
   - `F4` - Toggle FOV visualizer.
   - `F5` - Toggle FPS counter.
   ---

### Configuration

- The overlay menu provides intuitive tabs for configuring:
  - **Aim Assist**: tracking speed, smoothing, humanizer, FOV size.
  - **Visuals**: colors, bounding boxes, FOV circle, FPS display.
  - **Trigger Bot**: delay and timing settings.
  - **Keybinds**: customize all control hotkeys.
  - **Settings**: enable optimizations, GPU capture options, reset defaults.
- Configuration is persisted as JSON in a config directory.
- Color pickers enable custom color selection for UI elements.
- Sliders allow fine-grained tuning of parameters.

---

### Development Notes

- The project uses Win32 API calls through `pywin32` and `ctypes` for rendering the transparent overlay window and handling input.
- The detection loop uses multi-threading and queues to balance frame capture, detection, and UI painting efficiently.
- GPU device detection supports NVIDIA CUDA, AMD DirectML, and Apple MPS with fallback to CPU.
- A custom tracker (`BYTETracker`) is integrated for object tracking between frames.
- All rendering uses GDI calls to draw elements such as FOV circle, bounding boxes, and text.
- Exception handling and logging throughout to facilitate debugging.

---

### Potential Improvements & Future Work

- Add support for more complex tracking and prediction algorithms.
- Integrate a fully cross-platform UI with modern frameworks like Qt or Electron.
- Add new AI models for different games or more object classes.
- Extend trigger bot for advanced firing conditions using confidence scores.
- Improve performance with asynchronous GPU frame capture methods.
- Support for more input devices for aim keys including mouse buttons.

---

### References

- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Win32 API for Python (pywin32)](https://github.com/mhammond/pywin32)
- [BYTETracker Algorithm](https://github.com/ifzhang/ByteTrack)
- [torch-directml for AMD GPU Acceleration](https://github.com/microsoft/DirectML)
