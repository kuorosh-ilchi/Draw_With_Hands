
import cv2
import mediapipe as mp
import numpy as np
import customtkinter as ctk
from PIL import Image
import tkinter as tk
from tkinter import messagebox
import os
import json

APP_CONFIG = {
    'camera': {
        'min_detection_confidence': 0.6,
        'min_tracking_confidence': 0.5,
        'width': 1280,
        'height': 720
    },
    'drawing': {
        'default_brush_size': 5,
        'max_undo_steps': 10
    },
    'window': {
        'width': 1600,
        'height': 800,
        'canvas_width': 1280,
        'canvas_height': 720
    },
    'gestures': {
        'smoothing_factor': 0.3,
        'finger_distance': 0.3
    }
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class SettingsPanel(ctk.CTkToplevel):
    
    def __init__(self, parent, on_save_callback):
        super().__init__(parent)
        self.title("Gesture Settings")
        self.geometry("600x700")
        self.transient(parent)
        self.grab_set()

        self.settings = self.load_settings()
        self.main_frame = ctk.CTkScrollableFrame(self, width=580, height=680)
        self.main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        ctk.CTkLabel(self.main_frame, text="Gesture Customization", font=("Helvetica", 24, "bold")).pack(pady=10)
        ctk.CTkLabel(self.main_frame, text="Customize hand gestures", font=("Helvetica", 14)).pack(pady=5)

        self.create_gesture_sections()
        self.create_circle_finger_selection()

        button_frame = ctk.CTkFrame(self.main_frame)
        button_frame.pack(fill="x", pady=10)
        ctk.CTkButton(button_frame, text="Save", command=lambda: self.save_settings(on_save_callback), width=150).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Reset", command=self.reset_settings, width=150).pack(side="right", padx=10)

    def create_gesture_sections(self):
        gestures = {
            "draw_line": {"title": "Draw Line", "default": [0, 1, 0, 0, 0]},
            "draw_rectangle": {"title": "Draw Rectangle", "default": [0, 1, 1, 0, 0]},
            "draw_rectangle(confirm)": {"title": "Draw Rectangle (confirm)", "default": [1, 1, 0, 0, 1]},
            "fill_area": {"title": "Fill Area", "default": [0, 1, 1, 1, 1]},

            "erase": {"title": "Erase", "default": [1, 0, 0, 0, 0]}
        }
        for gesture_id, gesture_info in gestures.items():
            self.create_gesture_section(gesture_id, gesture_info)

    def create_gesture_section(self, gesture_id, gesture_info):
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(padx=10, pady=5, fill="x")
        ctk.CTkLabel(frame, text=gesture_info["title"], font=("Helvetica", 16, "bold")).pack(pady=5)

        fingers_frame = ctk.CTkFrame(frame)
        fingers_frame.pack(padx=10, pady=5, fill="x")
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        finger_vars = []

        for i, name in enumerate(finger_names):
            var = tk.BooleanVar(value=self.settings.get(gesture_id, gesture_info["default"])[i])
            finger_vars.append(var)
            toggle_frame = ctk.CTkFrame(fingers_frame)
            toggle_frame.pack(side="left", padx=5, pady=5)
            ctk.CTkSwitch(toggle_frame, text="", variable=var, width=40).pack(side="left")
            ctk.CTkLabel(toggle_frame, text=name, font=("Helvetica", 12)).pack(side="left")

        self.settings[gesture_id] = finger_vars

    def create_circle_finger_selection(self):
        """ایجاد بخش انتخاب انگشت‌ها برای رسم دایره."""
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(padx=10, pady=5, fill="x")
        ctk.CTkLabel(frame, text="Circle Drawing Fingers", font=("Helvetica", 16, "bold")).pack(pady=5)
        ctk.CTkLabel(frame, text="Select two fingers to activate circle drawing", font=("Helvetica", 12)).pack(pady=5)

        fingers_frame = ctk.CTkFrame(frame)
        fingers_frame.pack(padx=10, pady=5, fill="x")
        
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        self.circle_finger_vars = [tk.StringVar(value="None"), tk.StringVar(value="None")]
        
        default_circle_fingers = self.settings.get("circle_fingers", [1, 2])
        self.circle_finger_vars[0].set(finger_names[default_circle_fingers[0]] if default_circle_fingers[0] < len(finger_names) else "None")
        self.circle_finger_vars[1].set(finger_names[default_circle_fingers[1]] if default_circle_fingers[1] < len(finger_names) else "None")

        for i in range(2):
            ctk.CTkLabel(fingers_frame, text=f"Finger {i+1}:", font=("Helvetica", 12)).pack(side="left", padx=5)
            ctk.CTkOptionMenu(
                fingers_frame,
                values=["None"] + finger_names,
                variable=self.circle_finger_vars[i]
            ).pack(side="left", padx=5)

        self.settings["circle_fingers"] = self.circle_finger_vars

    def reset_settings(self):
        if messagebox.askyesno("Reset", "Reset all gesture settings to default?"):
            self.settings = {}
            self.main_frame.destroy()
            self.main_frame = ctk.CTkScrollableFrame(self, width=580, height=680)
            self.main_frame.pack(padx=10, pady=10, fill="both", expand=True)
            self.create_gesture_sections()
            self.create_circle_finger_selection()

    def load_settings(self):
        try:
            with open('gesture_settings.json', 'r') as f:
                return json.load(f)
        except:
            return {}

    def save_settings(self, callback):
        settings = {}
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        for gesture_id, vars in self.settings.items():
            if gesture_id == "circle_fingers":
                settings[gesture_id] = [
                    finger_names.index(var.get()) if var.get() in finger_names else -1
                    for var in vars
                ]
            else:
                settings[gesture_id] = [var.get() for var in vars]
        try:
            with open('gesture_settings.json', 'w') as f:
                json.dump(settings, f)
        except:
            messagebox.showwarning("Warning", "Failed to save gesture settings.")
        callback(settings)
        self.destroy()

class HandDrawingApp:
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=10,
            min_detection_confidence=APP_CONFIG['camera']['min_detection_confidence'],
            min_tracking_confidence=APP_CONFIG['camera']['min_tracking_confidence']
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.camera_index = 0
        self.camera = None
        self.initialize_camera()

        self.canvas = np.zeros_like(self.frame, dtype=np.uint8) if hasattr(self, 'frame') else None
        self.current_color = [255, 255, 255]
        self.brush_size = APP_CONFIG['drawing']['default_brush_size']
        self.hand_data = {}
        self.shapes = []
        self.line_drawn_between_hands = False
        self.max_undo_steps = APP_CONFIG['drawing']['max_undo_steps']
        self.hand_states = {}
        self.prev_landmarks = {}

        self.color_rects = [
            {"color": [255, 0, 0], "rect": (50, 10, 150, 60)},
            {"color": [0, 255, 0], "rect": (200, 10, 300, 60)},
            {"color": [0, 0, 255], "rect": (350, 10, 450, 60)},
            {"color": [0, 255, 255], "rect": (500, 10, 600, 60)},
            {"color": [255, 255, 255], "rect": (650, 10, 750, 60)}
        ]

        self.app = ctk.CTk()
        self.app.title("Hand Drawing Studio")
        self.app.geometry(f"{APP_CONFIG['window']['width']}x{APP_CONFIG['window']['height']}")

        self.main_container = ctk.CTkFrame(self.app)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas_frame = ctk.CTkFrame(self.main_container, width=APP_CONFIG['window']['canvas_width'], height=APP_CONFIG['window']['canvas_height'])
        self.canvas_frame.pack(side=ctk.LEFT, padx=5, pady=5)
        ctk.CTkLabel(self.canvas_frame, text="Drawing Canvas", font=("Helvetica", 16, "bold")).pack(pady=5)
        self.image_label = ctk.CTkLabel(self.canvas_frame, text="")
        self.image_label.pack(padx=5, pady=5)

        self.control_frame = ctk.CTkFrame(self.main_container, width=300)
        self.control_frame.pack(side=ctk.RIGHT, padx=5, pady=5)
        ctk.CTkLabel(self.control_frame, text="Control Panel", font=("Helvetica", 16, "bold")).pack(pady=5)

        self.setup_controls()
        self.gesture_settings = self.load_gesture_settings()

    def setup_controls(self):
        camera_frame = ctk.CTkFrame(self.control_frame)
        camera_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(camera_frame, text="Select Camera:", font=("Helvetica", 12, "bold")).pack(pady=5)
        self.camera_var = tk.StringVar(value="Camera 0")
        ctk.CTkOptionMenu(camera_frame, values=[f"Camera {i}" for i in range(4)], variable=self.camera_var, command=self.change_camera).pack(pady=5)

        brush_frame = ctk.CTkFrame(self.control_frame)
        brush_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(brush_frame, text="Brush Size", font=("Helvetica", 12, "bold")).pack(pady=5)
        self.brush_slider = ctk.CTkSlider(brush_frame, from_=2, to=20, command=self.update_brush_size)
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(pady=5)
        self.brush_size_label = ctk.CTkLabel(brush_frame, text=str(self.brush_size), font=("Helvetica", 12))
        self.brush_size_label.pack(pady=2)

        action_frame = ctk.CTkFrame(self.control_frame)
        action_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(action_frame, text="Clear Canvas", command=self.clear_canvas).pack(fill="x", pady=5)
        ctk.CTkButton(action_frame, text="Undo", command=self.undo_last).pack(fill="x", pady=5)
        ctk.CTkButton(action_frame, text="Save Drawing", command=self.save_drawing).pack(fill="x", pady=5)
        ctk.CTkButton(action_frame, text="Gesture Settings", command=self.open_settings).pack(fill="x", pady=5)

    def initialize_camera(self):
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                messagebox.showerror("Camera Error", f"Failed to open camera {self.camera_index}")
                return
            self.camera.set(3, APP_CONFIG['camera']['width'])
            self.camera.set(4, APP_CONFIG['camera']['height'])
            ret, self.frame = self.camera.read()
            if not ret:
                messagebox.showerror("Camera Error", "Failed to read initial frame")
                self.camera.release()
                self.camera = None
        except Exception as e:
            messagebox.showerror("Camera Error", f"Error initializing camera: {str(e)}")
            self.camera = None

    def save_drawing(self):
        try:
            if not os.path.exists('drawings'):
                os.makedirs('drawings')
            filename = f"drawings/drawing_{len(os.listdir('drawings')) + 1}.png"
            cv2.imwrite(filename, self.canvas)
            messagebox.showinfo("Save", f"Drawing saved as {filename}")
        except:
            messagebox.showwarning("Warning", "Failed to save drawing.")

    def clear_canvas(self):
        try:
            if messagebox.askyesno("Clear", "Are you sure you want to clear the drawing?"):
                self.canvas = np.zeros_like(self.frame, dtype=np.uint8)
                self.shapes = []
                self.line_drawn_between_hands = False
                self.redraw_canvas()
        except:
            messagebox.showwarning("Warning", "Failed to clear canvas.")

    def change_camera(self, selection):
        try:
            new_index = int(selection.split()[-1])
            if new_index != self.camera_index:
                if self.camera:
                    self.camera.release()
                self.camera_index = new_index
                self.initialize_camera()
        except:
            messagebox.showwarning("Warning", "Failed to change camera.")

    def update_brush_size(self, value):
        try:
            self.brush_size = int(value)
            self.brush_size_label.configure(text=str(self.brush_size))
        except:
            pass

    def set_color(self, color):
        self.current_color = color

    def undo_last(self):
        try:
            if self.shapes:
                self.shapes.pop()
                if len(self.shapes) > self.max_undo_steps:
                    self.shapes = self.shapes[-self.max_undo_steps:]
                self.redraw_canvas()
        except:
            messagebox.showwarning("Warning", "Failed to undo last action.")

    def get_distance(self, point1, point2):
        try:
            x1, y1 = point1[1], point1[2]
            x2, y2 = point2[1], point2[2]
            return int(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        except:
            return 0

    def get_finger_base(self, finger_id):
        return {4: 5, 8: 5, 12: 9, 16: 13, 20: 17}.get(finger_id, 0)

    def get_hand_scale(self, landmarks):
        try:
            if not landmarks:
                return 0
            return self.get_distance(landmarks[0], landmarks[5])
        except:
            return 0

    def smooth_landmarks(self, current_landmarks, hand_idx):
        try:
            if hand_idx not in self.prev_landmarks:
                self.prev_landmarks[hand_idx] = current_landmarks
                return current_landmarks
            smoothed = []
            for curr, prev in zip(current_landmarks, self.prev_landmarks[hand_idx]):
                x = int(curr[1] * (1 - APP_CONFIG['gestures']['smoothing_factor']) + prev[1] * APP_CONFIG['gestures']['smoothing_factor'])
                y = int(curr[2] * (1 - APP_CONFIG['gestures']['smoothing_factor']) + prev[2] * APP_CONFIG['gestures']['smoothing_factor'])
                smoothed.append([curr[0], x, y])
            self.prev_landmarks[hand_idx] = smoothed
            return smoothed
        except:
            return current_landmarks

    def detect_fingers(self, landmarks, handedness=None, is_mirrored=True):
        try:
            if not landmarks or len(landmarks) < 21:
                print("Error: Insufficient landmarks")
                return [], 0

            finger_tips = [4, 8, 12, 16, 20]
            finger_bases = [2, 5, 9, 13, 17]

            wrist = np.array(landmarks[0][1:3])
            index_mcp = np.array(landmarks[5][1:3])
            hand_scale = np.linalg.norm(index_mcp - wrist)
            if hand_scale == 0:
                print("Error: Hand scale is zero")
                return [], 0

            is_right_hand = True if handedness is None or handedness == 'Right' else False

            finger_states = []

            for tip_id, base_id in zip(finger_tips, finger_bases):
                tip_x, tip_y = landmarks[tip_id][1], landmarks[tip_id][2]
                base_x, base_y = landmarks[base_id][1], landmarks[base_id][2]

                is_thumb = tip_id == 4

                if is_thumb:
                    thumb_vector = np.array([tip_x - base_x, tip_y - base_y])
                    distance = np.linalg.norm(thumb_vector) / hand_scale
                    if is_right_hand:
                        is_open = thumb_vector[0] < 0 if is_mirrored else thumb_vector[0] > 0
                    else:
                        is_open = thumb_vector[0] > 0 if is_mirrored else thumb_vector[0] < 0
                    is_open = is_open and distance > 0.4
                else:
                    threshold = hand_scale * 0.2
                    is_open = tip_y < base_y - threshold

                finger_states.append(1 if is_open else 0)

            return finger_states, sum(finger_states)

        except Exception as e:
            print(f"Error: {e}")
            return [], 0

    def get_landmarks(self, rgb_frame, hand_landmarks):
        try:
            if not hand_landmarks:
                return []
            height, width, _ = rgb_frame.shape
            return [[idx, int(lm.x * width), int(lm.y * height)] for idx, lm in enumerate(hand_landmarks.landmark)]
        except:
            return []

    def draw_shape(self, shape_type, x, y, size, hand_idx, preview=False):
        try:
            target = self.preview_frame if preview else self.canvas
            if shape_type == "line" and self.hand_data.get(hand_idx, {}).get("prev_x"):
                prev_x, prev_y = self.hand_data[hand_idx]["prev_x"], self.hand_data[hand_idx]["prev_y"]
                temp_line = np.zeros_like(self.canvas, dtype=np.uint8)
                cv2.line(temp_line, (prev_x, prev_y), (x, y), self.current_color, self.brush_size)
                if not preview:
                    self.canvas = cv2.add(self.canvas, temp_line)
                    self.shapes.append({"type": "line", "start": (prev_x, prev_y), "end": (x, y), "color": self.current_color.copy(), "size": self.brush_size})
                else:
                    cv2.line(target, (prev_x, prev_y), (x, y), self.current_color, self.brush_size)
            elif shape_type == "circle":
                cv2.circle(target, (x, y), size, self.current_color, self.brush_size)
                if not preview:
                    self.shapes.append({"type": "circle", "center": (x, y), "radius": size, "color": self.current_color.copy()})
            elif shape_type == "rectangle" and self.hand_data.get(hand_idx, {}).get("thumb_x"):
                thumb_x, thumb_y = self.hand_data[hand_idx]["thumb_x"], self.hand_data[hand_idx]["thumb_y"]
                cv2.rectangle(target, (x, y), (thumb_x, thumb_y), self.current_color, self.brush_size)
                if not preview:
                    self.shapes.append({"type": "rectangle", "top_left": (x, y), "bottom_right": (thumb_x, thumb_y), "color": self.current_color.copy()})
        except:
            pass

    def draw_line_between_hands(self, hand1_landmarks, hand2_landmarks, hand_idx1, hand_idx2, preview=False):
        try:
            x1, y1 = hand1_landmarks[8][1], hand1_landmarks[8][2]
            x2, y2 = hand2_landmarks[8][1], hand2_landmarks[8][2]
            target = self.preview_frame if preview else self.canvas
            if preview:
                cv2.line(target, (x1, y1), (x2, y2), self.current_color, self.brush_size)
            else:
                cv2.line(self.canvas, (x1, y1), (x2, y2), self.current_color, self.brush_size)
                self.shapes.append({"type": "line", "start": (x1, y1), "end": (x2, y2), "color": self.current_color.copy(), "size": self.brush_size})
                self.line_drawn_between_hands = True
        except:
            pass

    def is_point_in_shape(self, x, y, shape):
        try:
            if shape["type"] == "line":
                start, end = shape["start"], shape["end"]
                line_vec = np.array(end) - np.array(start)
                point_vec = np.array([x, y]) - np.array(start)
                length = np.linalg.norm(line_vec)
                if length == 0:
                    return False
                t = max(0, min(1, np.dot(point_vec, line_vec) / (length ** 2)))
                projection = np.array(start) + t * line_vec
                return np.linalg.norm(np.array([x, y]) - projection) < self.brush_size
            elif shape["type"] == "circle":
                center, radius = shape["center"], shape["radius"]
                return self.get_distance([0, x, y], [0, center[0], center[1]]) <= radius
            elif shape["type"] == "rectangle":
                x1, y1 = shape["top_left"]
                x2, y2 = shape["bottom_right"]
                return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)
            elif shape["type"] == "filled_area":
                mask = np.zeros((self.canvas.shape[0] + 2, self.canvas.shape[1] + 2), dtype=np.uint8)
                temp_canvas = self.canvas.copy()
                _, _, _, rect = cv2.floodFill(temp_canvas, mask, (x, y), (255, 255, 255), loDiff=5, upDiff=5)
                return (0 <= x < self.canvas.shape[1] and 0 <= y < self.canvas.shape[0] and 
                        np.array_equal(self.canvas[y, x], shape["color"]))
            return False
        except:
            return False

    def is_point_in_rect(self, x, y, rect):
        try:
            x1, y1, x2, y2 = rect
            return x1 <= x <= x2 and y1 <= y <= y2
        except:
            return False

    def erase_area(self, thumb_x, thumb_y, radius=20):
        try:
            shapes_to_remove = []
            for i, shape in enumerate(self.shapes):
                if shape["type"] == "line":
                    start, end = shape["start"], shape["end"]
                    line_vec = np.array(end) - np.array(start)
                    point_vec = np.array([thumb_x, thumb_y]) - np.array(start)
                    length = np.linalg.norm(line_vec)
                    if length == 0:
                        continue
                    t = max(0, min(1, np.dot(point_vec, line_vec) / (length ** 2)))
                    projection = np.array(start) + t * line_vec
                    if np.linalg.norm(np.array([thumb_x, thumb_y]) - projection) < radius:
                        shapes_to_remove.append(i)
                elif shape["type"] == "circle":
                    center, r = shape["center"], shape["radius"]
                    if self.get_distance([0, thumb_x, thumb_y], [0, center[0], center[1]]) <= (r + radius):
                        shapes_to_remove.append(i)
                elif shape["type"] == "rectangle":
                    x1, y1 = shape["top_left"]
                    x2, y2 = shape["bottom_right"]
                    if min(x1, x2) - radius <= thumb_x <= max(x1, x2) + radius and min(y1, y2) - radius <= thumb_y <= max(y1, y2) + radius:
                        shapes_to_remove.append(i)
                elif shape["type"] == "filled_area":
                    if self.is_point_in_shape(thumb_x, thumb_y, shape):
                        shapes_to_remove.append(i)
            if shapes_to_remove:
                for i in sorted(shapes_to_remove, reverse=True):
                    self.shapes.pop(i)
                self.redraw_canvas()
        except:
            pass

    def fill_shape(self, shape):
        try:
            if shape["type"] == "line":
                cv2.line(self.canvas, shape["start"], shape["end"], shape["color"], self.brush_size)
            elif shape["type"] == "circle":
                cv2.circle(self.canvas, shape["center"], shape["radius"], shape["color"], -1)
            elif shape["type"] == "rectangle":
                cv2.rectangle(self.canvas, shape["top_left"], shape["bottom_right"], shape["color"], -1)
        except:
            pass

    def is_point_inside_closed_area(self, x, y):
        try:
            mask = np.zeros((self.canvas.shape[0] + 2, self.canvas.shape[1] + 2), dtype=np.uint8)
            temp_canvas = self.canvas.copy()
            _, _, _, rect = cv2.floodFill(temp_canvas, mask, (x, y), (255, 255, 255), loDiff=5, upDiff=5)
            area = rect[2] * rect[3]
            total_area = self.canvas.shape[0] * self.canvas.shape[1]
            return rect[2] > 0 and rect[3] > 0 and area < total_area * 0.8
        except:
            return False

    def fill_closed_area(self, x, y):
        try:
            if not (0 <= x < self.canvas.shape[1] and 0 <= y < self.canvas.shape[0]):
                return False
            mask = np.zeros((self.canvas.shape[0] + 2, self.canvas.shape[1] + 2), dtype=np.uint8)
            _, _, _, rect = cv2.floodFill(self.canvas, mask, (x, y), self.current_color, loDiff=5, upDiff=5)
            filled = rect[2] > 0 and rect[3] > 0
            area = rect[2] * rect[3]
            total_area = self.canvas.shape[0] * self.canvas.shape[1]
            if filled and area < total_area * 0.8:
                self.shapes.append({"type": "filled_area", "x": x, "y": y, "color": self.current_color.copy()})
                return True
            else:
                self.redraw_canvas()
                return False
        except:
            return False

    def redraw_canvas(self):
        try:
            self.canvas = np.zeros_like(self.frame, dtype=np.uint8)
            for shape in self.shapes:
                if shape["type"] == "line":
                    cv2.line(self.canvas, shape["start"], shape["end"], shape["color"], shape.get("size", self.brush_size))
                elif shape["type"] == "circle":
                    cv2.circle(self.canvas, shape["center"], shape["radius"], shape["color"], self.brush_size)
                elif shape["type"] == "rectangle":
                    cv2.rectangle(self.canvas, shape["top_left"], shape["bottom_right"], shape["color"], self.brush_size)
                elif shape["type"] == "filled_area":
                    mask = np.zeros((self.canvas.shape[0] + 2, self.canvas.shape[1] + 2), dtype=np.uint8)
                    cv2.floodFill(self.canvas, mask, (shape["x"], shape["y"]), shape["color"], loDiff=5, upDiff=5)
        except:
            pass

    def load_gesture_settings(self):
        try:
            with open('gesture_settings.json', 'r') as f:
                settings = json.load(f)
                if "circle_fingers" not in settings:
                    settings["circle_fingers"] = [1, 2]
                return settings
        except:
            return {
                "draw_line": [0, 1, 0, 0, 0],
                "draw_rectangle": [0, 1, 1, 0, 0],
                "fill_area": [0, 1, 1, 1, 1],
                "erase": [1, 0, 0, 0, 0],
                "circle_fingers": [1, 2]
            }

    def open_settings(self):
        try:
            SettingsPanel(self.app, self.update_gesture_settings)
        except:
            messagebox.showwarning("Warning", "Failed to open settings panel.")

    def update_gesture_settings(self, new_settings):
        self.gesture_settings = new_settings
    
    def toreverse(self , num) :
        nnum = []
        for i in num :
            if i == 0 :
                nnum.append(1)
            else :
                nnum.append(0)
        return nnum


    def process_gestures(self, landmarks, finger_states, hand_idx):
        """
        پردازش ژست‌های دست برای انجام اقدامات نقاشی مانند رسم خط، دایره، مستطیل، پر کردن ناحیه یا پاک کردن.
        """
        try:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # بررسی اولیه و آماده‌سازی داده‌ها
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if not landmarks:
                return

            landmarks = self.smooth_landmarks(landmarks, hand_idx)
            finger_positions = {
                0: (landmarks[4][1], landmarks[4][2]),   # شست
                1: (landmarks[8][1], landmarks[8][2]),   # اشاره
                2: (landmarks[12][1], landmarks[12][2]), # وسط
                3: (landmarks[16][1], landmarks[16][2]), # حلقه
                4: (landmarks[20][1], landmarks[20][2])  # کوچک
            }
            finger = {
                0: landmarks[4],   # شست
                1: landmarks[8],   # اشاره
                2: landmarks[12], # وسط
                3: landmarks[16], # حلقه
                4: landmarks[20]  # کوچک
            }
            active_finger = next(
                (i for i, state in enumerate(finger_states) if state == 1 and self.gesture_settings.get("draw_line", [0, 1, 0, 0, 0])[i] == 1),
                None
            )
            x, y = finger_positions[active_finger] if active_finger is not None else (landmarks[8][1], landmarks[8][2])
            thumb_x, thumb_y = landmarks[4][1], landmarks[4][2]
            circle_fingers = self.gesture_settings.get("circle_fingers", [1, 2])

            size = self.get_distance(finger[circle_fingers[0]], finger[circle_fingers[1]])
            hand_scale = self.get_hand_scale(landmarks)
            size_threshold = hand_scale * APP_CONFIG['gestures']['finger_distance']

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # مقداردهی اولیه داده‌های دست
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if hand_idx not in self.hand_states:
                self.hand_states[hand_idx] = {
                    'state': 0,
                    'prev_x': None,
                    'prev_y': None,
                    'thumb_x': None,
                    'thumb_y': None,
                    'gesture_start_time': 0
                }
            if hand_idx not in self.hand_data:
                self.hand_data[hand_idx] = {
                    'prev_x': None,
                    'prev_y': None,
                    'thumb_x': None,
                    'thumb_y': None
                }
            state = self.hand_states[hand_idx]['state']

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # مدیریت خط رسم‌شده بین دو دست
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if self.line_drawn_between_hands and finger_states not in [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]:
                return
            elif self.line_drawn_between_hands and finger_states in [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]:
                self.line_drawn_between_hands = False

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # پردازش ژست رسم خط
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if finger_states == self.gesture_settings.get("draw_line", [0, 1, 0, 0, 0]) and state == 0:
                for color_rect in self.color_rects:
                    if self.is_point_in_rect(x, y, color_rect["rect"]):
                        self.set_color(color_rect["color"])
                        return
                if self.hand_data[hand_idx]['prev_x'] is not None:
                    self.draw_shape("line", x, y, size, hand_idx)
                self.hand_data[hand_idx]['prev_x'] = x
                self.hand_data[hand_idx]['prev_y'] = y
            else:
                self.hand_data[hand_idx]['prev_x'] = None
                self.hand_data[hand_idx]['prev_y'] = None

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # پردازش ژست رسم دایره
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            circle_fingers = self.gesture_settings.get("circle_fingers", [1, 2])
            circle_state = [1] * 5
            circle_state_prv = [0] * 5
            for finger_idx in circle_fingers:
                if finger_idx != -1:
                    circle_state[finger_idx] = 1
                    circle_state_prv[finger_idx] = 1


            if size < size_threshold and state == 0 and finger_states == circle_state:
                self.hand_states[hand_idx]['state'] = 1
            elif state == 1:
                self.draw_shape("circle", finger_positions[circle_fingers[1]][0], finger_positions[circle_fingers[1]][1], size, hand_idx, preview=True)
                confirm_state = self.toreverse(circle_state_prv)
                print(circle_state_prv)
                if finger_states == circle_state_prv:
                    self.draw_shape("circle", finger_positions[circle_fingers[1]][0], finger_positions[circle_fingers[1]][1], size, hand_idx)
                    self.hand_states[hand_idx]['state'] = 0

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # پردازش ژست رسم مستطیل
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            rectangle_state = self.gesture_settings.get("draw_rectangle")
            rectangle_finger = []
            num = -1
            for i in rectangle_state:
                num+=1
                if i == 1 :
                    rectangle_finger.append(num)
            rectangle_state_confirm = self.gesture_settings.get("draw_rectangle(confirm)")
            if finger_states == rectangle_state and state == 0:
                self.hand_data[hand_idx]['thumb_x'] = thumb_x
                self.hand_data[hand_idx]['thumb_y'] = thumb_y
                self.hand_states[hand_idx]['state'] = 2
            elif state == 2:
                self.hand_data[hand_idx]['thumb_x'] = thumb_x
                self.hand_data[hand_idx]['thumb_y'] = thumb_y
                self.draw_shape("rectangle", finger_positions[rectangle_finger[1]][0], finger_positions[rectangle_finger[1]][1], size, hand_idx, preview=True)
                if finger_states == rectangle_state_confirm:
                    self.draw_shape("rectangle", finger_positions[rectangle_finger[1]][0], finger_positions[rectangle_finger[1]][1], size, hand_idx)
                    self.hand_states[hand_idx]['state'] = 0
                    self.hand_data[hand_idx]['thumb_x'] = None
                    self.hand_data[hand_idx]['thumb_y'] = None

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # پردازش ژست پر کردن ناحیه بسته
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if finger_states == self.gesture_settings["fill_area"] and state == 0:
                if self.is_point_inside_closed_area(finger_positions[1][0], finger_positions[1][1]):
                    self.fill_closed_area(finger_positions[1][0], finger_positions[1][1])

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # پردازش ژست پاک کردن
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if finger_states == self.gesture_settings["erase"]:
                erase_finger = next(
                    (i for i, state in enumerate(finger_states) if state == 1 and self.gesture_settings["erase"][i] == 1),
                    None
                )
                if erase_finger is not None:
                    x, y = finger_positions[erase_finger]
                    cv2.circle(self.preview_frame, (x, y), 20, (255, 255, 255), 2)
                    self.erase_area(x, y, radius=20)

        except Exception as e:
            print(f"خطا در پردازش ژست‌ها: {e}")
            pass

    def update_frame(self):
        try:
            if not self.camera or not self.camera.isOpened():
                self.app.after(10, self.update_frame)
                return

            ret, frame = self.camera.read()
            if not ret:
                self.app.after(10, self.update_frame)
                return

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            self.preview_frame = frame.copy()

            for color_rect in self.color_rects:
                x1, y1, x2, y2 = color_rect["rect"]
                cv2.rectangle(self.preview_frame, (x1, y1), (x2, y2), color_rect["color"], -1)
            cv2.rectangle(self.preview_frame, (1180, 10), (1230, 60), self.current_color, -1)

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 2:
                cv2.rectangle(self.preview_frame, (0, 50, self.preview_frame.shape[1], 130), (0, 0, 0), -1)
                cv2.putText(
                    self.preview_frame,
                    "Two-handed machine is allowed in the image. Please remove the extra hands.",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
                if self.canvas is not None:
                    combined = cv2.addWeighted(self.preview_frame, 0.5, self.canvas, 1, 0)
                    img_pil = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
                    ctk_img = ctk.CTkImage(light_image=img_pil, size=(1280, 720))
                    self.image_label.configure(image=ctk_img)
                    self.image_label.image = ctk_img
                self.app.after(10, self.update_frame)
                return

            active_hands = set()
            hands_data = []

            if results.multi_hand_landmarks:
                handedness_list = results.multi_handedness if results.multi_handedness else []
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                    hand_label = handedness_list[idx].classification[0].label if idx < len(handedness_list) else 'Unknown'
                    landmarks = self.get_landmarks(rgb_frame, hand_landmarks)
                    finger_states, _ = self.detect_fingers(landmarks, handedness=hand_label, is_mirrored=True)
                    self.process_gestures(landmarks, finger_states, idx)
                    active_hands.add(idx)
                    hands_data.append((landmarks, finger_states, idx))

                if len(hands_data) == 2 and not self.line_drawn_between_hands:
                    landmarks1, finger_states1, idx1 = hands_data[0]
                    landmarks2, finger_states2, idx2 = hands_data[1]
                    if finger_states1 == [1, 1, 0, 0, 0] and finger_states2 == [1, 1, 0, 0, 0]:
                        self.draw_line_between_hands(landmarks1, landmarks2, idx1, idx2, preview=True)
                    elif (finger_states1 == [1, 1, 0, 0, 1] and finger_states2 == [1, 1, 0, 0, 0]) or \
                         (finger_states1 == [1, 1, 0, 0, 0] and finger_states2 == [1, 1, 0, 0, 1]):
                        self.draw_line_between_hands(landmarks1, landmarks2, idx1, idx2, preview=False)
            else:
                self.hand_states.clear()
                self.line_drawn_between_hands = False

            self.hand_states = {k: v for k, v in self.hand_states.items() if k in active_hands}

            if self.canvas is not None:
                combined = cv2.addWeighted(self.preview_frame, 0.5, self.canvas, 1, 0)
                img_pil = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
                ctk_img = ctk.CTkImage(light_image=img_pil, size=(1280, 720))
                self.image_label.configure(image=ctk_img)
                self.image_label.image = ctk_img
            self.app.after(10, self.update_frame)
        except:
            self.app.after(10, self.update_frame)

    def start(self):
        try:
            self.app.after(33, self.update_frame)
            self.app.mainloop()
        except:
            pass

    def cleanup(self):
        try:
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()
            if hasattr(self, 'app'):
                self.app.quit()
        except:
            pass

if __name__ == "__main__":
    app = HandDrawingApp()
    try:
        app.start()
    finally:
        app.cleanup()
