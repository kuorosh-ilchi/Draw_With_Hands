import cv2
import mediapipe as mp
import numpy as np
import customtkinter as ctk
from PIL import Image
import tkinter as tk
from tkinter import messagebox
import os
import sys
import logging
from config import APP_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

class HandDrawingApp:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=APP_CONFIG['camera']['min_detection_confidence'],
            min_tracking_confidence=APP_CONFIG['camera']['min_tracking_confidence']
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.camera_index = 0
        self.camera = None
        self.initialize_camera()
        
        self.canvas = np.zeros_like(self.frame, dtype=np.uint8)
        self.current_color = [255, 255, 255]
        self.brush_size = APP_CONFIG['drawing']['default_brush_size']
        self.hand_data = {}
        self.shapes = []
        self.line_drawn_between_hands = False
        self.max_undo_steps = APP_CONFIG['drawing']['max_undo_steps']
        self.hand_states = {}

        self.color_rects = [
            {"color": [255, 0, 0], "rect": (50, 10, 150, 60)},   # Red
            {"color": [0, 255, 0], "rect": (200, 10, 300, 60)},  # Green
            {"color": [0, 0, 255], "rect": (350, 10, 450, 60)},  # Blue
            {"color": [0, 255, 255], "rect": (500, 10, 600, 60)}, # Cyan
            {"color": [255, 255, 255], "rect": (650, 10, 750, 60)} # White
        ]

        self.app = ctk.CTk()
        self.app.title("Hand Drawing App")
        self.app.geometry(f"{APP_CONFIG['window']['width']}x{APP_CONFIG['window']['height']}")

        self.image_frame = ctk.CTkFrame(
            self.app,
            width=APP_CONFIG['window']['canvas_width'],
            height=APP_CONFIG['window']['canvas_height']
        )
        self.image_frame.pack(side=ctk.LEFT, padx=10, pady=10)
        self.image_label = ctk.CTkLabel(self.image_frame, text="")
        self.image_label.pack()

        self.control_frame = ctk.CTkFrame(self.app, width=200, height=720)
        self.control_frame.pack(side=ctk.RIGHT, padx=10, pady=10)
        self.setup_controls()

        self.setup_gesture_recognition()

    def setup_controls(self):
        ctk.CTkLabel(self.control_frame, text="Select Camera:").pack(pady=5)
        self.camera_var = tk.StringVar(value="Camera 0")
        self.camera_menu = ctk.CTkOptionMenu(
            self.control_frame,
            values=[f"Camera {i}" for i in range(4)],
            variable=self.camera_var,
            command=self.change_camera
        )
        self.camera_menu.pack(pady=5)

        ctk.CTkLabel(self.control_frame, text="Brush Size").pack(pady=5)
        self.brush_slider = ctk.CTkSlider(
            self.control_frame,
            from_=2,
            to=20,
            command=self.update_brush_size
        )
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(pady=5)
        
        self.brush_size_label = ctk.CTkLabel(self.control_frame, text=str(self.brush_size))
        self.brush_size_label.pack(pady=2)

        ctk.CTkButton(self.control_frame, text="Clear", command=self.clear_canvas).pack(pady=5)
        ctk.CTkButton(self.control_frame, text="Undo", command=self.undo_last).pack(pady=5)
        ctk.CTkButton(self.control_frame, text="Save", command=self.save_drawing).pack(pady=5)

    def setup_gesture_recognition(self):
        self.gesture_thresholds = APP_CONFIG['gestures']
        self.prev_landmarks = {}

    def initialize_camera(self):
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            print(f"Camera opened: {self.camera.isOpened()}")
            if not self.camera.isOpened():
                raise RuntimeError(f"Error opening camera {self.camera_index}")
            self.camera.set(3, APP_CONFIG['camera']['width'])
            self.camera.set(4, APP_CONFIG['camera']['height'])
            self.ret, self.frame = self.camera.read()
            print(f"Frame read: {self.ret}, Shape: {self.frame.shape}")
            if not self.ret:
                raise RuntimeError("Error reading initial frame from camera")
        except Exception as e:
            messagebox.showerror("Camera Error", f"Error initializing camera: {str(e)}")
            logging.error(f"Camera initialization error: {str(e)}")
            sys.exit(1)

    def save_drawing(self):
        try:
            if not os.path.exists('drawings'):
                os.makedirs('drawings')
            filename = f"drawings/drawing_{len(os.listdir('drawings')) + 1}.png"
            cv2.imwrite(filename, self.canvas)
            messagebox.showinfo("Save", f"Drawing saved as {filename}")
            logging.info(f"Drawing saved as {filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving drawing: {str(e)}")
            logging.error(f"Save error: {str(e)}")

    def clear_canvas(self):
        if messagebox.askyesno("Clear", "Are you sure you want to clear the drawing?"):
            self.canvas = np.zeros_like(self.frame, dtype=np.uint8)
            self.shapes = []
            self.line_drawn_between_hands = False
            self.redraw_canvas()
            logging.info("Canvas cleared")

    def change_camera(self, selection):
        new_index = int(selection.split()[-1])
        if new_index != self.camera_index:
            try:
                self.camera.release()
                self.camera_index = new_index
                self.camera = cv2.VideoCapture(self.camera_index)
                self.camera.set(3, APP_CONFIG['camera']['width'])
                self.camera.set(4, APP_CONFIG['camera']['height'])
                logging.info(f"Camera changed to index {new_index}")
            except Exception as e:
                messagebox.showerror("Camera Error", f"Error changing camera: {str(e)}")
                logging.error(f"Camera change error: {str(e)}")

    def update_brush_size(self, value):
        self.brush_size = int(value)
        self.brush_size_label.configure(text=str(self.brush_size))

    def set_color(self, color):
        self.current_color = color
        logging.info(f"Color changed to {color}")

    def undo_last(self):
        if self.shapes:
            self.shapes.pop()
            if len(self.shapes) > self.max_undo_steps:
                self.shapes = self.shapes[-self.max_undo_steps:]
            self.redraw_canvas()
            logging.info("Last action undone")

    def get_distance(self, point1, point2):
        x1, y1 = point1[1], point1[2]
        x2, y2 = point2[1], point2[2]
        return int(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    def get_finger_base(self, finger_id):
        base_map = {4: 5, 8: 5, 12: 9, 16: 13, 20: 17}
        return base_map.get(finger_id, 0)

    def get_hand_scale(self, landmarks):
        if not landmarks:
            return 0
        wrist = landmarks[0]
        index_base = landmarks[5]
        return self.get_distance(wrist, index_base)

    def smooth_landmarks(self, current_landmarks, hand_idx):
        if hand_idx not in self.prev_landmarks:
            self.prev_landmarks[hand_idx] = current_landmarks
            return current_landmarks
            
        smoothed = []
        for curr, prev in zip(current_landmarks, self.prev_landmarks[hand_idx]):
            x = int(curr[1] * (1 - self.gesture_thresholds['smoothing_factor']) + 
                   prev[1] * self.gesture_thresholds['smoothing_factor'])
            y = int(curr[2] * (1 - self.gesture_thresholds['smoothing_factor']) + 
                   prev[2] * self.gesture_thresholds['smoothing_factor'])
            smoothed.append([curr[0], x, y])
        
        self.prev_landmarks[hand_idx] = smoothed
        return smoothed

    def detect_fingers(self, landmarks):
        if not landmarks:
            return [], 0
        finger_tips = [4, 8, 12, 16, 20]
        wrist = landmarks[0]
        finger_states = []
        hand_scale = self.get_hand_scale(landmarks)
        for tip_id in finger_tips:
            tip = landmarks[tip_id]
            base = landmarks[self.get_finger_base(tip_id)]
            tip_to_wrist = self.get_distance(tip, wrist)
            base_to_wrist = self.get_distance(base, wrist)
            threshold = hand_scale * self.gesture_thresholds['finger_distance']
            finger_states.append(1 if tip_to_wrist - base_to_wrist > threshold else 0)
        return finger_states, sum(finger_states)

    def get_landmarks(self, rgb_frame, hand_landmarks):
        if not hand_landmarks:
            return []
        height, width, _ = rgb_frame.shape
        landmarks = []
        for idx, lm in enumerate(hand_landmarks.landmark):
            x = int(lm.x * width)
            y = int(lm.y * height)
            landmarks.append([idx, x, y])
        return landmarks

    def draw_shape(self, shape_type, x, y, size, hand_idx, preview=False):
        target = self.preview_frame if preview else self.canvas
        if shape_type == "line" and self.hand_data.get(hand_idx, {}).get("prev_x") is not None:
            prev_x = self.hand_data[hand_idx]["prev_x"]
            prev_y = self.hand_data[hand_idx]["prev_y"]
            temp_line = np.zeros_like(self.canvas, dtype=np.uint8)
            cv2.line(temp_line, (prev_x, prev_y), (x, y), self.current_color, self.brush_size)
            if not preview:
                self.canvas = cv2.add(self.canvas, temp_line)
                self.shapes.append({"type": "line", "start": (prev_x, prev_y), "end": (x, y), 
                                  "color": self.current_color.copy(), "size": self.brush_size})
            else:
                cv2.line(target, (prev_x, prev_y), (x, y), self.current_color, self.brush_size)
        elif shape_type == "circle":
            cv2.circle(target, (x, y), size, self.current_color, self.brush_size)
            if not preview:
                self.shapes.append({"type": "circle", "center": (x, y), "radius": size, 
                                  "color": self.current_color.copy()})
        elif shape_type == "rectangle":
            thumb_x = self.hand_data.get(hand_idx, {}).get("thumb_x")
            thumb_y = self.hand_data.get(hand_idx, {}).get("thumb_y")
            if thumb_x is not None and thumb_y is not None:
                cv2.rectangle(target, (x, y), (thumb_x, thumb_y), self.current_color, self.brush_size)
                if not preview:
                    self.shapes.append({"type": "rectangle", "top_left": (x, y), 
                                      "bottom_right": (thumb_x, thumb_y), "color": self.current_color.copy()})

    def draw_line_between_hands(self, hand1_landmarks, hand2_landmarks, hand_idx1, hand_idx2, preview=False):
        x1 = hand1_landmarks[8][1]
        y1 = hand1_landmarks[8][2]
        x2 = hand2_landmarks[8][1]
        y2 = hand2_landmarks[8][2]
        size = self.get_distance(hand1_landmarks[8], hand1_landmarks[4])
        target = self.preview_frame if preview else self.canvas
        if preview:
            cv2.line(target, (x1, y1), (x2, y2), self.current_color, self.brush_size)
        else:
            cv2.line(self.canvas, (x1, y1), (x2, y2), self.current_color, self.brush_size)
            self.shapes.append({"type": "line", "start": (x1, y1), "end": (x2, y2), 
                              "color": self.current_color.copy(), "size": self.brush_size})
            self.line_drawn_between_hands = True

    def is_point_in_shape(self, x, y, shape):
        if shape["type"] == "line":
            start, end = shape["start"], shape["end"]
            line_vec = np.array(end) - np.array(start)
            point_vec = np.array([x, y]) - np.array(start)
            length = np.linalg.norm(line_vec)
            if length == 0:
                return False
            t = max(0, min(1, np.dot(point_vec, line_vec) / (length ** 2)))
            projection = np.array(start) + t * line_vec
            distance = np.linalg.norm(np.array([x, y]) - projection)
            return distance < self.brush_size
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

    def is_point_in_rect(self, x, y, rect):
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def erase_area(self, thumb_x, thumb_y, radius=20):
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
                distance = np.linalg.norm(np.array([thumb_x, thumb_y]) - projection)
                if distance < radius:
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

    def fill_shape(self, shape):
        if shape["type"] == "line":
            cv2.line(self.canvas, shape["start"], shape["end"], shape["color"], self.brush_size)
        elif shape["type"] == "circle":
            cv2.circle(self.canvas, shape["center"], shape["radius"], shape["color"], -1)
        elif shape["type"] == "rectangle":
            cv2.rectangle(self.canvas, shape["top_left"], shape["bottom_right"], shape["color"], -1)

    def is_point_inside_closed_area(self, x, y):
        mask = np.zeros((self.canvas.shape[0] + 2, self.canvas.shape[1] + 2), dtype=np.uint8)
        temp_canvas = self.canvas.copy()
        _, _, _, rect = cv2.floodFill(temp_canvas, mask, (x, y), (255, 255, 255), loDiff=5, upDiff=5)
        area = rect[2] * rect[3]
        total_area = self.canvas.shape[0] * self.canvas.shape[1]
        return rect[2] > 0 and rect[3] > 0 and area < total_area * 0.8

    def fill_closed_area(self, x, y):
        if not (0 <= x < self.canvas.shape[1] and 0 <= y < self.canvas.shape[0]):
            print("Coordinates out of bounds")
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
            print("Area too large, fill cancelled")
            return False

    def redraw_canvas(self):
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

    def process_gestures(self, landmarks, finger_states, hand_idx):
        if not landmarks:
            return
        landmarks = self.smooth_landmarks(landmarks, hand_idx)
        x = landmarks[8][1]  # Index finger tip x
        y = landmarks[8][2]  # Index finger tip y
        thumb_x = landmarks[4][1]  # Thumb tip x
        thumb_y = landmarks[4][2]  # Thumb tip y
        size = self.get_distance(landmarks[8], landmarks[4])
        hand_scale = self.get_hand_scale(landmarks)
        size_threshold = hand_scale * self.gesture_thresholds['finger_distance']

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
        
        if self.line_drawn_between_hands and finger_states not in [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]:
            return
        elif self.line_drawn_between_hands and finger_states in [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]:
            self.line_drawn_between_hands = False

        if finger_states == [0, 1, 0, 0, 0] and state == 0:  # Select color or draw line
            for color_rect in self.color_rects:
                if self.is_point_in_rect(x, y, color_rect["rect"]):
                    self.set_color(color_rect["color"])
                    print(f"Color changed to {self.current_color}")
                    return  # Exit to avoid drawing line when selecting color
            # Draw line with index finger
            if self.hand_data[hand_idx]['prev_x'] is not None:
                self.draw_shape("line", x, y, size, hand_idx)
            self.hand_data[hand_idx]['prev_x'] = x
            self.hand_data[hand_idx]['prev_y'] = y

        elif finger_states != [0, 1, 0, 0, 0]:  # Reset line drawing when gesture changes
            self.hand_data[hand_idx]['prev_x'] = None
            self.hand_data[hand_idx]['prev_y'] = None

        if size < size_threshold and state == 0 and finger_states == [1, 1, 1, 1, 1]:
            self.hand_states[hand_idx]['state'] = 1
        elif state == 1:
            self.draw_shape("circle", x, y, size, hand_idx, preview=True)
            if finger_states == [1, 1, 0, 0, 0]:
                self.draw_shape("circle", x, y, size, hand_idx)
                self.hand_states[hand_idx]['state'] = 0

        elif finger_states == [0, 1, 1, 0, 0] and state == 0:
            self.hand_data[hand_idx]['thumb_x'] = thumb_x
            self.hand_data[hand_idx]['thumb_y'] = thumb_y
            self.hand_states[hand_idx]['state'] = 2
        elif state == 2:
            self.hand_data[hand_idx]['thumb_x'] = thumb_x
            self.hand_data[hand_idx]['thumb_y'] = thumb_y
            self.draw_shape("rectangle", x, y, size, hand_idx, preview=True)
            if finger_states == [1, 1, 0, 0, 1]:
                self.draw_shape("rectangle", x, y, size, hand_idx)
                self.hand_states[hand_idx]['state'] = 0
                self.hand_data[hand_idx]['thumb_x'] = None
                self.hand_data[hand_idx]['thumb_y'] = None

        if finger_states == [0, 1, 1, 1, 1] and state == 0:
            inside_closed = self.is_point_inside_closed_area(x, y)
            inside_shape = any(self.is_point_in_shape(x, y, shape) for shape in self.shapes)
            if inside_closed:
                filled = self.fill_closed_area(x, y)
                print(f"Area filled: {filled}")

        if finger_states == [1, 0, 0, 0, 0]:
            cv2.circle(self.preview_frame, (thumb_x, thumb_y), 20, (255, 255, 255), 2)
            self.erase_area(thumb_x, thumb_y, radius=20)

    def update_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            print("Failed to read frame")
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

        active_hands = set()
        hands_data = []

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks = self.get_landmarks(rgb_frame, hand_landmarks)
                finger_states, _ = self.detect_fingers(landmarks)
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

        combined = cv2.addWeighted(self.preview_frame, 0.5, self.canvas, 1, 0)
        img_pil = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        ctk_img = ctk.CTkImage(light_image=img_pil, size=(1280, 720))
        self.image_label.configure(image=ctk_img)
        self.image_label.image = ctk_img
        self.app.after(10, self.update_frame)

    def start(self):
        self.app.after(33, self.update_frame)
        self.app.mainloop()

    def cleanup(self):
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'app'):
            self.app.quit()
        logging.info("Application cleaned up")

if __name__ == "__main__":
    app = HandDrawingApp()
    try:
        app.start()
    finally:
        app.cleanup()