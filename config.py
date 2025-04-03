APP_CONFIG = {
    'window': {
        'title': 'drawing',
        'width': 1500,
        'height': 800,
        'canvas_width': 1280,
        'canvas_height': 720
    },
    'camera': {
        'width': 1000,
        'height': 800,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5
    },
    'drawing': {
        'max_undo_steps': 50,
        'default_brush_size': 5,
        'eraser_radius': 20,
        'flood_fill_threshold': 5
    },
    'gestures': {
        'finger_distance': 0.3,
        'smoothing_factor': 0.5
    },
    'colors': [
        {'name': 'Red', 'rgb': [255, 0, 0], 'hex': '#FF0000'},
        {'name': 'Green', 'rgb': [0, 255, 0], 'hex': '#00FF00'},
        {'name': 'Blue', 'rgb': [0, 0, 255], 'hex': '#0000FF'},
        {'name': 'Yellow', 'rgb': [0, 255, 255], 'hex': '#00FFFF'},
        {'name': 'White', 'rgb': [255, 255, 255], 'hex': '#FFFFFF'}
    ]
} 