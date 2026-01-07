import os
from evaluators.evaluator import * 
from detectors.detector import GroundingDINODetector
from drones.dronecore.camera import build_camera


class Drone:
    def __init__(self, name, initial_position=(0,0,0), initial_orientation=(0,0,0)):
        self.name = name
        self.initial_position = initial_position
        self.position = initial_position
        self.orientation = initial_orientation  # Euler angles? # (roll, pitch, yaw)
        
    def move(self, delta_position, delta_orientation):
        self.position = tuple(p + dp for p, dp in zip(self.position, delta_position))
        self.orientation = tuple(o + do for o, do in zip(self.orientation, delta_orientation))
        
    def move_to(self, position, orientation):
        self.position = position
        self.orientation = orientation
    
    def get_pose(self):
        return self.position, self.orientation
    
    def get_state(self):
        return {
            "name": self.name,
            "position": self.position,
            "orientation": self.orientation
        }
    
    
class DroneVisionAgent(Drone):
    def __init__(
        self, 
        name, 
        camera_info, 
        initial_position=(0,0,0), 
        initial_orientation=(0,0,0),
        device='cuda',
        detector_config_path='repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
        detector_weights_path='repos/GroundingDINO/weights/groundingdino_swint_ogc.pth',
        target_prompt=None,
        target_image=None,
        target_image_embedding=None,
    ):
        super().__init__(name, initial_position, initial_orientation)
        
        # Camera
        self.camera_info = camera_info
        
        # Detector & Evaluator
        self.detector = GroundingDINODetector(detector_config_path, detector_weights_path, device=device)
        self.evaluator = None
        
        # Target
        self.target_prompt = target_prompt
        self.target_image = target_image
        self.target_image_embedding = target_image_embedding
        
        # Memory
        self.captured_images = []  # List of (image, pose, detections, evaluation) tuples
        self.best_view = None  # (image, pose, detections, evaluation)
        self.best_score = float('-inf')
        self.history = []  # List of past poses and actions
        
        # Callbacks
        self.move_ = None
        self.get_pose_ = None
        self.render_ = None
        
        
    # Call back (to interact with blender or real world)
    def callback_move_function(self, move_func):
        self.move_ = move_func      
        
    def callback_pose_function(self, get_pose_func):
        self.get_pose_ = get_pose_func

    def callback_render_function(self, render_func):
        self.render_ = render_func
            

    # Target
    def set_target(self, target_prompt=None, target_image=None):
        if target_prompt is not None:
            self.target_prompt = target_prompt
        if target_image is not None:
            self.target_image = target_image
            self.target_image_embedding = None #TODO
    
    def clear_target_prompt(self):
        self.target_prompt = None
        
    def clear_target_image(self):
        self.target_image = None
        self.target_image_embedding = None
        

    # Detect
    def detect_target(self, image, target_prompt=None, target_image=None, box_threshold=0.35, text_threshold=0.25):
    # TODO: detect with reference frame?
        if target_prompt is None:
            target_prompt = self.target_prompt
        if not target_prompt or image is None:
            return []
        if isinstance(image, (str, bytes, os.PathLike)):
            return self.detector.detect_file(
                image_path=image,
                caption=target_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        return self.detector.detect(
            image_bgr=image,
            caption=target_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        
    # Evaluate
    def evaluate_image(self, image, detections, metrics=None):
        # Metrics = [...] (see src/evaluators/evaluator.py for available metrics)
        
        if image is None:
            return EvaluationResult(name="evaluation", score=0.0, details={"reason": "image_missing"})

        if isinstance(image, (str, bytes, os.PathLike)):
            import cv2

            image_bgr = cv2.imread(str(image))
            if image_bgr is None:
                return EvaluationResult(
                    name="evaluation",
                    score=0.0,
                    details={"reason": "image_load_failed", "path": str(image)},
                )
        else:
            image_bgr = image

        evaluator = self.evaluator
        if metrics:
            if isinstance(metrics, str):
                metrics = [metrics]
            metric_builders = {
                "color_contrast": ColorContrastEvaluator,
                "subject_size": SubjectSizeEvaluator,
                "rule_of_thirds": RuleOfThirdsEvaluator,
                "breathing_space": BreathingSpaceEvaluator,
                "brightness": BrightnessEvaluator,
                "laplacian": LaplacianEvaluator,
                "stddev": StddevEvaluator,
                "qalign": QAlignEvaluator,
                "composition": default_composition_evaluator,
            }
            evaluators = []
            for metric in metrics:
                key = str(metric).strip().lower()
                builder = metric_builders.get(key)
                if builder is None:
                    raise ValueError(f"Unknown metric: {metric}")
                evaluators.append(builder() if callable(builder) else builder)
            evaluator = evaluators[0] if len(evaluators) == 1 else CompositeEvaluator(evaluators, name="custom")

        if evaluator is None:
            evaluator = default_composition_evaluator()
            self.evaluator = evaluator

        return evaluator.evaluate(image_bgr, detections)

    # Action
    def get_action(self):
        return NotImplementedError("get_action() not implemented yet.")
    