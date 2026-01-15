from typing import Dict, Tuple, List, Optional, Any

class BaseModelWrapper:
    def __init__(self, model_type: str, model_path: str, sample_frames: int, device: str = "cuda"):
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.sample_frames = int(sample_frames)
        self.device = device

    def predict(self, video_path: str, question: str, options: Dict[str, str]) -> str:
        raise NotImplementedError
    
