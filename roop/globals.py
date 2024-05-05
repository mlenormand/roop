from typing import List, Optional, Dict, Tuple, Any
from collections import defaultdict

source_paths: List[str] = []
faces_path: Optional[str] = None
target_path: Optional[str] = None
output_path: Optional[str] = None
headless: Optional[bool] = None
frame_processors: List[str] = []
keep_fps: Optional[bool] = None
keep_frames: Optional[bool] = None
skip_audio: Optional[bool] = None
many_faces: Optional[bool] = None
reference_face_position: Optional[int] = None
reference_frame_number: Optional[int] = None
similar_face_distance: Optional[float] = None
temp_frame_format: Optional[str] = None
temp_frame_quality: Optional[int] = None
output_video_encoder: Optional[str] = None
output_video_quality: Optional[int] = None
max_memory: Optional[int] = None
execution_providers: List[str] = []
execution_threads: Optional[int] = None
log_level: str = 'error'
face_data: None
known_faces: Dict[int, Any] = {}  # Stocke les images des visages connus par numéro
known_face_encodings: Dict[int, Any] = {}  # Stocke les encodages des visages connus par numéro
auto: bool = False  # Indique si le mode auto est activé