from typing import Any, List, Callable
import cv2
import insightface
import threading

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video
import os

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER

    FACE_SWAPPER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

def is_point_in_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def process_frame(frame_number, source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    print('process_frame')

    if roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        for face in many_faces:
            box = face.bbox.astype(int)
            print(f'Frame {frame_number}: Face detected at ({box[0]}, {box[1]}), ({box[2]}, {box[3]})')
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        frame_width = temp_frame.shape[1]
        frame_height = temp_frame.shape[0]
        current_frame_positions = roop.globals.face_positions.get(f'frame_{frame_number}', [])
        print(f'current_frame_positions={current_frame_positions}')
        many_faces = get_many_faces(temp_frame)
        for face in many_faces:
            box = face.bbox.astype(int)
            print(f'box={box}')
            for position in current_frame_positions:
                if 0 <= position['x'] <= 1 and 0 <= position['y'] <= 1:
                    # Convertir les coordonnées relatives en coordonnées absolues
                    absolute_x = int(position['x'] * frame_width)
                    absolute_y = int(position['y'] * frame_height)
                else:
                    # Utiliser les positions absolues telles quelles
                    absolute_x = int(position['x'])
                    absolute_y = int(position['y'])
                print(f'position={position}, absolute_position=({absolute_x}, {absolute_y})')

                if is_point_in_bbox((absolute_x, absolute_y), box):
                    print(f'Swaping face in frame {frame_number}')
                    temp_frame = swap_face(source_face, face, temp_frame)
                    break  # Assuming only one swap per detected face

    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    print(f'process_frames')
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()

    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        filename = temp_frame_path.split('/')[-1]
        frame_number = int(filename.split('.')[0])
        print(f'Frame {frame_number}')
        result = process_frame(frame_number, source_face, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(frame_number, source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(frame_number, source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)

def save_image(image, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(os.path.join(path, filename), image)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    debug_path = '/kaggle/working/data'  # Définissez le chemin vers le dossier de débogage

    if not roop.globals.many_faces and not get_face_reference():
        print(f'reference_frame_number={roop.globals.reference_frame_number}')
        print(f'temp_frame_paths[roop.globals.reference_frame_number]={temp_frame_paths[roop.globals.reference_frame_number]}')
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])

        save_image(reference_frame, debug_path, 'reference_frame.jpg')

        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)

        # Vérifier si un visage a été détecté et l'enregistrer
        if reference_face is not None:
            print('Face detected !')

        set_face_reference(reference_face)
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
