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
    valid = False
    if not roop.globals.source_paths:  # Check if the list is empty
        update_status('No source images selected.', NAME)
        return False

    for source_path in roop.globals.source_paths:
        if not is_image(source_path):
            update_status(f'Select a valid image for source path: {source_path}.', NAME)
            continue
        if not get_one_face(cv2.imread(source_path)):
            update_status(f'No face detected in source path: {source_path}.', NAME)
            continue
        valid = True

    if not valid:
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select a valid image or video for target path.', NAME)
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

def process_frame(frame_number, source_faces: List[Face], reference_face: Face, temp_frame: Frame) -> Frame:
    print('process_frame')

    if roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        for face in many_faces:
            box = face.bbox.astype(int)
            print(f'Frame {frame_number}: Face detected at ({box[0]}, {box[1]}), ({box[2]}, {box[3]})')
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_faces[0], target_face, temp_frame)
    else:
        frame_width = temp_frame.shape[1]
        frame_height = temp_frame.shape[0]
        current_frame_positions = roop.globals.face_data[frame_number]
        print(f'current_frame_positions={current_frame_positions}')
        many_faces = get_many_faces(temp_frame)
        for face in many_faces:
            face_box = face.bbox.astype(int)
            print(f'face_box={face_box}')
            for position in current_frame_positions:
                rect_x = int(position['x'] * frame_width if 0 <= position['x'] <= 1 else position['x'])
                rect_y = int(position['y'] * frame_height if 0 <= position['y'] <= 1 else position['y'])
                rect_w = int(position['w'] * frame_width)
                rect_h = int(position['h'] * frame_height)
                rect_box = [rect_x, rect_y, rect_x + rect_w, rect_y + rect_h]

                print(f'Checking position={position} with rectangle={rect_box}')

                if rectangles_intersect(face_box, rect_box):
                    num = position['num']
                    if num > 0 and num <= len(source_faces):
                        selected_source_face = source_faces[num - 1]  # num is 1-based index
                        print(f'Swapping face in frame {frame_number} with source face #{num}')
                        temp_frame = swap_face(selected_source_face, face, temp_frame)
                    else:
                        print(f'No valid source face for number {num}, skipping.')
                    break  # Assume only one swap per detected face
    return temp_frame



def rectangles_intersect(r1, r2):
    """Vérifie si deux rectangles r1 et r2 s'intersectent."""
    x1, y1, x2, y2 = r1  # Rectangle 1
    x3, y3, x4, y4 = r2  # Rectangle 2

    # Vérifier si les rectangles se chevauchent horizontalement et verticalement
    horizontal_overlap = (x1 <= x4) and (x3 <= x2)
    vertical_overlap = (y1 <= y4) and (y3 <= y2)

    return horizontal_overlap and vertical_overlap


def parse_face_images(args):
    face_images = {}
    for i in range(1, 10):  # Assumant que vous avez des images numérotées de 1 à 9
        face_image_path = getattr(args, f'face{i}', None)
        if face_image_path:
            face_images[i] = face_image_path
    return face_images


def process_frames(source_paths: list[str], temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    print(f'process_frames')

    source_faces = []
    for source_path in source_paths:
        image = cv2.imread(source_path)
        if image is None:
            print(f"Erreur: Impossible de lire l'image à partir du chemin {source_path}.")
            continue
        face = get_one_face(image)
        if face is not None:
            source_faces.append(face)
        else:
            print(f"Aucun visage détecté dans {source_path}.")

    reference_face = None if roop.globals.many_faces else get_face_reference()

    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        filename = temp_frame_path.split('/')[-1]
        frame_number = int(filename.split('.')[0])
        print(f'Frame {frame_number}')
        result = process_frame(frame_number, source_faces, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(frame_number, source_paths: list[str], target_path: str, output_path: str) -> None:
    source_faces = []
    for source_path in source_paths:
        image = cv2.imread(source_path)
        if image is None:
            print(f"Erreur: Impossible de lire l'image à partir du chemin {source_path}.")
            continue
        face = get_one_face(image)
        if face is not None:
            source_faces.append(face)
        else:
            print(f"Aucun visage détecté dans {source_path}.")
            
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(frame_number, source_faces, reference_face, target_frame)
    cv2.imwrite(output_path, result)


def save_image(image, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(os.path.join(path, filename), image)

def process_video(source_paths: List[str], temp_frame_paths: List[str]) -> None:
    debug_path = '/kaggle/working/data'
    if not roop.globals.many_faces and not get_face_reference():
        print(f'reference_frame_number={roop.globals.reference_frame_number}')
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        save_image(reference_frame, debug_path, 'reference_frame.jpg')
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        if reference_face is not None:
            print('Face detected !')
        set_face_reference(reference_face)
    roop.processors.frame.core.process_video(source_paths, temp_frame_paths, process_frames)