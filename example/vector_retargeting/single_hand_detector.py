import mediapipe as mp
import mediapipe.framework as framework
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


class SingleHandDetector:
    def __init__(
        self,
        hand_type="Right",
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        selfie=False,
    ):
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.selfie = selfie
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

    @staticmethod
    def draw_skeleton_on_image(
        image, keypoint_2d: landmark_pb2.NormalizedLandmarkList, style="white"
    ):
        if style == "default":
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )
        elif style == "white":
            landmark_style = {}
            for landmark in HandLandmark:
                landmark_style[landmark] = DrawingSpec(
                    color=(255, 48, 48), circle_radius=4, thickness=-1
                )

            connections = hands_connections.HAND_CONNECTIONS
            connection_style = {}
            for pair in connections:
                connection_style[pair] = DrawingSpec(thickness=2)

            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                landmark_style,
                connection_style,
            )

        return image

    def detect(self, rgb, return_meta: bool = False):
        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            if return_meta:
                return 0, None, None, None, {}
            return 0, None, None, None

        desired_hand_num = -1
        label = None
        for i in range(len(results.multi_hand_landmarks)):
            label = results.multi_handedness[i].ListFields()[0][1][0].label
            if label == self.detected_hand_type:
                desired_hand_num = i
                break
        if desired_hand_num < 0:
            if return_meta:
                return 0, None, None, None, {}
            return 0, None, None, None

        keypoint_3d = results.multi_hand_world_landmarks[desired_hand_num]
        keypoint_2d = results.multi_hand_landmarks[desired_hand_num]
        num_box = len(results.multi_hand_landmarks)

        # Parse 3d keypoint from MediaPipe hand detector
        raw_keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
        keypoint_3d_array = raw_keypoint_3d_array - raw_keypoint_3d_array[0:1, :]
        palm_frame_points, palm_frame, palm_frame_info = self.normalize_to_palm_frame(
            raw_keypoint_3d_array
        )
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano

        if return_meta:
            return (
                num_box,
                joint_pos,
                keypoint_2d,
                mediapipe_wrist_rot,
                {
                    "handedness": label,
                    "raw_landmarks": keypoint_3d_array,
                    "operator_landmarks": joint_pos,
                    "palm_frame_landmarks": palm_frame_points,
                    "palm_frame": palm_frame,
                    "palm_frame_info": palm_frame_info,
                },
            )
        return num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot

    @staticmethod
    def parse_keypoint_3d(
        keypoint_3d: framework.formats.landmark_pb2.LandmarkList,
    ) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d.landmark[i].x
            keypoint[i][1] = keypoint_3d.landmark[i].y
            keypoint[i][2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(
        keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size
    ) -> np.ndarray:
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d.landmark[i].x
            keypoint[i][1] = keypoint_2d.landmark[i].y
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame

    @staticmethod
    def _safe_normalize(vector: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, bool]:
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            fallback_norm = np.linalg.norm(fallback)
            if fallback_norm < 1e-8:
                return np.array([1.0, 0.0, 0.0]), False
            return fallback / fallback_norm, False
        return vector / norm, True

    @staticmethod
    def normalize_to_palm_frame(keypoint_3d_array: np.ndarray):
        """Return wrist-centered landmarks in an explicit palm frame.

        Palm-frame axes are:
        x: palm normal, y: index-to-pinky spread, z: finger-forward.
        This keeps z mostly along the fingers so the existing x2 ref transform
        can map [z, x, y] into the robot target-vector frame.
        """
        centered = keypoint_3d_array - keypoint_3d_array[0:1, :]

        spread_raw = centered[5] - centered[17]
        spread, spread_ok = SingleHandDetector._safe_normalize(
            spread_raw, np.array([0.0, 1.0, 0.0])
        )

        forward_raw = centered[9]
        forward_projected = forward_raw - float(np.dot(forward_raw, spread)) * spread
        forward, forward_ok = SingleHandDetector._safe_normalize(
            forward_projected, np.array([0.0, 0.0, 1.0])
        )

        normal_raw = np.cross(spread, forward)
        normal, normal_ok = SingleHandDetector._safe_normalize(
            normal_raw, np.array([1.0, 0.0, 0.0])
        )

        # Re-orthogonalize spread after the normal/forward pair is stable.
        spread = np.cross(forward, normal)
        spread, reorthogonalized_spread_ok = SingleHandDetector._safe_normalize(
            spread, np.array([0.0, 1.0, 0.0])
        )

        frame = np.stack([normal, spread, forward], axis=1)
        palm_points = centered @ frame
        return palm_points, frame, {
            "spread_ok": spread_ok and reorthogonalized_spread_ok,
            "forward_ok": forward_ok,
            "normal_ok": normal_ok,
        }
