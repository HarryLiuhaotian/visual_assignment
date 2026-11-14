import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml


INPUT_SIZE = 640

COLORS = (
    "blue",
    "red",
    "neutral",
    "purple",
)

PATTERNS = (
    "sentry",
    "1",
    "2",
    "3",
    "4",
    "5",
    "outpost",
    "base_small",
    "base_big",
)

# Pattern 1 and base_big use the large armor model.
LARGE_PATTERN_IDS = frozenset(
    (1, 8)
)

SMALL_WIDTH_M = 0.135
LARGE_WIDTH_M = 0.230
ARMOR_HEIGHT_M = 0.056


def parse_args():
    # Read model, video, calibration and output paths
    # from the command line.
    parser = argparse.ArgumentParser(
        description="Armor detection and pose estimation"
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="objectness threshold",
    )

    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="optional frame limit for testing",
    )

    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        parser.error(
            "--confidence must be in [0, 1]"
        )

    if not 0.0 <= args.nms_iou <= 1.0:
        parser.error(
            "--nms-iou must be in [0, 1]"
        )

    if (
        args.max_frames is not None
        and args.max_frames < 0
    ):
        parser.error(
            "--max-frames must be non-negative"
        )

    return args


def load_calibration(path: Path):
    # open the calibration file and load the camera matrix
    # and distortion coefficients, changing to numpy arrays

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        data = yaml.safe_load(stream)

    camera_matrix = np.asarray(
        data["camera_matrix"],
        dtype=np.float64,
    )

    distortion = np.asarray(
        data["distort_coeffs"],
        dtype=np.float64,
    ).reshape(-1, 1)

    if camera_matrix.shape != (3, 3):
        raise ValueError(
            "camera matrix must be 3 x 3"
        )

    if distortion.size < 4:
        raise ValueError(
            "invalid distortion coefficients"
        )

    if not np.isfinite(
        camera_matrix
    ).all():
        raise ValueError(
            "camera matrix contains non-finite values"
        )

    if not np.isfinite(
        distortion
    ).all():
        raise ValueError(
            "distortion contains non-finite values"
        )

    return camera_matrix, distortion


def letterbox(frame: np.ndarray):
    # Resize the image while preserving its aspect ratio.
    height, width = frame.shape[:2]

    scale = min(
        INPUT_SIZE / width,
        INPUT_SIZE / height,
    )

    resized_width = int(
        round(width * scale)
    )

    resized_height = int(
        round(height * scale)
    )

    resized = cv2.resize(
        frame,
        (
            resized_width,
            resized_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    # Calculate the centered padding.
    pad_x = (
        INPUT_SIZE - resized_width
    ) // 2

    pad_y = (
        INPUT_SIZE - resized_height
    ) // 2

    # Create a 640 x 640 image filled with gray value 114.
    canvas = np.full(
        (
            INPUT_SIZE,
            INPUT_SIZE,
            3,
        ),
        114,
        dtype=np.uint8,
    )

    # Place the resized image in the center of the canvas.
    canvas[
        pad_y : pad_y + resized_height,
        pad_x : pad_x + resized_width,
    ] = resized

    return canvas, scale, pad_x, pad_y


def preprocess(frame: np.ndarray):
    # Resize and pad the original frame.
    image, scale, pad_x, pad_y = (
        letterbox(frame)
    )

    # OpenCV reads images in BGR order,
    # but the model expects RGB.
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    # Change the image layout from:
    #
    # HWC: (640, 640, 3)
    #
    # to:
    #
    # NCHW: (1, 3, 640, 640)
    tensor = image.transpose(
        2,
        0,
        1,
    )[None]

    # The model declares a float16 input.
    tensor = tensor.astype(
        np.float16
    )

    # Normalize pixel values from [0, 255]
    # to [0, 1].
    tensor /= np.float16(255.0)

    # Store the tensor in contiguous memory.
    tensor = np.ascontiguousarray(
        tensor
    )

    return tensor, scale, pad_x, pad_y


def sigmoid(values: np.ndarray):
    # Convert objectness logits into probabilities
    # between 0 and 1.
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    result = np.empty_like(values)

    # Use separate formulas for positive and negative
    # values to avoid numerical overflow.
    positive = values >= 0

    result[positive] = 1.0 / (
        1.0
        + np.exp(-values[positive])
    )

    exp_values = np.exp(
        values[~positive]
    )

    result[~positive] = (
        exp_values
        / (1.0 + exp_values)
    )

    return result


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
):
    # Return an empty result if there are no boxes.
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate the area of every box.
    areas = (
        np.maximum(
            0.0,
            x2 - x1,
        )
        * np.maximum(
            0.0,
            y2 - y1,
        )
    )

    # Process boxes from the highest confidence
    # to the lowest confidence.
    order = np.argsort(
        -scores,
        kind="stable",
    )

    keep = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)

        if order.size == 1:
            break

        remaining = order[1:]

        intersection_width = np.maximum(
            0.0,
            np.minimum(
                x2[current],
                x2[remaining],
            )
            - np.maximum(
                x1[current],
                x1[remaining],
            ),
        )

        intersection_height = np.maximum(
            0.0,
            np.minimum(
                y2[current],
                y2[remaining],
            )
            - np.maximum(
                y1[current],
                y1[remaining],
            ),
        )

        intersection = (
            intersection_width
            * intersection_height
        )

        union = (
            areas[current]
            + areas[remaining]
            - intersection
        )

        # IoU = intersection area / union area.
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(
                intersection
            ),
            where=union > 0,
        )

        # Remove boxes whose IoU is greater than
        # the selected threshold.
        order = remaining[
            iou <= iou_threshold
        ]

    return keep


def decode_predictions(
    output: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    confidence_threshold: float,
    nms_iou_threshold: float,
):
    # Change the model output shape from:
    #
    # (1, 25200, 22)
    #
    # to:
    #
    # (25200, 22)
    predictions = np.asarray(
        output,
        dtype=np.float32,
    ).reshape(-1, 22)

    # Column 8 contains the objectness logits.
    confidence = sigmoid(
        predictions[:, 8]
    )

    selected = np.flatnonzero(
        confidence
        >= confidence_threshold
    )

    if selected.size == 0:
        return []

    predictions = predictions[selected]
    confidence = confidence[selected]

    # Columns 0 to 7 contain four (x, y)
    # corner points.
    corners = predictions[
        :,
        :8,
    ].reshape(
        -1,
        4,
        2,
    ).copy()

    # Restore corner coordinates to the
    # original video image.
    corners[:, :, 0] = (
        corners[:, :, 0] - pad_x
    ) / scale

    corners[:, :, 1] = (
        corners[:, :, 1] - pad_y
    ) / scale

    # Columns 9 to 12 contain color logits.
    color_ids = np.argmax(
        predictions[:, 9:13],
        axis=1,
    )

    # Columns 13 to 21 contain pattern logits.
    pattern_ids = np.argmax(
        predictions[:, 13:22],
        axis=1,
    )

    # Build one axis-aligned rectangle around
    # the four corners of each detection.
    boxes = np.column_stack(
        (
            corners[:, :, 0].min(
                axis=1
            ),
            corners[:, :, 1].min(
                axis=1
            ),
            corners[:, :, 0].max(
                axis=1
            ),
            corners[:, :, 1].max(
                axis=1
            ),
        )
    )

    keep = nms(
        boxes,
        confidence,
        nms_iou_threshold,
    )

    detections = []

    for index in keep:
        color_id = int(
            color_ids[index]
        )

        pattern_id = int(
            pattern_ids[index]
        )

        detections.append(
            {
                "color": COLORS[
                    color_id
                ],
                "pattern": PATTERNS[
                    pattern_id
                ],
                "pattern_id": (
                    pattern_id
                ),
                "confidence": float(
                    confidence[index]
                ),
                "corners_px": (
                    corners[index]
                ),
            }
        )

    return detections


def object_corners(pattern_id: int):
    # Select the physical width according to
    # the detected armor pattern.
    if pattern_id in LARGE_PATTERN_IDS:
        width = LARGE_WIDTH_M
    else:
        width = SMALL_WIDTH_M

    half_width = width / 2.0
    half_height = (
        ARMOR_HEIGHT_M / 2.0
    )

    # Corner order:
    # upper-left
    # lower-left
    # lower-right
    # upper-right
    return np.asarray(
        [
            [
                -half_width,
                -half_height,
                0.0,
            ],
            [
                -half_width,
                half_height,
                0.0,
            ],
            [
                half_width,
                half_height,
                0.0,
            ],
            [
                half_width,
                -half_height,
                0.0,
            ],
        ],
        dtype=np.float64,
    )


def estimate_pose(
    image_corners: np.ndarray,
    pattern_id: int,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
):
    points_object = object_corners(
        pattern_id
    )

    points_image = np.asarray(
        image_corners,
        dtype=np.float64,
    ).reshape(4, 2)

    try:
        result = cv2.solvePnPGeneric(
            points_object,
            points_image,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return None, None

    if not result[0]:
        return None, None

    best_error = None
    best_camera_corners = None

    # A planar target may produce two pose candidates.
    for (
        rotation_vector,
        translation_vector,
    ) in zip(
        result[1],
        result[2],
    ):
        rotation_vector = np.asarray(
            rotation_vector,
            dtype=np.float64,
        ).reshape(3, 1)

        translation_vector = np.asarray(
            translation_vector,
            dtype=np.float64,
        ).reshape(3, 1)

        # Improve the IPPE candidate.
        try:
            (
                rotation_vector,
                translation_vector,
            ) = cv2.solvePnPRefineLM(
                points_object,
                points_image,
                camera_matrix,
                distortion,
                rotation_vector,
                translation_vector,
            )
        except cv2.error:
            pass

        rotation_matrix, _ = (
            cv2.Rodrigues(
                rotation_vector
            )
        )

        # Transform every physical corner into
        # the camera coordinate system.
        camera_corners = (
            rotation_matrix
            @ points_object.T
        ).T + translation_vector.reshape(
            1,
            3,
        )

        if not np.isfinite(
            camera_corners
        ).all():
            continue

        # Reject poses behind the camera.
        if np.any(
            camera_corners[:, 2]
            <= 0.0
        ):
            continue

        projected, _ = cv2.projectPoints(
            points_object,
            rotation_vector,
            translation_vector,
            camera_matrix,
            distortion,
        )

        residual = (
            projected.reshape(4, 2)
            - points_image
        )

        # RMS reprojection error for four points.
        error = float(
            math.sqrt(
                float(
                    np.sum(
                        residual
                        * residual
                    )
                )
                / 4.0
            )
        )

        if not math.isfinite(error):
            continue

        if (
            best_error is None
            or error < best_error
        ):
            best_error = error
            best_camera_corners = (
                camera_corners
            )

    if best_camera_corners is None:
        return None, None

    return (
        best_camera_corners.tolist(),
        best_error,
    )


def process_video(args):
    # Load camera calibration and model once.
    camera_matrix, distortion = (
        load_calibration(
            args.calibration
        )
    )

    session = ort.InferenceSession(
        str(args.model),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]

    print(
        "model input:",
        model_input.name,
        model_input.shape,
        model_input.type,
    )

    print(
        "model output:",
        model_output.name,
        model_output.shape,
        model_output.type,
    )

    video = cv2.VideoCapture(
        str(args.video)
    )

    if not video.isOpened():
        raise OSError(
            f"cannot open video: {args.video}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_index = 0
    total_detections = 0

    try:
        with args.output.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            while True:
                if (
                    args.max_frames
                    is not None
                    and frame_index
                    >= args.max_frames
                ):
                    break

                ok, frame = video.read()

                if not ok:
                    break

                (
                    tensor,
                    scale,
                    pad_x,
                    pad_y,
                ) = preprocess(frame)

                raw_output = session.run(
                    [model_output.name],
                    {
                        model_input.name:
                            tensor
                    },
                )[0]

                decoded = (
                    decode_predictions(
                        raw_output,
                        scale,
                        pad_x,
                        pad_y,
                        args.confidence,
                        args.nms_iou,
                    )
                )

                detections = []

                for detection in decoded:
                    (
                        camera_corners,
                        reprojection_error,
                    ) = estimate_pose(
                        detection[
                            "corners_px"
                        ],
                        detection[
                            "pattern_id"
                        ],
                        camera_matrix,
                        distortion,
                    )

                    detections.append(
                        {
                            "color": detection[
                                "color"
                            ],
                            "pattern": detection[
                                "pattern"
                            ],
                            "confidence": (
                                detection[
                                    "confidence"
                                ]
                            ),
                            "corners_px": (
                                detection[
                                    "corners_px"
                                ].tolist()
                            ),
                            "corners_camera_m": (
                                camera_corners
                            ),
                            "reprojection_error_px": (
                                reprojection_error
                            ),
                        }
                    )

                record = {
                    "frame_index": (
                        frame_index
                    ),
                    "timestamp_s": (
                        frame_index / 30.0
                    ),
                    "detections": (
                        detections
                    ),
                }

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )

                total_detections += len(
                    detections
                )

                frame_index += 1

                if frame_index % 30 == 0:
                    print(
                        "processed frames:",
                        frame_index,
                    )

    finally:
        video.release()

    print(
        "finished:",
        frame_index,
        "frames,",
        total_detections,
        "detections",
    )

    print(
        "output:",
        args.output,
    )


def main():
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()