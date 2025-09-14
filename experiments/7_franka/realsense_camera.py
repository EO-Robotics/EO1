from typing import Dict, List, Optional, Tuple

import numpy as np
import pyrealsense2 as rs

# the camera intrinsics from the camera calibration
serial_number_to_cam_intr = {
    "243222074139": {
        "fx": 604.987548828125,
        "fy": 604.9332885742188,
        "px": 325.9312438964844,
        "py": 243.00851440429688,
    },  # eye-to-hand
    "213522070137": {
        "fx": 596.382688,
        "fy": 596.701788,
        "px": 333.837001,
        "py": 254.401211,
    },  # eye-in-hand
}


class Camera:
    def __init__(
        self,
        serial_number: Optional[str] = "213522070137",
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
    ) -> None:
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.pipeline = rs.pipeline()
        # self.pipeline.wait_for_frames(9999)
        self.config = rs.config()

        if serial_number is not None:
            self.config.enable_device(serial_number)
        self.config.enable_stream(
            rs.stream.depth,
            self.camera_width,
            self.camera_height,
            rs.format.z16,
            self.camera_fps,
        )
        self.config.enable_stream(
            rs.stream.color,
            self.camera_width,
            self.camera_height,
            rs.format.bgr8,
            self.camera_fps,
        )

        self.profile = self.pipeline.start(self.config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale: float = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", self.depth_scale)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def close(self):
        self.pipeline.stop()

    def get_frame(self, filter=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            filter bool optional Whether to apply filters to depth frames. Defaults to True.
        Returns:
            color_image np.ndarray shape=(H, W, 3) Color image(BGR)
            depth_image np.ndarray shape=(H, W) Depth image in meters
        """

        depth_to_disparity = rs.disparity_transform(True)

        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)

        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        temporal.set_option(rs.option.filter_smooth_delta, 20)

        disparity_to_depth = rs.disparity_transform(False)

        hole_filling = rs.hole_filling_filter()

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Apply the filters to depth frames
        if filter:
            filtered_depth = depth_to_disparity.process(depth_frame)
            filtered_depth = spatial.process(filtered_depth)
            filtered_depth = temporal.process(filtered_depth)
            filtered_depth = disparity_to_depth.process(filtered_depth)
            filtered_depth = hole_filling.process(filtered_depth)

            depth_image = np.asanyarray(filtered_depth.get_data())
            color_image = np.asanyarray(color_frame.get_data())
        else:
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image * self.depth_scale

    def get_camera_intrinsics(self, use_raw=False, serial_number: Optional[str] = "213522070137"):
        """
        Args:
            use_raw bool optional Whether to use the camera intrinsics from the raw stream. Defaults to False.
        Returns:
            {
                "fx": Focal length x,
                "fy": Focal length y,
                "px": Principal point x,
                "py": Principal point y
            }
        """
        if use_raw:
            profile = self.profile.get_stream(rs.stream.color)
            intr = profile.as_video_stream_profile().get_intrinsics()
            return {"px": intr.ppx, "py": intr.ppy, "fx": intr.fx, "fy": intr.fy}
        return serial_number_to_cam_intr[serial_number]


class MultiCamera:
    def __init__(
        self,
        serial_numbers: Optional[List[str]] = None,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
    ) -> None:
        all_serial_numbers = []
        for d in sorted(rs.context().devices, key=lambda x: x.get_info(rs.camera_info.serial_number)):
            print(
                "Found device: ",
                d.get_info(rs.camera_info.name),
                " ",
                d.get_info(rs.camera_info.serial_number),
            )
            if d.get_info(rs.camera_info.name).lower() != "platform camera":
                all_serial_numbers.append(d.get_info(rs.camera_info.serial_number))

        if serial_numbers is None:
            serial_numbers = all_serial_numbers
        else:
            serial_numbers = [
                serial_number for serial_number in serial_numbers if serial_number in all_serial_numbers
            ]

        print("Using cameras with serial numbers: ", serial_numbers)

        self.cameras = {
            serial_number: Camera(serial_number, camera_width, camera_height, camera_fps)
            for serial_number in serial_numbers
        }

        for _ in range(20):
            print("filter")
            self.get_frame()

    def close(self):
        for camera in self.cameras.values():
            camera.close()

    def get_frame(
        self, serial_numbers: Optional[List[str]] = None, filter=True
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if serial_numbers is None:
            serial_numbers = list(self.cameras.keys())
        return {serial_number: self.cameras[serial_number].get_frame() for serial_number in serial_numbers}

    def get_camera_intrinsics(
        self, serial_numbers: Optional[List[str]] = None, use_raw=False
    ) -> Dict[str, Dict[str, float]]:
        if serial_numbers is None:
            serial_numbers = self.cameras.keys()
        return {
            serial_number: self.cameras[serial_number].get_camera_intrinsics(use_raw, serial_number)
            for serial_number in serial_numbers
        }
