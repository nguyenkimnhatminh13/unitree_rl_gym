from collections import deque
from dataclasses import dataclass
import importlib
from typing import Optional, Union

import numpy as np
import time
import torch
import yaml
import os

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import MotorMode, create_damping_cmd, create_zero_cmd, init_cmd_hg
from common.remote_controller import KeyMap, RemoteController


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def detect_policy_type(policy: torch.jit.ScriptModule) -> str:
    try:
        state_dict = policy.state_dict()
        if "student_encoder.embed.weight" in state_dict:
            print("Detected distillation policy: student encoder found.")
            return "distillation"
        if "actor_body.0.base.weight" in state_dict:
            print("Detected distillation policy: FiLM actor body found.")
            return "distillation"
        if (
            "frozen_actor.0.weight" in state_dict
            or "residual_adapter.residual_mlp.0.weight" in state_dict
        ):
            print("Detected distillation policy: residual/frozen actor found.")
            return "distillation"
    except Exception:
        pass

    try:
        code = policy.code
        if "student_encoder_obs" in code or "policy_obs" in code:
            print("Detected distillation policy: dual input signature found.")
            return "distillation"
    except Exception:
        pass

    try:
        graph = policy.graph
        if "student_encoder_obs" in str(graph):
            print("Detected distillation policy: encoder input found in graph.")
            return "distillation"
        if len(list(graph.inputs())) > 2:
            print("Detected distillation policy: graph has >2 inputs.")
            return "distillation"
    except Exception:
        pass

    print("Detected standard policy.")
    return "standard"


def detect_encoder_obs_size(policy: torch.jit.ScriptModule) -> int:
    try:
        state_dict = policy.state_dict()
        if "student_encoder.embed.weight" in state_dict:
            encoder_obs_dim = int(state_dict["student_encoder.embed.weight"].shape[1])
            print(f"Auto-detected encoder obs dim: {encoder_obs_dim}")
            return encoder_obs_dim
    except Exception:
        pass

    fallback = 96 + 7
    print(f"Using fallback encoder obs dim: {fallback}")
    return fallback


def build_student_encoder_obs(
    omega_normalized: np.ndarray,
    gravity_orientation: np.ndarray,
    cmd: np.ndarray,
    qj_policy: np.ndarray,
    dqj_policy: np.ndarray,
    action_policy: np.ndarray,
    cmd_scale: np.ndarray,
    object_obs: np.ndarray,
) -> np.ndarray:
    num_actions = qj_policy.shape[0]
    obs = np.zeros(3 + 3 + 3 + num_actions + num_actions + num_actions + 7, dtype=np.float32)
    idx = 0
    obs[idx : idx + 3] = omega_normalized
    idx += 3
    obs[idx : idx + 3] = gravity_orientation
    idx += 3
    obs[idx : idx + 3] = cmd * cmd_scale
    idx += 3
    obs[idx : idx + num_actions] = qj_policy
    idx += num_actions
    obs[idx : idx + num_actions] = dqj_policy
    idx += num_actions
    obs[idx : idx + num_actions] = action_policy
    idx += num_actions
    obs[idx : idx + 7] = object_obs[:7]
    return obs


def resize_encoder_obs(obs: np.ndarray, target_dim: int) -> np.ndarray:
    if obs.shape[0] == target_dim:
        return obs
    resized = np.zeros(target_dim, dtype=np.float32)
    n = min(target_dim, obs.shape[0])
    resized[:n] = obs[:n]
    return resized


def _safe_getattr_chain(obj, names):
    cur = obj
    for name in names:
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur


def normalize_quat_to_wxyz(quat: np.ndarray, quat_order: str) -> np.ndarray:
    if quat_order == "wxyz":
        qw, qx, qy, qz = quat
        return np.array([qw, qx, qy, qz], dtype=np.float32)
    if quat_order == "xyzw":
        qx, qy, qz, qw = quat
        return np.array([qw, qx, qy, qz], dtype=np.float32)
    raise ValueError(f"Unsupported quat_order: {quat_order}")


def quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat_wxyz
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)
    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)
    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    return np.array(
        [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]],
        dtype=np.float32,
    )


def build_object_obs(
    pos: np.ndarray,
    quat_raw: np.ndarray,
    quat_order: str,
    pose_is_top_surface: bool,
    object_half_height: float,
) -> np.ndarray:
    quat_wxyz = normalize_quat_to_wxyz(quat_raw, quat_order=quat_order)
    if pose_is_top_surface:
        pos_top = pos
    else:
        # Match get_object_pose(): position should represent object top surface.
        rot = quat_wxyz_to_rotmat(quat_wxyz)
        pos_top = pos + rot @ np.array([0.0, 0.0, object_half_height], dtype=np.float32)
    return np.concatenate([pos_top.astype(np.float32), quat_wxyz.astype(np.float32)], axis=0)


def extract_object_pose_raw_from_msg(msg) -> Optional[np.ndarray]:
    # Pattern 1: flat array in msg.data, expected [x, y, z, q*, q*, q*, q*]
    if hasattr(msg, "data"):
        data = np.array(getattr(msg, "data"), dtype=np.float32).reshape(-1)
        if data.shape[0] >= 7:
            return data[:7]

    # Pattern 2: geometry pose-like message (ROS-style orientation is xyzw)
    px = _safe_getattr_chain(msg, ["pose", "position", "x"])
    py = _safe_getattr_chain(msg, ["pose", "position", "y"])
    pz = _safe_getattr_chain(msg, ["pose", "position", "z"])
    qx = _safe_getattr_chain(msg, ["pose", "orientation", "x"])
    qy = _safe_getattr_chain(msg, ["pose", "orientation", "y"])
    qz = _safe_getattr_chain(msg, ["pose", "orientation", "z"])
    qw = _safe_getattr_chain(msg, ["pose", "orientation", "w"])
    if None not in (px, py, pz, qx, qy, qz, qw):
        return np.array([px, py, pz, qx, qy, qz, qw], dtype=np.float32)

    # Pattern 3: direct fields on message
    px = getattr(msg, "x", None)
    py = getattr(msg, "y", None)
    pz = getattr(msg, "z", None)
    qx = getattr(msg, "qx", None)
    qy = getattr(msg, "qy", None)
    qz = getattr(msg, "qz", None)
    qw = getattr(msg, "qw", None)
    if None not in (px, py, pz, qx, qy, qz, qw):
        return np.array([px, py, pz, qx, qy, qz, qw], dtype=np.float32)

    return None


def import_msg_type(msg_type_path: str):
    module_name, attr_name = msg_type_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


@dataclass
class SteadyTrayConfig:
    control_dt: float
    msg_type: str
    imu_type: str
    lowcmd_topic: str
    lowstate_topic: str
    policy_path: str
    max_cmd: np.ndarray
    num_actions: int
    num_obs: int
    cmd_scale: np.ndarray
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    action_scale: np.ndarray
    default_angles: np.ndarray
    kps: np.ndarray
    kds: np.ndarray
    policy_to_robot: np.ndarray
    robot_to_policy: np.ndarray
    control_decimation: int
    encoder_seq_len: int
    distill_object_obs: np.ndarray
    object_pose_topic: str
    object_pose_msg_type: str
    object_pose_timeout_s: float
    object_pose_quat_order: str
    object_pose_is_top_surface: bool
    object_half_height: float

    @classmethod
    def from_yaml(cls, file_path: str) -> "SteadyTrayConfig":
        with open(file_path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cls(
            control_dt=cfg["control_dt"],
            msg_type=cfg["msg_type"],
            imu_type=cfg.get("imu_type", "pelvis"),
            lowcmd_topic=cfg["lowcmd_topic"],
            lowstate_topic=cfg["lowstate_topic"],
            policy_path=cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR),
            max_cmd=np.array(cfg["max_cmd"], dtype=np.float32),
            num_actions=int(cfg["num_actions"]),
            num_obs=int(cfg["num_obs"]),
            cmd_scale=np.array(cfg["cmd_scale"], dtype=np.float32),
            ang_vel_scale=float(cfg["ang_vel_scale"]),
            dof_pos_scale=float(cfg["dof_pos_scale"]),
            dof_vel_scale=float(cfg["dof_vel_scale"]),
            action_scale=np.array(cfg["action_scale"], dtype=np.float32),
            default_angles=np.array(cfg["default_angles"], dtype=np.float32),
            kps=np.array(cfg["kps"], dtype=np.float32),
            kds=np.array(cfg["kds"], dtype=np.float32),
            policy_to_robot=np.array(cfg["policy_to_robot"], dtype=np.int32),
            robot_to_policy=np.array(cfg["robot_to_policy"], dtype=np.int32),
            control_decimation=int(cfg.get("control_decimation", 1)),
            encoder_seq_len=int(cfg.get("encoder_seq_len", 32)),
            distill_object_obs=np.array(cfg.get("distill_object_obs", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), dtype=np.float32),
            object_pose_topic=str(cfg.get("object_pose_topic", "/object_pose")),
            object_pose_msg_type=str(cfg.get("object_pose_msg_type", "")),
            object_pose_timeout_s=float(cfg.get("object_pose_timeout_s", 0.5)),
            object_pose_quat_order=str(cfg.get("object_pose_quat_order", "xyzw")),
            object_pose_is_top_surface=bool(cfg.get("object_pose_is_top_surface", False)),
            object_half_height=float(cfg.get("object_half_height", 0.05)),
        )


class Controller:
    def __init__(self, config: SteadyTrayConfig) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        self.policy = torch.jit.load(config.policy_path)
        self.policy_type = detect_policy_type(self.policy)
        self.encoder_obs_dim: Optional[int] = None
        if self.policy_type == "distillation":
            self.encoder_obs_dim = detect_encoder_obs_size(self.policy)
            print("Using fixed object observation for real deployment.")
        self.counter = 0

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        self.object_pose_subscriber = None
        self.latest_object_obs = config.distill_object_obs.copy()
        self.last_object_obs_time = 0.0

        self.wait_for_low_state()
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action_robot = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles[config.policy_to_robot].copy()
        self.frame_stack = deque(maxlen=5)
        for _ in range(5):
            self.frame_stack.append(np.zeros(config.num_obs, dtype=np.float32))
        self.encoder_frame_stack: Optional[deque] = None
        if self.encoder_obs_dim is not None:
            self.encoder_frame_stack = deque(maxlen=config.encoder_seq_len)
            for _ in range(config.encoder_seq_len):
                self.encoder_frame_stack.append(np.zeros(self.encoder_obs_dim, dtype=np.float32))
            self.try_init_object_pose_subscriber()

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def ObjectPoseHandler(self, msg):
        raw = extract_object_pose_raw_from_msg(msg)
        if raw is None:
            return
        pos = raw[:3]
        quat_raw = raw[3:7]
        try:
            object_obs = build_object_obs(
                pos=pos,
                quat_raw=quat_raw,
                quat_order=self.config.object_pose_quat_order,
                pose_is_top_surface=self.config.object_pose_is_top_surface,
                object_half_height=self.config.object_half_height,
            )
        except Exception:
            return
        self.latest_object_obs = object_obs
        self.last_object_obs_time = time.time()

    def send_cmd(self, cmd: Union[LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def try_init_object_pose_subscriber(self) -> None:
        topic = self.config.object_pose_topic
        if not topic:
            print("Object pose topic is empty; using default distill_object_obs.")
            return

        msg_type_paths = []
        if self.config.object_pose_msg_type:
            msg_type_paths.append(self.config.object_pose_msg_type)
        msg_type_paths.extend(
            [
                "unitree_sdk2py.idl.geometry_msgs.msg.dds_.PoseStamped_",
                "unitree_sdk2py.idl.geometry_msgs.msg.dds_.Pose_",
                "unitree_sdk2py.idl.std_msgs.msg.dds_.Float32MultiArray_",
                "unitree_sdk2py.idl.std_msgs.msg.dds_.Float64MultiArray_",
            ]
        )

        for msg_type_path in msg_type_paths:
            try:
                msg_type = import_msg_type(msg_type_path)
                sub = ChannelSubscriber(topic, msg_type)
                sub.Init(self.ObjectPoseHandler, 10)
                self.object_pose_subscriber = sub
                print(f"Subscribed object pose topic '{topic}' with type '{msg_type_path}'.")
                return
            except Exception:
                continue

        print(
            f"Unable to subscribe '{topic}'. Falling back to distill_object_obs from config. "
            "Set object_pose_msg_type for your custom message type."
        )

    def get_current_object_obs(self) -> np.ndarray:
        # Use live object pose only when recent; otherwise use configured fallback.
        if self.last_object_obs_time > 0.0:
            if (time.time() - self.last_object_obs_time) <= self.config.object_pose_timeout_s:
                return self.latest_object_obs
        return self.config.distill_object_obs

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default position.")
        total_time = 2.0
        num_step = int(total_time / self.config.control_dt)
        default_pos = self.config.default_angles[self.config.policy_to_robot]
        init_dof_pos = np.zeros(self.config.num_actions, dtype=np.float32)
        for i in range(self.config.num_actions):
            init_dof_pos[i] = self.low_state.motor_state[i].q

        for i in range(num_step):
            alpha = i / num_step
            for motor_idx in range(self.config.num_actions):
                q_target = init_dof_pos[motor_idx] * (1 - alpha) + default_pos[motor_idx] * alpha
                self.low_cmd.motor_cmd[motor_idx].q = q_target
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default position state.")
        print("Waiting for Button A signal...")
        default_pos = self.config.default_angles[self.config.policy_to_robot]
        while self.remote_controller.button[KeyMap.A] != 1:
            for motor_idx in range(self.config.num_actions):
                self.low_cmd.motor_cmd[motor_idx].q = default_pos[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def build_stacked_obs(self, omega: np.ndarray, gravity: np.ndarray, cmd: np.ndarray) -> np.ndarray:
        cfg = self.config
        qj_policy = (self.qj - cfg.default_angles[cfg.policy_to_robot]) * cfg.dof_pos_scale
        dqj_policy = self.dqj * cfg.dof_vel_scale
        action_policy = self.action_robot[cfg.robot_to_policy]

        obs = np.zeros(cfg.num_obs, dtype=np.float32)
        obs[:3] = omega * cfg.ang_vel_scale
        obs[3:6] = gravity
        obs[6:9] = cmd * cfg.cmd_scale
        obs[9 : 9 + cfg.num_actions] = qj_policy[cfg.robot_to_policy]
        obs[9 + cfg.num_actions : 9 + 2 * cfg.num_actions] = dqj_policy[cfg.robot_to_policy]
        obs[9 + 2 * cfg.num_actions : 9 + 3 * cfg.num_actions] = action_policy
        self.frame_stack.append(obs)

        stacked = np.array(self.frame_stack, dtype=np.float32)
        obs_omega = stacked[:, :3].ravel()
        obs_gravity = stacked[:, 3:6].ravel()
        obs_cmd = stacked[:, 6:9].ravel()
        obs_pos = stacked[:, 9 : 9 + cfg.num_actions].ravel()
        obs_vel = stacked[:, 9 + cfg.num_actions : 9 + 2 * cfg.num_actions].ravel()
        obs_action = stacked[:, 9 + 2 * cfg.num_actions : 9 + 3 * cfg.num_actions].ravel()
        return np.concatenate([obs_omega, obs_gravity, obs_cmd, obs_pos, obs_vel, obs_action], axis=0)

    def run(self):
        self.counter += 1
        cfg = self.config

        for i in range(cfg.num_actions):
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq

        quat = np.array(self.low_state.imu_state.quaternion, dtype=np.float32)
        omega = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        gravity = get_gravity_orientation(quat)

        cmd = np.array(
            [
                self.remote_controller.ly * cfg.max_cmd[0],
                -self.remote_controller.lx * cfg.max_cmd[1],
                -self.remote_controller.rx * cfg.max_cmd[2],
            ],
            dtype=np.float32,
        )

        if self.counter % cfg.control_decimation != 0:
            # Keep streaming latest target at high-rate while policy runs at decimated rate.
            for motor_idx in range(cfg.num_actions):
                self.low_cmd.motor_cmd[motor_idx].q = float(self.target_dof_pos[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = float(cfg.kps[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].kd = float(cfg.kds[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(cfg.control_dt)
            return

        stacked_obs = self.build_stacked_obs(omega=omega, gravity=gravity, cmd=cmd)
        with torch.no_grad():
            if self.policy_type == "distillation":
                if self.encoder_frame_stack is None:
                    raise RuntimeError("encoder_frame_stack is not initialized for distillation policy.")
                omega_norm = omega * cfg.ang_vel_scale
                qj_norm = (self.qj - cfg.default_angles[cfg.policy_to_robot]) * cfg.dof_pos_scale
                dqj_norm = self.dqj * cfg.dof_vel_scale
                qj_policy = qj_norm[cfg.robot_to_policy]
                dqj_policy = dqj_norm[cfg.robot_to_policy]
                action_policy_prev = self.action_robot[cfg.robot_to_policy]
                encoder_obs = build_student_encoder_obs(
                    omega_normalized=omega_norm,
                    gravity_orientation=gravity,
                    cmd=cmd,
                    qj_policy=qj_policy,
                    dqj_policy=dqj_policy,
                    action_policy=action_policy_prev,
                    cmd_scale=cfg.cmd_scale,
                    object_obs=self.get_current_object_obs(),
                )
                if self.encoder_obs_dim is not None:
                    encoder_obs = resize_encoder_obs(encoder_obs, self.encoder_obs_dim)
                self.encoder_frame_stack.append(encoder_obs)
                encoder_tensor = torch.from_numpy(np.array(self.encoder_frame_stack, dtype=np.float32)).unsqueeze(1)
                policy_obs_tensor = torch.from_numpy(stacked_obs).unsqueeze(0)
                action_policy = self.policy(encoder_tensor, policy_obs_tensor).detach().cpu().numpy().squeeze()
            else:
                obs_tensor = torch.from_numpy(stacked_obs).unsqueeze(0)
                action_policy = self.policy(obs_tensor).detach().cpu().numpy().squeeze()
        if action_policy.shape[0] != cfg.num_actions:
            raise RuntimeError(
                f"Policy action dim mismatch, expected {cfg.num_actions}, got {action_policy.shape[0]}"
            )

        self.action_robot = action_policy[cfg.policy_to_robot]
        self.target_dof_pos = cfg.default_angles[cfg.policy_to_robot] + self.action_robot * cfg.action_scale[cfg.policy_to_robot]

        for motor_idx in range(cfg.num_actions):
            self.low_cmd.motor_cmd[motor_idx].q = float(self.target_dof_pos[motor_idx])
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = float(cfg.kps[motor_idx])
            self.low_cmd.motor_cmd[motor_idx].kd = float(cfg.kds[motor_idx])
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        self.send_cmd(self.low_cmd)
        time.sleep(cfg.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument(
        "--config",
        type=str,
        default=f"{LEGGED_GYM_ROOT_DIR}/deploy_real/configs/g1_steadytray.yaml",
        help="SteadyTray deployment config file path",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="",
        help="Optional override for policy path",
    )
    args = parser.parse_args()

    config = SteadyTrayConfig.from_yaml(args.config)
    if args.policy:
        config.policy_path = args.policy

    ChannelFactoryInitialize(0, args.net)
    controller = Controller(config)

    controller.zero_torque_state()
    controller.move_to_default_pos()
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break

    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
