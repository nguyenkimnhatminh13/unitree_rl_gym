from collections import deque
from dataclasses import dataclass
import logging
from typing import Any, Optional, Union

import numpy as np
import time
import torch
import yaml
import os
try:
    import onnxruntime as ort
except Exception:
    ort = None

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC
try:
    from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_ as DdsObjectPoseStamped_
except Exception:
    DdsObjectPoseStamped_ = None

from common.command_helper import MotorMode, create_damping_cmd, create_zero_cmd, init_cmd_hg
from common.remote_controller import KeyMap, RemoteController

logger = logging.getLogger(__name__)


def keymap_index_from_name(name: str) -> int:
    """Map a short name (e.g. 'B', 'X') to RemoteController button index."""
    m = (name or "B").strip().upper()
    table = {
        "A": KeyMap.A,
        "B": KeyMap.B,
        "X": KeyMap.X,
        "Y": KeyMap.Y,
        "R1": KeyMap.R1,
        "L1": KeyMap.L1,
        "R2": KeyMap.R2,
        "L2": KeyMap.L2,
        "START": KeyMap.start,
        "SELECT": KeyMap.select,
        "F1": KeyMap.F1,
        "F2": KeyMap.F2,
        "UP": KeyMap.up,
        "DOWN": KeyMap.down,
        "LEFT": KeyMap.left,
        "RIGHT": KeyMap.right,
    }
    if m not in table:
        raise ValueError(
            f"Unknown key name {name!r}. Use one of: {', '.join(sorted(table))}."
        )
    return table[m]


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


def load_policy_model(path: str):
    """
    Load a policy from TorchScript (.pt) or ONNX (.onnx).
    Returns: (model, backend, policy_type)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".onnx":
        if ort is None:
            raise RuntimeError(
                "ONNX policy requested but onnxruntime is not available. "
                "Install it with: pip install onnxruntime"
            )
        model = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_count = len(model.get_inputs())
        policy_type = "distillation" if input_count >= 2 else "standard"
        logger.info(
            "Loaded ONNX policy: path=%s inputs=%d policy_type=%s",
            path,
            input_count,
            policy_type,
        )
        return model, "onnx", policy_type

    model = torch.jit.load(path)
    policy_type = detect_policy_type(model)
    return model, "torch", policy_type


def detect_encoder_obs_size_from_model(model: Any, backend: str) -> int:
    if backend == "torch":
        return detect_encoder_obs_size(model)
    # ONNX distillation policy: expect encoder obs as first input
    try:
        shape = model.get_inputs()[0].shape
        if shape and isinstance(shape[-1], int) and shape[-1] > 0:
            dim = int(shape[-1])
            print(f"Auto-detected ONNX encoder obs dim: {dim}")
            return dim
    except Exception:
        pass
    fallback = 96 + 7
    print(f"Using fallback ONNX encoder obs dim: {fallback}")
    return fallback


def run_policy_inference(
    model: Any,
    backend: str,
    distillation: bool,
    policy_obs: np.ndarray,
    encoder_obs_seq: Optional[np.ndarray] = None,
) -> np.ndarray:
    if backend == "torch":
        with torch.no_grad():
            if distillation:
                if encoder_obs_seq is None:
                    raise RuntimeError("encoder_obs_seq is required for distillation policy.")
                encoder_tensor = torch.from_numpy(encoder_obs_seq).unsqueeze(1)
                policy_obs_tensor = torch.from_numpy(policy_obs).unsqueeze(0)
                out = model(encoder_tensor, policy_obs_tensor).detach().cpu().numpy().squeeze()
            else:
                policy_obs_tensor = torch.from_numpy(policy_obs).unsqueeze(0)
                out = model(policy_obs_tensor).detach().cpu().numpy().squeeze()
        return np.asarray(out, dtype=np.float32)

    # ONNX
    if distillation:
        if encoder_obs_seq is None:
            raise RuntimeError("encoder_obs_seq is required for distillation policy.")
        input_defs = model.get_inputs()
        if len(input_defs) < 2:
            raise RuntimeError("Distillation ONNX policy expects at least 2 inputs.")
        encoder_input = encoder_obs_seq.astype(np.float32)
        expected_rank = len(input_defs[0].shape)
        # Torch path feeds [seq, 1, obs_dim] for encoder input.
        # If exported ONNX expects rank-3 but we currently have rank-2 [seq, obs_dim],
        # insert singleton axis at dim=1 to match [seq, 1, obs_dim].
        if expected_rank == encoder_input.ndim + 1:
            encoder_input = np.expand_dims(encoder_input, axis=1)
        elif expected_rank != encoder_input.ndim:
            raise RuntimeError(
                f"Unexpected ONNX encoder input rank: got {encoder_input.ndim}, "
                f"expected {expected_rank} for input '{input_defs[0].name}' "
                f"with shape spec {input_defs[0].shape}"
            )
        feed = {
            input_defs[0].name: encoder_input,
            input_defs[1].name: policy_obs.astype(np.float32)[None, :],
        }
    else:
        input_name = model.get_inputs()[0].name
        feed = {input_name: policy_obs.astype(np.float32)[None, :]}
    out = model.run(None, feed)[0]
    return np.asarray(out, dtype=np.float32).squeeze()


# def build_student_encoder_obs(
#     omega_normalized: np.ndarray,
#     gravity_orientation: np.ndarray,
#     cmd: np.ndarray,
#     qj_policy: np.ndarray,
#     dqj_policy: np.ndarray,
#     action_policy: np.ndarray,
#     cmd_scale: np.ndarray,
#     object_obs: np.ndarray,
# ) -> np.ndarray:
#     num_actions = qj_policy.shape[0]
#     obs = np.zeros(3 + 3 + 3 + num_actions + num_actions + num_actions + 7, dtype=np.float32)
#     idx = 0
#     obs[idx : idx + 3] = omega_normalized
#     idx += 3
#     obs[idx : idx + 3] = gravity_orientation
#     idx += 3
#     obs[idx : idx + 3] = cmd * cmd_scale
#     idx += 3
#     obs[idx : idx + num_actions] = qj_policy
#     idx += num_actions
#     obs[idx : idx + num_actions] = dqj_policy
#     idx += num_actions
#     obs[idx : idx + num_actions] = action_policy
#     idx += num_actions
#     obs[idx : idx + 7] = object_obs[:7]
#     return obs
def build_student_encoder_obs(
    omega: np.ndarray,
    gravity_orientation: np.ndarray,
    cmd: np.ndarray,
    qj: np.ndarray,
    dqj: np.ndarray,
    action: np.ndarray,
    cmd_scale: np.ndarray,
    num_actions: int,
    object_obs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Build student encoder observations for a single timestep.
    
    For distillation policies, the student encoder receives observations that match
    the StudentEncoderCfg structure:
    - base_ang_vel (3)
    - projected_gravity (3)
    - velocity_commands (3)
    - joint_pos_rel (29)
    - joint_vel_rel (29)
    - last_action (29)
    - object_pos_cam (3) - object position
    - object_quat_cam (4) - object quaternion
    
    Total: 96 + 7 = 103 dimensions per timestep.
    
    Args:
        omega: Angular velocity (3,) - normalized (corresponds to base_ang_vel)
        gravity_orientation: Gravity vector in body frame (3,) (corresponds to projected_gravity)
        cmd: Command values (3,) - typically [vel_x, vel_y, yaw_rate] (corresponds to velocity_commands)
        qj: Joint positions (num_actions,) - normalized, policy order (corresponds to joint_pos_rel)
        dqj: Joint velocities (num_actions,) - normalized, policy order (corresponds to joint_vel_rel)
        action: Previous action (num_actions,) - policy order (corresponds to last_action)
        cmd_scale: Scale factors for commands (3,)
        num_actions: Number of actions (29 for G1)
        object_obs: Object observations (7,)
                   Format: [pos(3), quat(4)] = position + quaternion
                   Must be provided for distillation policies.
        
    Returns:
        student_encoder_obs: Student encoder observations for one timestep
                            Shape: (103,) = 96 base + 7 object
    """
    # Base proprioceptive observations: omega(3) + gravity(3) + cmd(3) + qj(29) + dqj(29) + action(29) = 96
    base_obs_dim = 3 + 3 + 3 + num_actions + num_actions + num_actions
    # Determine object observation size
    if object_obs is not None and len(object_obs) > 0:
        object_obs_size = len(object_obs)
        # Ensure we have at least position (3 dims)
        if object_obs_size < 3:
            print(f"Warning: object_obs size {object_obs_size} < 3, padding with zeros")
            object_data = np.zeros(6, dtype=np.float32)  # Default to 6 dims
            object_data[:object_obs_size] = object_obs
            object_obs_size = 6
        else:
            object_data = object_obs.astype(np.float32)
    else:
        raise ValueError("object_obs must be provided for distillation policies")
    # Total dimension per timestep: base_obs + object_obs_size
    total_dim = base_obs_dim + object_obs_size
    student_obs = np.zeros(total_dim, dtype=np.float32)
    # Fill in observations following StudentEncoderCfg order
    idx = 0
    # base_ang_vel (3)
    student_obs[idx:idx+3] = omega
    idx += 3
    # projected_gravity (3)
    student_obs[idx:idx+3] = gravity_orientation
    idx += 3
    # velocity_commands (3)
    student_obs[idx:idx+3] = cmd * cmd_scale
    idx += 3
    # joint_pos_rel (29)
    student_obs[idx:idx+num_actions] = qj
    idx += num_actions
    # joint_vel_rel (29)
    student_obs[idx:idx+num_actions] = dqj
    idx += num_actions
    # last_action (29)
    student_obs[idx:idx+num_actions] = action
    idx += num_actions
    # object_pos_cam (first 3 dims) + object_quat_cam (remaining 4 dims)
    student_obs[idx:idx+3] = object_data[:3]
    idx += 3
    student_obs[idx:idx+4] = object_data[3:7]
    idx += 4
    return student_obs

def resize_encoder_obs(obs: np.ndarray, target_dim: int) -> np.ndarray:
    if obs.shape[0] == target_dim:
        return obs
    resized = np.zeros(target_dim, dtype=np.float32)
    n = min(target_dim, obs.shape[0])
    resized[:n] = obs[:n]
    return resized


def object_obs_from_pose_stamped(msg) -> Optional[np.ndarray]:
    """
    Parse DDS geometry_msgs/PoseStamped and return [x, y, z, qw, qx, qy, qz].
    Incoming PoseStamped orientation is ROS order [qx, qy, qz, qw].
    """
    try:
        p = msg.pose.position
        o = msg.pose.orientation
        qx, qy, qz, qw = float(o.x), float(o.y), float(o.z), float(o.w)
        return np.array([float(p.x), float(p.y), float(p.z), qw, qx, qy, qz], dtype=np.float32)
        # return np.array([float(0.32), float(-0.0175), float(p.z), 0.9, 0.0, -0.4035, 0.0], dtype=np.float32)
    except Exception:
        return None


@dataclass
class SteadyTrayConfig:
    control_dt: float
    msg_type: str
    imu_type: str
    lowcmd_topic: str
    lowstate_topic: str
    policy_path: str
    policy_path_stable: str
    key_start_stable: str
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
    object_pose_topic: str
    object_pose_timeout_s: float

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
            policy_path_stable=str(
                (cfg.get("policy_path_stable") or "")
            ).replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR),
            key_start_stable=str(cfg.get("key_start_stable", "B")),
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
            object_pose_topic=str(cfg.get("object_pose_topic", "rt/object_pose")),
            object_pose_timeout_s=float(cfg.get("object_pose_timeout_s", 0.5)),
        )


class Controller:
    def __init__(self, config: SteadyTrayConfig) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        self.policy, self.policy_backend, self.policy_type = load_policy_model(config.policy_path)
        self.encoder_obs_dim: Optional[int] = None
        if self.policy_type == "distillation":
            self.encoder_obs_dim = detect_encoder_obs_size_from_model(self.policy, self.policy_backend)
            print("Distillation policy: object_obs comes from object_pose_topic (no YAML fallback).")

        self.policy_stable: Optional[Any] = None
        self.policy_backend_stable: Optional[str] = None
        self.policy_type_stable: Optional[str] = None
        self.encoder_obs_dim_stable: Optional[int] = None
        p_stable = (config.policy_path_stable or "").strip()
        if p_stable:
            self.policy_stable, self.policy_backend_stable, self.policy_type_stable = load_policy_model(p_stable)
            if self.policy_type_stable == "distillation":
                self.encoder_obs_dim_stable = detect_encoder_obs_size_from_model(
                    self.policy_stable, self.policy_backend_stable
                )
            logger.info(
                "init: policy_path_stable=%s policy_type=%s backend=%s",
                p_stable,
                self.policy_type_stable,
                self.policy_backend_stable,
            )
        self._use_stable_policy = False
        self._key_start_stable = keymap_index_from_name(config.key_start_stable)
        self._prev_a_for_policy_switch = 0
        self._prev_stable_key_for_policy_switch = 0

        self.counter = 0
        self._policy_loop_started = False

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0
        self._lowstate_ready = False

        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        self.object_pose_subscriber = None
        needs_object = (self.encoder_obs_dim is not None) or (self.encoder_obs_dim_stable is not None)
        if needs_object:
            topic = (config.object_pose_topic or "").strip()
            if not topic:
                raise RuntimeError(
                    "object_pose_topic is empty. At least one policy is distillation and needs object pose."
                )
            if DdsObjectPoseStamped_ is None:
                raise RuntimeError(
                    "unitree_sdk2py geometry_msgs PoseStamped_ is not importable; "
                    "object pose subscription requires this DDS type."
                )
            self.object_pose_subscriber = ChannelSubscriber(topic, DdsObjectPoseStamped_)
            self.object_pose_subscriber.Init(self.ObjectPoseHandler, 10)
            logger.info("init: object_pose topic=%s type=PoseStamped_", topic)
        self._object_pose_ready = False
        self.latest_object_obs = np.zeros(7, dtype=np.float32)
        self.last_object_obs_time = 0.0

        self.wait_for_low_state()
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        logger.info(
            "init: policy_path=%s policy_type=%s backend=%s net_topics lowcmd=%s lowstate=%s",
            config.policy_path,
            self.policy_type,
            self.policy_backend,
            config.lowcmd_topic,
            config.lowstate_topic,
        )

        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action_robot = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles[config.policy_to_robot].copy()
        self.frame_stack = deque(maxlen=5)
        for _ in range(5):
            self.frame_stack.append(np.zeros(config.num_obs, dtype=np.float32))
        self.encoder_frame_stack: Optional[deque] = None
        self._current_encoder_stack_dim: Optional[int] = None
        if needs_object:
            self.wait_for_object_pose()

    def _active_is_distillation(self) -> bool:
        if self._use_stable_policy and self.policy_stable is not None:
            return self.policy_type_stable == "distillation"
        return self.policy_type == "distillation"

    def _active_policy_module(self) -> Any:
        if self._use_stable_policy and self.policy_stable is not None:
            return self.policy_stable
        return self.policy

    def _active_policy_backend(self) -> str:
        if self._use_stable_policy and self.policy_stable is not None:
            if self.policy_backend_stable is None:
                raise RuntimeError("stable policy backend is not initialized.")
            return self.policy_backend_stable
        return self.policy_backend

    def _active_encoder_dim(self) -> Optional[int]:
        if not self._active_is_distillation():
            return None
        if self._use_stable_policy and self.policy_stable is not None:
            return self.encoder_obs_dim_stable
        return self.encoder_obs_dim

    def _ensure_encoder_frame_stack(self) -> None:
        """Allocate or reallocate the encoder history deque for the currently active distillation policy."""
        dim = self._active_encoder_dim()
        if dim is None:
            self.encoder_frame_stack = None
            self._current_encoder_stack_dim = None
            return
        if self.encoder_frame_stack is not None and self._current_encoder_stack_dim == dim:
            return
        self.encoder_frame_stack = deque(maxlen=self.config.encoder_seq_len)
        for _ in range(self.config.encoder_seq_len):
            self.encoder_frame_stack.append(np.zeros(dim, dtype=np.float32))
        self._current_encoder_stack_dim = dim
        logger.info(
            "encoder frame stack: dim=%d (policy=%s)",
            dim,
            "stable" if self._use_stable_policy else "main",
        )

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self._lowstate_ready = True
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def ObjectPoseHandler(self, msg):
        object_obs = object_obs_from_pose_stamped(msg)
        if object_obs is None:
            return
        self.latest_object_obs = object_obs
        self._object_pose_ready = True
        self.last_object_obs_time = time.time()

    def send_cmd(self, cmd: Union[LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def get_current_object_obs(self) -> np.ndarray:
        if not self._object_pose_ready:
            raise RuntimeError("object pose not available yet (wait_for_object_pose should run first).")
        if (time.time() - self.last_object_obs_time) > self.config.object_pose_timeout_s:
            logger.warning(
                "object pose stale (>%0.2fs), reusing last observation",
                self.config.object_pose_timeout_s,
            )
        return self.latest_object_obs

    def wait_for_object_pose(self, timeout_s: float = 120.0, log_wait_interval_s: float = 2.0) -> None:
        if self.object_pose_subscriber is None:
            raise RuntimeError("No object pose subscriber initialized.")
        t0 = time.time()
        last_log = t0
        while not self._object_pose_ready:
            now = time.time()
            if (now - t0) > timeout_s:
                raise RuntimeError(
                    f"Timeout ({timeout_s}s) waiting for first object pose on '{self.config.object_pose_topic}'."
                )
            if (now - last_log) >= log_wait_interval_s:
                logger.warning(
                    "Still waiting for first object pose on '%s' (%.0fs / %.0fs)...",
                    self.config.object_pose_topic,
                    now - t0,
                    timeout_s,
                )
                last_log = now
            time.sleep(self.config.control_dt)
        print(f"First object pose received on '{self.config.object_pose_topic}'.")
        logger.info("stage: object pose stream ok")

    def wait_for_low_state(self, timeout_s: float = 30.0, log_wait_interval_s: float = 2.0):
        t0 = time.time()
        last_log = t0
        while not self._lowstate_ready:
            now = time.time()
            if (now - t0) > timeout_s:
                raise RuntimeError(
                    f"Timeout ({timeout_s}s) waiting for first LowState on '{self.config.lowstate_topic}'."
                )
            if (now - last_log) >= log_wait_interval_s:
                logger.warning(
                    "Still waiting for first LowState on '%s' (%.0fs / %.0fs)...",
                    self.config.lowstate_topic,
                    now - t0,
                    timeout_s,
                )
                last_log = now
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")
        logger.info(
            "stage: connected to robot (lowstate stream ok) tick=%s",
            int(self.low_state.tick),
        )

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        logger.info("stage: zero torque -- press remote START to continue")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        logger.info("stage: zero torque -- START received, moving to default pose next")

    def move_to_default_pos(self):
        print("Moving to default position.")
        logger.info("stage: move to default pose (ramp ~2s)")
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
        logger.info("stage: default pose reached")

    def default_pos_state(self):
        print("Enter default position state.")
        stable_hint = (
            f"press {self.config.key_start_stable.upper()} for policy_path_stable"
            if self.policy_stable is not None
            else "policy_path_stable not set (only main policy available)"
        )
        print(
            f"Waiting for remote: A = main (policy_path), {stable_hint}. "
            f"While running: A -> main, {self.config.key_start_stable.upper()} -> stable (if both loaded)."
        )
        logger.info(
            "stage: hold default pose -- A=main; %s=stable (if configured); hot-swap same keys in policy loop",
            self.config.key_start_stable,
        )
        default_pos = self.config.default_angles[self.config.policy_to_robot]
        while True:
            for motor_idx in range(self.config.num_actions):
                self.low_cmd.motor_cmd[motor_idx].q = default_pos[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
            if self.remote_controller.button[KeyMap.A] == 1:
                self._use_stable_policy = False
                logger.info("stage: A -- starting main policy (policy_path). SELECT to exit")
                break
            if self.policy_stable is not None and self.remote_controller.button[self._key_start_stable] == 1:
                self._use_stable_policy = True
                logger.info(
                    "stage: %s -- starting stable policy (policy_path_stable); "
                    "A=main, %s=stable (while running)",
                    self.config.key_start_stable,
                    self.config.key_start_stable,
                )
                break
        self._ensure_encoder_frame_stack()
        self._prev_a_for_policy_switch = int(self.remote_controller.button[KeyMap.A])
        self._prev_stable_key_for_policy_switch = int(
            self.remote_controller.button[self._key_start_stable]
        )

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
        if not self._policy_loop_started:
            self._policy_loop_started = True
            logger.info(
                "stage: policy loop running (control_dt=%.4f decimation=%d)",
                cfg.control_dt,
                cfg.control_decimation,
            )

        a_now = int(self.remote_controller.button[KeyMap.A])
        stable_key_now = int(self.remote_controller.button[self._key_start_stable])

        if self._use_stable_policy and self.policy_stable is not None and a_now == 1 and self._prev_a_for_policy_switch == 0:
            logger.info("stage: A -- switching from stable policy to main policy (policy_path)")
            self._use_stable_policy = False
            self._ensure_encoder_frame_stack()
        elif (
            not self._use_stable_policy
            and self.policy_stable is not None
            and stable_key_now == 1
            and self._prev_stable_key_for_policy_switch == 0
        ):
            logger.info(
                "stage: %s -- switching from main policy to stable policy (policy_path_stable); press A to return to main",
                self.config.key_start_stable,
            )
            self._use_stable_policy = True
            self._ensure_encoder_frame_stack()

        self._prev_a_for_policy_switch = a_now
        self._prev_stable_key_for_policy_switch = stable_key_now

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
        active_policy = self._active_policy_module()
        active_backend = self._active_policy_backend()
        if self._active_is_distillation():
            if self.encoder_frame_stack is None:
                raise RuntimeError("encoder_frame_stack is not initialized for distillation policy.")
            omega_norm = omega * cfg.ang_vel_scale
            qj_norm = (self.qj - cfg.default_angles[cfg.policy_to_robot]) * cfg.dof_pos_scale
            dqj_norm = self.dqj * cfg.dof_vel_scale
            qj_policy = qj_norm[cfg.robot_to_policy]
            dqj_policy = dqj_norm[cfg.robot_to_policy]
            action_policy_prev = self.action_robot[cfg.robot_to_policy]
            encoder_obs = build_student_encoder_obs(
                omega=omega_norm,
                gravity_orientation=gravity,
                cmd=cmd,
                qj=qj_policy,
                dqj=dqj_policy,
                action=action_policy_prev,
                cmd_scale=cfg.cmd_scale,
                num_actions=cfg.num_actions,
                object_obs=self.latest_object_obs,
            )
            enc_dim = self._active_encoder_dim()
            if enc_dim is not None:
                encoder_obs = resize_encoder_obs(encoder_obs, enc_dim)
            self.encoder_frame_stack.append(encoder_obs)
            encoder_seq = np.array(self.encoder_frame_stack, dtype=np.float32)
            action_policy = run_policy_inference(
                model=active_policy,
                backend=active_backend,
                distillation=True,
                policy_obs=stacked_obs,
                encoder_obs_seq=encoder_seq,
            )
        else:
            action_policy = run_policy_inference(
                model=active_policy,
                backend=active_backend,
                distillation=False,
                policy_obs=stacked_obs,
            )
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] deploy_real_steadytray: %(message)s",
    )
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

    logger.info(
        "startup: network=%s config=%s policy=%s policy_stable=%s",
        args.net,
        args.config,
        config.policy_path,
        (config.policy_path_stable or "(none)"),
    )
    ChannelFactoryInitialize(0, args.net)
    logger.info("startup: DDS ChannelFactoryInitialize done")
    controller = Controller(config)

    controller.zero_torque_state()
    controller.move_to_default_pos()
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            if controller.remote_controller.button[KeyMap.select] == 1:
                logger.info("stage: SELECT pressed -- entering damping and exiting")
                break
        except KeyboardInterrupt:
            logger.info("stage: keyboard interrupt -- entering damping and exiting")
            break

    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    logger.info("stage: damping command sent, goodbye")
    print("Exit")
