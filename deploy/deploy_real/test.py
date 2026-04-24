import os
import numpy as np
import time
import torch
from typing import Union

# Giả định các module này nằm trong cùng thư mục hoặc path của bạn
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.remote_controller import RemoteController, KeyMap
from config import Config

class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Định nghĩa mảng vị trí tĩnh (Static Position) bạn cung cấp
        # Thứ tự giả định: 12 khớp chân -> 3 khớp eo -> 14 khớp tay
        self.static_target_pos = np.array([
            -0.1, 0, 0, 0.3, -0.2, 0,       # Chân trái (6)
            -0.1, 0, 0, 0.3, -0.2, 0,       # Chân phải (6)
            0, 0, 0,                        # Eo (3)
            0, 0.25, 0, 0.97, 0.15, 0, 0,   # Tay trái (7)
            0, -0.25, 0, 0.97, -0.15, 0, 0  # Tay phải (7)
        ], dtype=np.float32)

        self._lowstate_ready = False
        
        # Khởi tạo DDS dựa trên loại tin nhắn (hg hoặc go)
        if config.msg_type == "hg":
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        elif config.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)
        else:
            raise ValueError("Invalid msg_type")

        # Chờ nhận dữ liệu từ robot
        self.wait_for_low_state()

        # Khởi tạo lệnh mặc định
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self._lowstate_ready = True
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self._lowstate_ready = True
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self, timeout_s: float = 30.0) -> None:
        t0 = time.time()
        while not self._lowstate_ready:
            if time.time() - t0 > timeout_s:
                raise RuntimeError(f"Timeout waiting for LowState on {self.config.lowstate_topic}")
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Trạng thái Torque bằng 0. Nhấn START trên tay cầm để bắt đầu...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Đang di chuyển về vị trí mặc định của Config...")
        total_time = 2.0
        num_step = int(total_time / self.config.control_dt)
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        
        init_dof_pos = np.zeros(len(dof_idx), dtype=np.float32)
        for i in range(len(dof_idx)):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        for i in range(num_step):
            alpha = i / num_step
            for j in range(len(dof_idx)):
                motor_idx = dof_idx[j]
                target_pos = init_dof_pos[j] * (1 - alpha) + default_pos[j] * alpha
                self.low_cmd.motor_cmd[motor_idx].q = target_pos
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def run(self):
        """
        Hàm thực thi chính: Truyền vị trí tĩnh đã định nghĩa
        """
        # 1. Truyền cho các khớp chân (12 khớp đầu tiên)
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.static_target_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # 2. Truyền cho các khớp eo và tay (từ index 12 trở đi)
        offset = 12
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            # Kiểm tra tránh tràn mảng nếu config và mảng tĩnh lệch nhau
            if (offset + i) < len(self.static_target_pos):
                self.low_cmd.motor_cmd[motor_idx].q = self.static_target_pos[offset + i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

        # Gửi lệnh đi
        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface (e.g. eth0)")
    parser.add_argument("config", type=str, help="config file name", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Khởi tạo DDS
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Quy trình vận hành
    controller.zero_torque_state()
    controller.move_to_default_pos()

    print("Bắt đầu chế độ vị trí tĩnh. Nhấn SELECT trên tay cầm để thoát.")
    while True:
        try:
            controller.run()
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break

    # Khi thoát, đưa robot về trạng thái damping (an toàn)
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Đã thoát và chuyển sang chế độ Damping.")