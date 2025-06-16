from pathlib import Path
import numpy as np
import time
from typing import List
import os
from copy import deepcopy
import torch
GX_STORAGE_PATH = os.getenv("GX_STORAGE_PATH")


from gx_utils import log, fm
from gx_utils.dtype import Render, ListRender, Physics, ListPhysics, Scene, ListScene, ID, ListPlan, Box, Mesh
from gx_utils.magic.profiler import profiler
from gx_utils.constant import GROUND
from gx_utils.magic.video import VideoRecorder

import sys
# TODO: add these packages' path to system path
sys.path.append('/mnt/afs/danshili/GalbotVLA/gx_sim')
sys.path.append('/mnt/afs/danshili/GalbotVLA/galbot_vla/src/sim_data_gen')
from gx_sim.types import MjBox, MjMesh, to_mj_obj
from gx_sim.render.batch_task import RenderTask, BatchRendererConfig
from gx_sim.render.batch_isaacsim import IsaacSimConfig
from gx_sim.render.config import TableTopRenderTaskConfig


from sim_data_gen.G1.robot_cfg import G1_HOME_POSE
from sim_data_gen.G1.robot_cfg import get_robot_cfg

FIXED_OBJ_NAMES = ["ground", "table"]
from sim_data_gen.G1.load_task import load_dexonomy_task, DEXONOMY_SAVE_ROOT
from sim_data_gen.G1.ik import IKSolver

from gx_utils.robot import RobotModel
from gx_utils.transform import rot2euler, euler2rot
robot = RobotModel.init("pin", f"{GX_STORAGE_PATH}/asset/robot/G1/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ_pyvista.urdf")
robot_cfg = get_robot_cfg("G1")
ik_solver = IKSolver.init(robot_cfg, ik_type='analytical', fix_fingers=True)
THUMB_HORIZONTAL_POSE=1.3

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def traj_to_action(traj_state, traj_command, physics=None, binary_closeness=False):
    traj_command = np.concatenate([traj_command[:,:41], traj_command[:,49:], traj_command[:,41:43], traj_command[:,43:45], traj_command[:,47:49], traj_command[:,45:47]],axis=-1)
    # input: [N,53] = full body joint qpos
    # output delta action: [N-1,8] = delta xyz + delta Euler + absolute thumb horizontal + absolute close
    # output preprioception: [N-1,8] = eef xyz + eef Euler + absolute thumb horizontal + absolute close
    seq_preprioception = []
    seq_delta_command = []
    cmd_next = None
    for i in range(len(traj_state)-1):
        eef_T = robot.fk_link(traj_state[i], link='R_hand_base_link')
        eef_trans = eef_T[0]
        eef_euler = rot2euler(eef_T[1])
        thumb_horizontal = np.array(traj_state[i,41]).reshape(1)
        finger_closeness = np.array([
            ((traj_state[i,42]) / 0.3 + traj_state[i,43] / 0.4 + traj_state[i,44] / 0.6) / 3,   # mean thumb closeness
            traj_state[i,45:].mean() / 1.7  # mean 4-finger closeness
        ]).mean()
        finger_closeness = np.array(finger_closeness).reshape(1)
        step_preprioception = np.concatenate([eef_trans, eef_euler, finger_closeness], axis=-1)

        cmd_eef_T = robot.fk_link(traj_state[i], link='R_hand_base_link')
        cmd_eef_trans = cmd_eef_T[0]
        cmd_eef_rot = cmd_eef_T[1]
        cmd_eef_euler = rot2euler(cmd_eef_T[1])
        cmd_thumb_horizontal = 1.3
        cmd_finger_closeness = float(physics.closeness[i] > 0)
        cmd_finger_closeness = np.array(cmd_finger_closeness).reshape(1)
        cmd_step = np.concatenate([cmd_eef_trans, cmd_eef_euler, cmd_finger_closeness], axis=-1)

        cmd_next_eef_T = robot.fk_link(traj_state[i+1], link='R_hand_base_link')
        cmd_next_eef_trans = cmd_next_eef_T[0]
        cmd_next_eef_rot = cmd_next_eef_T[1]
        cmd_next_eef_euler = rot2euler(cmd_next_eef_T[1])
        cmd_next_thumb_horizontal = np.array(traj_command[i+1,41]).reshape(1)

        cmd_next_finger_closeness = ((float(physics.closeness[i+1] >= physics.closeness[i])) and (physics.closeness[i+1] > 0.025))
        cmd_next_finger_closeness = np.array(cmd_next_finger_closeness).reshape(1)
        cmd_next = np.concatenate([cmd_next_eef_trans, cmd_next_eef_euler, cmd_next_finger_closeness], axis=-1)

        cmd_delta = deepcopy(cmd_next)
        # delta xyz command in Eucledian space 
        cmd_delta[:3] = cmd_next[:3] - cmd_step[:3]
        # delta Euler angle
        cmd_delta[3:6] = rot2euler(np.einsum('ab,bc->ac', cmd_next_eef_rot, cmd_eef_rot.T))
        seq_preprioception.append(step_preprioception)
        seq_delta_command.append(cmd_delta)

    seq_preprioception = np.array(seq_preprioception)
    seq_delta_command = np.array(seq_delta_command)    
    return seq_preprioception, seq_delta_command

def state_to_preprio(state):
    # input state: [7+12] = current state arm qpos + current state hand qpos

    # output step_preprioception [7] = current eef trans + current eef rot + current closeness
    eef_T = robot.fk_link(np.concatenate([np.array(G1_HOME_POSE[:-19]),state]), link='R_hand_base_link')
    eef_trans = eef_T[0]
    eef_euler = rot2euler(eef_T[1])
    thumb_horizontal = np.array(state[7]).reshape(1)
    finger_closeness = np.array([
        ((state[8]) / 0.3 + state[9] / 0.4 + state[10] / 0.6) / 3,   # mean thumb closeness
        state[11:].mean() / 1.7  # mean 4-finger closeness
    ]).mean()
    finger_closeness = np.array(finger_closeness).reshape(1)
    step_preprioception = np.concatenate([eef_trans, eef_euler, finger_closeness], axis=-1)

    # check if hand is closed and has not grasped any object. If grasped object, the finger joint angle coupling coefficient would deviate from control target.
    reset_signal = False
    # if finger_closeness>0.6 and np.abs(state[11]/state[12]-1)<0.15 and np.abs(state[13]/state[14]-1)<0.15 and np.abs(state[15]/state[16]-1)<0.15 and np.abs(state[17]/state[18]-1)<0.15:
    #     reset_signal = True
    return step_preprioception, reset_signal

def absolute_action_to_traj(last_command, curr_state, pred_chunk, delta_closeness=False, triggered=0, full_chunk=None, step=-1):
    # input last_command: [N, 7+12] = last command arm qpos +last command hand qpos
    # input curr_state: [N, 7+12] = current state arm qpos + current state hand qpos
    # input pred_chunk: [n_steps, 7] = pred delta xyz + pred delta Euler + pred colseness

    # output command_action_seq: [N, 7+12] = arm+hand command
    # output triggered: if positive, keep squeezing fingers unless caught 5 consecutive negative signal
    last_command = last_command[-19:]
    curr_state = curr_state[-19:]

    curr_eef_T = robot.fk_link(np.concatenate([np.array(G1_HOME_POSE[:-19]),curr_state]), link='R_hand_base_link')
    curr_eef_xyz = curr_eef_T[0]
    curr_eef_rot = curr_eef_T[1]
    curr_thumb_horizontal = last_command[7]
    curr_fingers = last_command[7:]
    curr_closeness = np.array([
            ((curr_fingers[1]) / 0.3 + curr_fingers[2] / 0.4 + curr_fingers[3] / 0.6) / 3,   # mean thumb closeness
            curr_fingers[4:].mean() / 1.7  # mean 4-finger closeness
        ]).mean()

    curr_state_fingers = curr_state[7:]
    curr_state_closeness = np.array([
            ((curr_state_fingers[1]) / 0.3 + curr_state_fingers[2] / 0.4 + curr_state_fingers[3] / 0.6) / 3,   # mean thumb closeness
            curr_state_fingers[4:].mean() / 1.7  # mean 4-finger closeness
        ]).mean()

    command_action_seq = []
    for i in range(len(pred_chunk)):
        delta_xyz = pred_chunk[i][:3]
        delta_euler = pred_chunk[i][3:6]
        pred_close_triggered = pred_chunk[i][6] 
        pred_close_triggered = np.where(pred_close_triggered>0.6, 1.0, 0.0)

        # open up fingers only if there's 5 consequent open signals. Otherwise keep closing the fingers.
        if pred_close_triggered:
            absolute_closeness = min(curr_closeness+0.0275, 1.0)
            triggered = 5
        else:
            triggered -= 1
            if triggered <= 0:
                absolute_closeness = max(curr_closeness-0.05, 0.0)
            else:
                absolute_closeness = min(curr_closeness+0.0275, 1.0)
        curr_closeness = absolute_closeness
            

        step_eef_xyz = curr_eef_xyz + delta_xyz
        step_eef_rot = np.einsum("ab,bc->ac", euler2rot(delta_euler), curr_eef_rot)
        # solve IK
        succ, step_arm_qpos = ik_solver.ik(step_eef_xyz, step_eef_rot, qpos=curr_state[:7], silent=True)


        # transform to joint qpos vector so that can apply control in MuJoco
        absolute_fingers = np.array([
            THUMB_HORIZONTAL_POSE,
            absolute_closeness * 0.3,
            absolute_closeness * 0.4,
            absolute_closeness * 0.6,
            absolute_closeness * 1.7,
            absolute_closeness * 1.7,
            absolute_closeness * 1.7,
            absolute_closeness * 1.7,
            absolute_closeness * 1.7,
            absolute_closeness * 1.7,
            absolute_closeness * 1.7,
            absolute_closeness * 1.7,
        ])
        step_fingers = absolute_fingers
        step_command = np.concatenate([step_arm_qpos,step_fingers])
        curr_eef_xyz = deepcopy(step_eef_xyz)
        curr_eef_rot = deepcopy(step_eef_rot)
        curr_fingers = deepcopy(step_fingers)
        command_action_seq.append(step_command)
    
    command_action_seq = np.array(command_action_seq)
    # import pdb;pdb.set_trace()
    return command_action_seq, triggered

class G1VLAEnv:
    def __init__(self,
        n_envs=1,
        render_task=None,
        n_obs_steps=2,
        horizon=8,
        uid=None,
        ii=0,
        n_action_steps=8,
        delta_closeness=False,
        output_dir='default'
    ):
        self.uid = uid
        self.ii = ii
        self.n_envs = n_envs
        self.n_obs_steps = n_obs_steps
        self.render_task_reseted = False
        self.delta_closeness = delta_closeness
        self.test_split_root = f'{GX_STORAGE_PATH}/sim'
        self.output_dir = output_dir

        self.curr_idx = 0
        # to enumerate all test setups
        self.curr_start_idx = 0
        self.curr_last_idx = self.n_envs
        self.robot_cfg = get_robot_cfg("G1")

        # initialize physics simulator
        from gx_sim.physics.simulator import MjSimConfig, MjSim
        self.sim_config = MjSimConfig(
                robot_cfg=self.robot_cfg,
                headless=1,
                realtime_sync=0,
                use_ground_plane=False,
                ctrl_dt=0.1,
            )

        # initialize render simulator
        self.task_cfg = TableTopRenderTaskConfig
        self.render_cfg = BatchRendererConfig(
            num_envs=1,
            debug=0,
            isaac_sim=IsaacSimConfig(headless=1),
        )
        from gx_sim.physics.simulator import MjSimConfig, MjSim
        self.sim = MjSim(self.sim_config)
        self.render_task = render_task

        self.num_success = 0
        self.num_total = 0
        # records control signal in absolute position
        self.absolute_qpos_command = np.array(G1_HOME_POSE[-19:])
        # records joint angle states in absolute position
        self.joint_state = np.array(G1_HOME_POSE[-19:])
        self.head_images = []
        self.wrist_images = []
        self.qposes = []
        self.absolute_qpos_commands = []


    def step(self, action_dict, step_physics=True):
        # action_dict: a dictionary of predicted action chunk
        # action_dict['action'] [B, n_action_steps. n_dim] = a batch of n_step action chunk prediction. Currently B=1 because of parallelization problems


        # execute many steps in one rollout. currently num_action_steps=8 steps.
        pred_chunk = action_dict['action'][0]
        for cmd_pred in pred_chunk:
            cmd_pred_delta_action = np.concatenate([cmd_pred[:,-8:-2],cmd_pred[:,-1:]], axis=-1)    # conforms with current lerobot info.json. use delta EEF xyz + delta eef Euler + closeness

            # compute MuJoCo control input from network prediction. Processes 1 action step.
            command_action_seq, triggered = absolute_action_to_traj(self.absolute_qpos_command, self.joint_state, cmd_pred[None,...], self.delta_closeness, self.close_triggered, full_chunk=pred_chunk, step=self.curr_step)
            if triggered < 0:
                self.close_triggered = 0
            else:
                self.close_triggered = triggered
            absolute_action = command_action_seq[0]

            # step physics simulation. At env initialization time do not run physics step.
            if step_physics: 

                # For debug usage.  May ignore
                try:                  
                    ref_command = np.concatenate([
                        self.physics.traj_command[self.curr_step][34:41],
                        self.physics.traj_command[self.curr_step][49:],
                        self.physics.traj_command[self.curr_step][41:49],
                    ])
                except:
                    ref_command = None


                self.absolute_qpos_command = absolute_action
                self.absolute_qpos_commands.append(deepcopy(self.absolute_qpos_command))
                cmd = self.sim.qpos2ctrl_matrix @ np.concatenate([self.absolute_qpos_command,np.array([0]*6)])
                self.sim.step(cmd)
                self.joint_state = self.sim.robot_qpos.copy()
                self.curr_step += 1

            # update the record of object poses per-timestep
            tar_obj_pose = np.zeros((len(self.state_obj_list_id), 4, 4))
            for i, obj_name in enumerate(self.state_obj_list_id):
                pos, rot = self.sim.get_body_pose(obj_name)
                tar_obj_pose[i, :3, 3] = pos
                tar_obj_pose[i, :3, :3] = rot
            self.state_obj_list_pose = tar_obj_pose.copy()

            # check success
            self.last_obj_position = self.sim.get_body_pose(self.target_name)[0].copy()
            state_robot_qpos = np.concatenate([np.array(G1_HOME_POSE[:34]),self.sim.robot_qpos.copy()])
            success = self.check_success()
            done = (success) or (self.curr_step > 500)
            
            # render updated frame
            robot_state_chunk = [
                dict(
                    trans = np.zeros(3),
                    rot = np.eye(3),
                    qpos = state_robot_qpos
                )
            ]
            obj_list = []
            for o_id, obj_uuid in enumerate(self.physics.obj_ids):
                obj = self.scene.obj_dict[obj_uuid]
                obj.pose = self.state_obj_list_pose[o_id]
                obj_list.append(obj)
            obj_list_chunk = [obj_list]

            if self.is_first_render:
                for _ in range(25):
                    self.render_task.render(robot_state_chunk, obj_list_chunk)
                self.is_first_render = False
            render_ret = self.render_task.render(robot_state_chunk, obj_list_chunk)
            head_image = render_ret[0]['head_camera']['rgb']    # HWC
            wrist_image = render_ret[0]['wrist_camera']['rgb']  # HWC
            third_image = render_ret[0]['third_view_camera']['rgb']
            full_image = np.concatenate([head_image, wrist_image,third_image], axis=1)
            sim_mjqpos = self.sim.robot_qpos.copy()
            sim_qpos, reset_signal = state_to_preprio(self.joint_state)
            sim_qpos = np.concatenate([self.joint_state[:7], sim_qpos], axis=-1)
            self.rgb_image_recorder.add_frame(full_image)

            # update observation
            if len(self.head_images) == 0:
                self.head_images = [head_image] * self.n_obs_steps
                self.wrist_images = [wrist_image]* self.n_obs_steps
                self.qposes = [sim_qpos] * self.n_obs_steps
            else:
                self.head_images.append(head_image)
                self.wrist_images.append(wrist_image)
                self.qposes.append(sim_qpos)
                self.head_images = self.head_images[-self.n_obs_steps:]
                self.wrist_images = self.wrist_images[-self.n_obs_steps:]
                self.qposes = self.qposes[-self.n_obs_steps:]


        # move to idle configuration if finger closed and caught nothing.
        # currently this is never activated. May ignore
        if reset_signal:
            self.close_triggered = 0
            reset_action_seq = []
            state_robot_qpos = np.concatenate([np.array(G1_HOME_POSE[:34]),self.sim.robot_qpos.copy()])
            for i in range(15):
                reset_action_seq.append(np.concatenate([
                    self.sim.robot_qpos.copy()[-19:-12] + (np.array(G1_HOME_POSE[-19:-12]) - self.sim.robot_qpos.copy()[-19:-12]) * (0 / 100), 
                    self.sim.robot_qpos.copy()[-12:-11],
                    self.sim.robot_qpos.copy()[-11:] + (np.array(G1_HOME_POSE[-11:]) - self.sim.robot_qpos.copy()[-11:]) * (min((i+1) / 10, 1.0)), 
                    
                ]))
            reset_action_seq = np.array(reset_action_seq)

            for action in reset_action_seq:
                cmd = self.sim.qpos2ctrl_matrix @ np.concatenate([action,np.array([0]*6)])
                self.sim.step(cmd)
                self.joint_state = self.sim.robot_qpos.copy()
                self.absolute_qpos_command = self.sim.robot_qpos.copy()

                tar_obj_pose = np.zeros((len(self.state_obj_list_id), 4, 4))
                for i, obj_name in enumerate(self.state_obj_list_id):
                    pos, rot = self.sim.get_body_pose(obj_name)
                    tar_obj_pose[i, :3, 3] = pos
                    tar_obj_pose[i, :3, :3] = rot
                self.state_obj_list_pose = tar_obj_pose.copy()
                
                self.last_obj_position = self.sim.get_body_pose(self.target_name)[0].copy()
                state_robot_qpos = np.concatenate([np.array(G1_HOME_POSE[:34]),self.sim.robot_qpos.copy()])
                success = self.check_success()
                done = (success) or (self.curr_step > 400)
                self.curr_step += 1
                # render updated frame
                robot_state_chunk = [
                    dict(
                        trans = np.zeros(3),
                        rot = np.eye(3),
                        qpos = state_robot_qpos
                    )
                ]
                obj_list = []
                for o_id, obj_uuid in enumerate(self.physics.obj_ids):
                    obj = self.scene.obj_dict[obj_uuid]
                    obj.pose = self.state_obj_list_pose[o_id]
                    obj_list.append(obj)
                obj_list_chunk = [obj_list]

                if self.is_first_render:
                    for _ in range(25):
                        self.render_task.render(robot_state_chunk, obj_list_chunk)
                    self.is_first_render = False
                render_ret = self.render_task.render(robot_state_chunk, obj_list_chunk)
                head_image = render_ret[0]['head_camera']['rgb']    # HWC
                wrist_image = render_ret[0]['wrist_camera']['rgb']  # HWC
                third_image = render_ret[0]['third_view_camera']['rgb']
                full_image = np.concatenate([head_image, wrist_image,third_image], axis=1)
                sim_mjqpos = self.sim.robot_qpos.copy()
                sim_qpos, reset_signal = state_to_preprio(self.joint_state)
                self.rgb_image_recorder.add_frame(full_image)

                if len(self.head_images) == 0:
                    self.head_images = [head_image] * self.n_obs_steps
                    self.wrist_images = [wrist_image]* self.n_obs_steps
                    self.qposes = [sim_qpos] * self.n_obs_steps
                else:
                    self.head_images.append(head_image)
                    self.wrist_images.append(wrist_image)
                    self.qposes.append(sim_qpos)
                    self.head_images = self.head_images[-self.n_obs_steps:]
                    self.wrist_images = self.wrist_images[-self.n_obs_steps:]
                    self.qposes = self.qposes[-self.n_obs_steps:]

        # write the dictionary as input to network. 
        # the obs['qpos'] item corresponds to "observation.state" in lerobot info.json.
        obs = dict()
        obs['head_image'] = np.moveaxis(np.array(self.head_images[-self.n_obs_steps:]),-1,1
                ).astype(np.float32) / 255. # self.n_obs_steps, 3, 300, 400
        obs['wrist_image'] = np.moveaxis(np.array(self.wrist_images[-self.n_obs_steps:]),-1,1
                ).astype(np.float32) / 255. # self.n_obs_steps, 3, 300, 400
        obs['qpos'] = np.array(self.qposes[-self.n_obs_steps:]).astype(np.float32)    # self.n_obs_steps, 19
        obs['head_image'] = obs['head_image'][None,...]
        obs['wrist_image'] = obs['wrist_image'][None,...]
        obs['qpos'] = obs['qpos'][None,...]


        return obs, success, done

    def reset(self, enable_randomize=False):
        self.close_triggered = 0
        # clean memory for obsolete rollouts
        if self.sim.launched:
            self.sim.close()
            del self.sim
            from gx_sim.physics.simulator import MjSimConfig, MjSim
            self.sim = MjSim(self.sim_config)

            del self.head_images
            del self.wrist_images
            del self.qposes
            self.head_images = []
            self.wrist_images = []
            self.qposes = []
            self.curr_idx += 1

        # sample initial pose from our test split
        uid, i = self.uid, self.ii
        physicss = ListPhysics.load_h5_list(Path(self.test_split_root) / 'physics' / 'debug_chomp' / uid / '0' / 'physics.h5')
        physics = physicss[i]
        self.id = physics.id
        scenes = ListScene.load_h5_list(Path(self.test_split_root) / 'scene' / 'debug' / uid / '0' / 'scene.h5')
        id_scene = physics.id_scene.index
        scene = scenes[id_scene]
        self.physics = physics
        self.scene = scene

        # load scene into physics simulator
        for obj_id, obj in scene.obj_dict.items():
            if isinstance(obj, Mesh):
                self.target_name = obj.uuid
                obj.name = obj.uuid
                obj.path = obj.path.replace('/raid/danshili/Workspace/Dexonomy','/mnt/afs/danshili/Workspace/Dexonomy-private')
                obj.path_visual = obj.path_visual.replace('/home/ubuntu/danshili/GalbotVLA/galbot_vla/src/sim_data_gen/sim_data_gen/storage',GX_STORAGE_PATH)
                obj.visual_path = obj.visual_path.replace('/home/ubuntu/danshili/GalbotVLA/galbot_vla/src/sim_data_gen/sim_data_gen/storage',GX_STORAGE_PATH)
                obj.path_list_collision = [s.replace('/raid/danshili/Workspace/Dexonomy','/mnt/afs/danshili/Workspace/Dexonomy-private') for s in obj.path_list_collision]
                mj_obj = MjMesh(**obj.model_dump())
            elif isinstance(obj, Box):
                mj_obj = MjBox(**obj.model_dump())
            else:
                raise NotImplementedError(f"Object type {type(obj)} not supported")
            if obj.name in FIXED_OBJ_NAMES:
                mj_obj.fixed_body = True

            # change path to global path
            if hasattr(mj_obj, "path"):
                convex_path = Path(mj_obj.path).parent.parent / "urdf/meshes"
                mj_obj.path_list_collision = list(
                    [str(p) for p in convex_path.iterdir()]
                )
            self.sim.add_obj(mj_obj)
        self.sim.launch()
        state_robot_qpos = []
        obj_list = list(scene.obj_dict.values())
        visible_objs_list = [list(scene.obj_dict.values())]
        self.state_obj_list_uuid = [o.uuid for o in obj_list]
        self.state_obj_list_id = [o.name for o in obj_list]
        self.state_obj_list_pose = []
        # self.sim.reset(physics.traj_state[0][34:])
        self.sim.reset(np.array(G1_HOME_POSE[34:]))
        
        self.first_obj_position = self.sim.get_body_pose(self.target_name)[0].copy()
        # absolute_qpos_command = physics.traj_command[0][34:]
        absolute_qpos_command = np.array(G1_HOME_POSE[34:])
        self.absolute_qpos_command = deepcopy(absolute_qpos_command)
        self.joint_state = deepcopy(absolute_qpos_command)
        # self.absolute_qpos_command = np.concatenate([absolute_qpos_command[:7], 
        #                     absolute_qpos_command[15:], 
        #                     absolute_qpos_command[7:9], 
        #                     absolute_qpos_command[9:11], 
        #                     absolute_qpos_command[13:15], 
        #                     absolute_qpos_command[11:13]])
        # self.joint_state = np.concatenate([absolute_qpos_command[:7], 
        #                     absolute_qpos_command[15:], 
        #                     absolute_qpos_command[7:9], 
        #                     absolute_qpos_command[9:11], 
        #                     absolute_qpos_command[13:15], 
        #                     absolute_qpos_command[11:13]])
        self.curr_step = 0

        self.traj_state = []
        self.traj_command = []

        # setup render environment
        with profiler.track(desc="create_scene", enabled=False):
            self.render_task.setup(self.robot_cfg, [[scene]], enable_randomize=enable_randomize)
        with profiler.track(desc="create_randomize_pool", enabled=False):
            self.render_task.create_randomize_pool()
        self.render_task.reset()
        self.render_task.apply_randomization(
            visible_objs_list,
            env_ids=list(range(1)),
            show_profile=enable_randomize,
            rand_material=enable_randomize,
            rand_light=enable_randomize,
            rand_robot_shader=enable_randomize,
            rand_obj_shader=enable_randomize,
            rand_spatial=enable_randomize,
            rand_camera=enable_randomize,
            rand_obj_material=enable_randomize,
        )
        self.is_first_render = True

        self.rgb_image_recorder = VideoRecorder()

        

    def check_success(self):
        success = self.last_obj_position[2] - self.first_obj_position[2] >= 0.05
        return success

    def save_video(self):
        uid, i = self.uid, self.ii
        video_dir = f'{GX_STORAGE_PATH}/sim/inference_video/{self.output_dir}/{self.id.group}'
        os.makedirs(video_dir, exist_ok=True)
        video_path = f"{video_dir}/{i}.mp4"
        self.rgb_image_recorder.output(video_path)
        try:
            self.rgb_image_recorder.close()
        except:
            pass        

        absolute_qpos_commands = np.array(self.absolute_qpos_commands)
        absolute_qpos_commands_path = video_path.replace('.mp4', '.npy')
        np.save(absolute_qpos_commands_path, absolute_qpos_commands)
