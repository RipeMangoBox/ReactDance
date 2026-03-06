import numpy as np
import cv2
import os
import shutil
import subprocess
import tempfile

import matplotlib
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip, VideoFileClip, AudioFileClip
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3

from datasets.dd100lf_all2 import (
    t2m_kinematic_chain,
    t2m_left_hand_chain,
    t2m_right_hand_chain,
)

SMPLX_JOINT_NAMES = [
    'pelvis', #0
    'left_hip', 
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle', # 7
    'right_ankle', # 8
    'spine3', 
    'left_foot', # 10
    'right_foot', # 11
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow', 
    'right_elbow',
    'left_wrist', #20
    'right_wrist', #21
    'jaw', #22
    'left_eye_smplhf', #23
    'right_eye_smplhf', #24
    'left_index1', #25
    'left_index2', #26
    'left_index3', #27
    'left_middle1', #28
    'left_middle2', #29
    'left_middle3', #30
    'left_pinky1', #31
    'left_pinky2', #32
    'left_pinky3', #33 
    'left_ring1', #34
    'left_ring2',# 35
    'left_ring3', #36
    'left_thumb1', #37
    'left_thumb2', #38
    'left_thumb3', #39
    'right_index1', #40
    'right_index2', 
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3'
]



# Define edges to connect joints
edges = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 4],
    [2, 5],
    [3, 6],
    [4, 7],
    [5, 8],
    [6, 9],
    [7, 10],
    [8, 11],
    [9, 12],
    [9, 13],
    [9, 14],
    [12, 15],
    [13, 16],
    [14, 17],
    [16, 18],
    [17, 19],
    [18, 20],
    [19, 21],
    [20, 25], [25, 26], [26, 27],
    [20, 28], [28, 29], [29, 30],
    [20, 31], [20, 32], [20, 33],
    [20, 34], [20, 35], [20, 36],
    [20, 37], [20, 38], [20, 39],

    [21, 40], [40, 41], [41, 42],
    [21, 43], [21, 44], [21, 45],
    [21, 46], [46, 47], [47, 48],
    [21, 49], [49, 50], [50, 51],
    [21, 52], [52, 53], [53, 54]
]

def determine_up_axis(joints, joints_num):
    # SMPL-X 关节索引
    head_idx = 15  # 'head'
    left_foot_idx = 10  # 'left_foot'
    right_foot_idx = 11  # 'right_foot'
    
    # 取第一组关节数据（形状为 (帧数, 关节数 * 3)）
    this_joints = joints[0][0].reshape(-1, joints_num, 3)  # 重塑为 (帧数, 关节数, 3)
    
    # 提取头部和脚部坐标
    head_pos = this_joints[:, head_idx, :]  # 形状为 (帧数, 3)
    left_foot_pos = this_joints[:, left_foot_idx, :]  # 形状为 (帧数, 3)
    right_foot_pos = this_joints[:, right_foot_idx, :]  # 形状为 (帧数, 3)
    
    # 计算脚部平均位置
    foot_pos = (left_foot_pos + right_foot_pos) / 2  # 形状为 (帧数, 3)
    
    # 计算头部和脚部的平均坐标差异
    diff = np.mean(head_pos - foot_pos, axis=0)  # 形状为 (3,)，表示 x, y, z 的平均差
    
    # 找到绝对值最大的轴
    max_axis = np.argmax(np.abs(diff))
    up_axis = ['x', 'y', 'z'][max_axis]
    
    # 判断方向（正或负）
    direction = 'positive' if diff[max_axis] > 0 else 'negative'
    
    print(f"Determined up axis: {up_axis} ({direction})")
    return up_axis, direction

def visualize2(joints, joints_leader, evaldir, dance_names):
    '''
    video 左上角原点，人物z为上方向（与video纵轴方向相反，所以需要-z）
     (0, 0)
        ------------->
        |
        |
        |
        |
        |
        v
    '''
    save_folder = os.path.join(evaldir, 'videos')
    os.makedirs(save_folder, exist_ok=True)
    
    joints_num = joints[0][0].shape[-1] // 3
    colors = [(80, 80, 255)]  # reactor color
    colors_l = [(255, 80, 80)]  # leader color
    output_videos = []
    
    # 判断上方向
    up_axis, direction = determine_up_axis(joints, joints_num)
    
    print(len(joints), len(joints_leader), flush=True)
    len_joints = min(len(joints), len(joints_leader))

    for ii in range(len_joints):
        this_joints = joints[ii][0]
        this_leader_joints = joints_leader[ii][0]
        this_leader_joints = this_leader_joints.reshape(len(this_leader_joints), joints_num, 3)
        this_joints = this_joints.reshape(len(this_joints), joints_num, 3)

        video_name = os.path.join(save_folder, f'{dance_names[ii]}.mp4')
        output_videos.append(video_name)

        frame_rate = 30.0
        frame_size = (960, 540)

        frames = []

        # Generate frames
        for i in range(len(this_joints)):
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

            # --- 添加 XYZ 坐标轴 ---
            origin = (480, 270)
            # axis_length = 50
            # x_end = (origin[0] + axis_length, origin[1])
            # cv2.line(frame, origin, x_end, (0, 0, 255), 2)
            # cv2.putText(frame, 'X', x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # y_end = (origin[0], origin[1] - axis_length)
            # cv2.line(frame, origin, y_end, (0, 255, 0), 2)
            # cv2.putText(frame, 'Y', y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # z_end = (origin[0], origin[1] + axis_length)
            # cv2.line(frame, origin, z_end, (255, 0, 0), 2)
            # cv2.putText(frame, 'Z', z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            y_offset, z_offset = origin
            # --- 绘制关节连线 ---
            for j in range(len(edges)):
                if edges[j][0] < joints_num and edges[j][1] < joints_num:
                    pt1 = tuple((this_joints[i][edges[j][0]] * 100).astype(int))
                    pt2 = tuple((this_joints[i][edges[j][1]] * 100).astype(int))
                    cv2.line(frame, [pt1[1] + y_offset, -pt1[2] + z_offset], [pt2[1] + y_offset, -pt2[2] + z_offset], colors[0], thickness=2)

            for j in range(len(edges)):
                if edges[j][0] < joints_num and edges[j][1] < joints_num:
                    pt1 = tuple((this_leader_joints[i][edges[j][0]] * 100).astype(int))
                    pt2 = tuple((this_leader_joints[i][edges[j][1]] * 100).astype(int))
                    cv2.line(frame, [pt1[1] + y_offset, -pt1[2] + z_offset], [pt2[1] + y_offset, -pt2[2] + z_offset], colors_l[0], thickness=2)

            # Convert BGR to RGB (OpenCV uses BGR by default)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        print(f"[{ii + 1}/{len(joints)}] Frames generated for {video_name}")

        # Create a video clip from the frames
        clip = ImageSequenceClip(frames, fps=frame_rate)

        # Add music if exists
        music_source = "./data_lazy/music/mp3/test/"
        if '_' in dance_names[ii]:
            mp3 = '_'.join(dance_names[ii].split('_')[:3])
            music_name = os.path.join(music_source, mp3 + '.mp3')
        else:
            music_name = os.path.join(music_source, dance_names[ii] + '.mp3')

        # Calculate video duration
        video_duration = len(frames) / frame_rate

        if os.path.exists(music_name):
            audio = AudioFileClip(music_name)
            min_length = min(audio.duration, video_duration)
            # 对齐音频与视频长度，只需对音频截短即可
            audio = audio.set_duration(min_length)
            clip = clip.set_audio(audio)

        # Write once with optional audio
        clip.write_videofile(
            video_name,
            codec='libx264',
            audio_codec='aac',
            # 建议使用绝对路径来防止临时文件问题
            temp_audiofile=os.path.abspath(os.path.join(save_folder, 'temp-audio.m4a')),
            remove_temp=True,
            # 可以添加logger='bar'来查看更详细的 ffmpeg 输出，以帮助调试
            # logger='bar'
        )
        
    return output_videos

def _add_title(frame, text, position, font_scale=1, color=(0, 255, 0), thickness=2):
    """ 在视频帧上添加文本和分割线 """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def plot_3d_motion_alternative(
    save_path,
    kinematic_tree,
    mp_joints,
    title,
    figsize=(10, 10),
    fps=120,
    radius=4,
    up_axis=1,
    nb_joints=22,
):
    """
    通过 ffmpeg 合成视频，避免 moviepy 并发冲突。
    """
    temp_dir = tempfile.mkdtemp()
    temp_silent_video = os.path.join(temp_dir, "temp_silent.mp4")

    print(f"临时文件将处理于: {temp_dir}")

    audio_path = f"./data_lazy/music/mp3/all/{title}.mp3"
    matplotlib.use("Agg")

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        ax.grid(b=False)
        ax.view_init(azim=-60, elev=30)
        ax.dist = 20

    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()

    mp_data = []
    frame_number = min([data.shape[0] for data in mp_joints])

    colors = [
        "red",
        "green",
        "black",
        "blue",
        "orange",
        "darkred",
        "purple",
        "brown",
        "gray",
        "olive",
        "maroon",
        "limegreen",
        "gold",
        "magenta",
        "darkorange",
        "saddlebrown"
        "darkblue",
    ]

    mp_colors = [colors[i] for i in range(len(mp_joints))]

    for joints in mp_joints:
        data = joints.copy().reshape(len(joints), -1, 3)
        height_offset = data.min(axis=0).min(axis=0)[up_axis]
        data[:, :, up_axis] -= height_offset
        mp_data.append({"joints": data})

    def plot_xyPlane(minx, maxx, miny, maxy, minz):
        verts = [
            [minx, miny, minz],
            [minx, maxy, minz],
            [maxx, maxy, minz],
            [maxx, miny, minz],
        ]
        xy_plane = Poly3DCollection([verts])
        xy_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xy_plane)

    try:
        print("开始生成帧图像...")
        for i in range(frame_number):
            ax.lines = []
            ax.collections = []
            plot_xyPlane(-4, 4, -4, 4, 0)
            verts = [
                [-radius, -radius, 0],
                [-radius, radius, 0],
                [radius, radius, 0],
                [radius, -radius, 0],
            ]
            ax.add_collection3d(
                Poly3DCollection([verts], facecolors="lightgrey", linewidths=0, alpha=0.2)
            )

            for pid, data in enumerate(mp_data):
                for chain, color in zip(
                    kinematic_tree, [mp_colors[pid]] * len(kinematic_tree)
                ):
                    linewidth = 2.0 if np.all(np.array(chain) < 5) else 1.0
                    ax.plot3D(
                        data["joints"][i, chain, 0],
                        data["joints"][i, chain, 1],
                        data["joints"][i, chain, 2],
                        linewidth=linewidth,
                        color=color,
                    )

                if nb_joints == 55:
                    for chain, color in zip(
                        t2m_left_hand_chain, [mp_colors[pid]] * len(t2m_left_hand_chain)
                    ):
                        ax.plot3D(
                            data["joints"][i, chain, 0],
                            data["joints"][i, chain, 1],
                            data["joints"][i, chain, 2],
                            linewidth=linewidth,
                            color=color,
                        )
                    for chain, color in zip(
                        t2m_right_hand_chain, [mp_colors[pid]] * len(t2m_right_hand_chain)
                    ):
                        ax.plot3D(
                            data["joints"][i, chain, 0],
                            data["joints"][i, chain, 1],
                            data["joints"][i, chain, 2],
                            linewidth=linewidth,
                            color=color,
                        )

            plt.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.png")
            plt.savefig(frame_path, dpi=100)

            if (i + 1) % 400 == 0:
                print(f"已生成 {i+1}/{frame_number} 帧")

        print("合成基础视频流...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        cmd_silent = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-r",
            str(fps),
            "-i",
            f"{temp_dir}/frame_%05d.png",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            temp_silent_video,
        ]
        subprocess.run(cmd_silent, check=True)

        if audio_path and os.path.exists(audio_path):
            print(f"正在合并音频: {os.path.basename(audio_path)}")
            cmd_audio = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                temp_silent_video,
                "-i",
                audio_path,
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                save_path,
            ]
            subprocess.run(cmd_audio, check=True)
        else:
            print("未找到音频或未提供，保存无声视频。")
            shutil.move(temp_silent_video, save_path)

        print(f"视频成功保存至: {save_path}")

    except Exception as e:
        print(f"生成过程出错: {e}")
    finally:
        plt.close(fig)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def plot_t2m(mp_data, save_path, caption, nb_joints=22):
    """
    多路 motion 到 T2M 3D 可视化入口，原先在 utils.intergen_vis 中。
    """
    mp_joint = []
    for i, data in enumerate(mp_data):
        if i == 0:
            joint = data[:, : nb_joints * 3].reshape(-1, nb_joints, 3)
        else:
            joint = data[:, : nb_joints * 3].reshape(-1, nb_joints, 3)

        mp_joint.append(joint)

    plot_3d_motion_alternative(
        save_path,
        t2m_kinematic_chain,
        mp_joint,
        title=caption,
        fps=30,
        nb_joints=nb_joints,
    )

def generate_one_sample(motion_output_multiple, name, result_path, nb_joints=22):
    """
    统一的 sample 可视化入口，原先在 utils.intergen_vis 中。
    """
    os.makedirs(result_path, exist_ok=True)
    save_path = os.path.join(result_path, f"{name}.mp4")

    if motion_output_multiple[0].ndim == 2:
        plot_t2m(motion_output_multiple, save_path, name, nb_joints=nb_joints)
    elif motion_output_multiple[0].ndim == 3:
        plot_t2m(
            [motion[0] for motion in motion_output_multiple],
            save_path,
            name,
            nb_joints=nb_joints,
        )
    else:
        raise ValueError(
            f"Invalid number of motions: {len(motion_output_multiple)}"
        )