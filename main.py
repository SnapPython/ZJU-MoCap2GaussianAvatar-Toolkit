import os
import numpy as np
import cv2
import argparse
from tqdm import tqdm


def process_subject(subj_path, output_root, view_id=0):
    subj_dir_name = os.path.basename(subj_path)
    subject_id = subj_dir_name.split('_')[-1]

    annots_path = os.path.join(subj_path, 'annots.npy')
    if not os.path.exists(annots_path):
        print(f"⚠️ 跳过 {subj_dir_name}: 找不到 annots.npy")
        return

    annots = np.load(annots_path, allow_pickle=True).item()

    # 建立输出结构
    subject_output = os.path.join(output_root, subject_id)
    img_out, msk_out = os.path.join(subject_output, 'images'), os.path.join(subject_output, 'masks')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)

    # ==========================================
    # 1. 提取相机参数
    # ==========================================
    K = annots['cams']['K'][view_id]
    R = annots['cams']['R'][view_id]
    T = annots['cams']['T'][view_id] / 1000.0
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = T.flatten()

    # 官方源码: intrinsic = np.array(camera["intrinsic"])
    np.savez(os.path.join(subject_output, 'cameras.npz'),
             intrinsic=K,
             extrinsic=extrinsic)

    # ==========================================
    # 2. 提取帧列表
    # ==========================================
    frame_list = []
    for f_idx, frame_data in enumerate(annots['ims']):
        frame_list.append({'img_path': frame_data['ims'][view_id], 'frame_idx': f_idx})

    # ==========================================
    # 3. 循环转换与提取 SMPL
    # ==========================================
    global_orients, body_poses, betas, transl = [], [], [], []

    print(f"📦 正在处理序列 {subject_id} (视角: {view_id}, 共 {len(frame_list)} 帧)...")

    for i, item in enumerate(tqdm(frame_list)):
        f_idx = item['frame_idx']

        param_path = os.path.join(subj_path, 'new_params', f'{f_idx}.npy')
        if not os.path.exists(param_path): continue

        data = np.load(param_path, allow_pickle=True).item()

        # 拆解 ZJU 原始位姿以匹配 GA
        p = data['poses'].copy().squeeze()  # 原始是 72 维
        rh = data['Rh'].squeeze()  # 全局旋转 (3维)

        global_orients.append(rh)
        body_poses.append(p[3:])
        betas.append(data['shapes'].squeeze()) # 依然收集，但存的时候只取一个
        transl.append(data['Th'].squeeze() / 1000.0)

        src_img_path = os.path.join(subj_path, item['img_path'])
        src_msk_path = os.path.join(subj_path, 'mask', item['img_path'].replace('.jpg', '.png'))

        img = cv2.imread(src_img_path)
        mask = cv2.imread(src_msk_path, 0)

        if img is None or mask is None: continue

        if mask.max() == 1:
            mask = mask * 255

        img[mask < 128] = 0
        cv2.imwrite(os.path.join(img_out, f"{i:06d}.png"), img)
        cv2.imwrite(os.path.join(msk_out, f"{i:06d}.png"), mask)

    if global_orients:
        np.savez(os.path.join(subject_output, 'poses_optimized.npz'),
                 global_orient=np.array(global_orients),
                 body_pose=np.array(body_poses),
                 betas=np.array(betas)[0],  # ✨ 这里是核心修正：只取第一帧的身材参数
                 transl=np.array(transl))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="./data/raw")
    parser.add_argument("--out_dir", default="./data/prepared")
    parser.add_argument("--view", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.raw_dir):
        print(f"❌ 找不到原始数据目录: {args.raw_dir}")
        return

    subjects = [d for d in os.listdir(args.raw_dir)
                if os.path.isdir(os.path.join(args.raw_dir, d)) and not d.startswith('.')]

    if not subjects:
        print(f"⚠️ 在 {args.raw_dir} 中未发现任何有效的数据文件夹")
        return

    for s in subjects:
        process_subject(os.path.join(args.raw_dir, s), args.out_dir, args.view)

    print("\n✨ Toolkit 处理完毕！")


if __name__ == "__main__":
    main()