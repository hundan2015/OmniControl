# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric,recover_from_rot
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate
from utils.text_control_example import collate_all
from os.path import join as pjoin


def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    n_frames = 196
    is_using_data = not any([args.text_prompt])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')

    hints = None
    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        if args.text_prompt == 'predefined':
            # generate hint and text
            texts, hints = collate_all(n_frames, args.dataset)
            args.num_samples = len(texts)
            if args.cond_mode == 'only_spatial':
                # only with spatial control signal, and the spatial control signal is defined in utils/text_control_example.py
                texts = ['' for i in texts]
            elif args.cond_mode == 'only_text':
                # only with text prompt, and the text prompt is defined in utils/text_control_example.py
                hints = None
        else:
            # otherwise we use text_prompt
            texts = [args.text_prompt]
            args.num_samples = 1
            hint = None

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        # t2m
        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        if hints is not None:
            collate_args = [dict(arg, hint=hint) for arg, hint in zip(collate_args, hints)]

        _, model_kwargs = collate(collate_args)

    for k, v in model_kwargs['y'].items():
        if torch.is_tensor(v):
            model_kwargs['y'][k] = v.to(dist_util.dev())

    all_motions = []
    all_lengths = []
    all_text = []
    all_hint = []
    all_hint_for_vis = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop
        ##############
        # Custom input
        ##############

        def make_raw_data(data_):
            dataset = data.dataset.t2m_dataset
            return (data_-dataset.mean)/dataset.std
        def make_ano_data(data_):
            dataset = data.dataset.t2m_dataset
            return (data_)*dataset.std+dataset.mean

        skip_timesteps = 980
        
        # 添加自定义数据
        ori_loaded_data_np = np.load(fr"dataset\HumanML3D\new_joint_vecs\000000.npy")
        ori_loaded_data = torch.from_numpy(ori_loaded_data_np).clone().to("cuda:0")
        
        loaded_data = make_raw_data(ori_loaded_data_np)
        # loaded_data = make_ano_data(loaded_data)
        loaded_data = torch.from_numpy(loaded_data).to("cuda:0")
        
        # 生成自定义hints
        loaded_pos_data = recover_from_ric(ori_loaded_data, 22)
        loaded_pos_data[:, :20, :] = 0
        loaded_pos_data[:, 21, :] = 0
        # loaded_pos_data = loaded_pos_data.view(-1, 3*22)
        loaded_pos_data = torch.flatten(loaded_pos_data,start_dim=1,end_dim=-1)
        
        if loaded_pos_data.shape[0] < 196:
            zeros = torch.zeros(
                (196-loaded_pos_data.shape[0], loaded_pos_data.shape[1])).to("cuda:0")
            loaded_pos_data = torch.cat((loaded_pos_data, zeros))
        loaded_pos_data = loaded_pos_data.unsqueeze(0)
        
        model_kwargs['y']['hint'] = loaded_pos_data
            
        if loaded_data.shape[0] < 196:
            zeros = torch.zeros(
                (196-loaded_data.shape[0], loaded_data.shape[1])).to("cuda:0")
            loaded_data = torch.cat((loaded_data, zeros))
        loaded_data = loaded_data.unsqueeze(0)
        loaded_data = loaded_data.unsqueeze(0)
        loaded_data = loaded_data.permute(0, 3, 1, 2)
        
        '''
        Noise: 直接作为中间结果输入，不添加任何噪声
        init_image: 作为初始情况输入，根据skip_timesteps剩下的添加噪声
        hint: Tensor(batch_count,frame_count,joint_count * 3)
        '''
        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=skip_timesteps,  # 0 is the default value - i.e. don't skip any step
            init_image=loaded_data,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        ##############
        # Original
        
        # sample = sample_fn(
        #     model,
        #     (args.batch_size, model.njoints, model.nfeats, n_frames),
        #     clip_denoised=False,
        #     model_kwargs=model_kwargs,
        #     skip_timesteps=skip_timesteps,  # 0 is the default value - i.e. don't skip any step
        #     init_image=None,
        #     progress=True,
        #     dump_steps=None,
        #     noise=None,
        #     const_noise=False,
        # )

        sample = sample[:, :263]
        # Recover XYZ *positions* from HumanML3D vector representation
        
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            ##########
            np.save("sample_original.npy",sample.numpy())
            
            from common.skeleton import Skeleton
            
            n_raw_offsets = torch.from_numpy(paramUtil.t2m_raw_offsets)
            kinematic_chain = paramUtil.t2m_kinematic_chain
            tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, sample.device)
            from paramUtil import tgt_offsets
            tgt_offsets_now = torch.from_numpy(tgt_offsets).clone()
            # 脊柱缩放
            tgt_offsets_now[3] *= 1.5
            tgt_offsets_now[6] *= 1.5
            tgt_offsets_now[9] *= 1.5

            # 手臂
            for i in range(14,22):
                tgt_offsets_now[i] *= 1.7

            tgt_skel.set_offset(tgt_offsets_now)
            # print(tgt_skel.offset()[3])
            joint_pos = recover_from_rot(
                sample.view(-1, 263), n_joints, tgt_skel)
            joint_pos = joint_pos.view(args.batch_size, n_frames, n_joints, -1)
            joint_pos.unsqueeze_(0)
            sample = joint_pos
            # sample = recover_from_ric(sample, n_joints)
            ##########
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
            

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]

            if 'hint' in model_kwargs['y']:
                hint = model_kwargs['y']['hint']
                # denormalize hint
                if args.dataset == 'humanml':
                    spatial_norm_path = './dataset/humanml_spatial_norm'
                elif args.dataset == 'kit':
                    spatial_norm_path = './dataset/kit_spatial_norm'
                else:
                    raise NotImplementedError('unknown dataset')
                raw_mean = torch.from_numpy(np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))).cuda()
                raw_std = torch.from_numpy(np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))).cuda()
                mask = hint.view(hint.shape[0], hint.shape[1], n_joints, 3).sum(-1) != 0
                hint = hint * raw_std + raw_mean
                hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3) * mask.unsqueeze(-1)
                hint = hint.view(hint.shape[0], hint.shape[1], -1)
                # ---
                all_hint.append(hint.data.cpu().numpy())
                hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3)
                all_hint_for_vis.append(hint.data.cpu().numpy())

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    if 'hint' in model_kwargs['y']:
        all_hint = np.concatenate(all_hint, axis=0)[:total_num_samples]
        all_hint_for_vis = np.concatenate(all_hint_for_vis, axis=0)[:total_num_samples]
    
    if len(all_hint) != 0:
        from utils.simple_eval import simple_eval
        results = simple_eval(all_motions, all_hint, n_joints)
        print(results)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths, "hint": all_hint_for_vis,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    sample_files = []
    num_samples_in_out_file = 7

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            if 'hint' in model_kwargs['y']:
                hint = all_hint_for_vis[rep_i*args.batch_size + sample_i]
            else:
                hint = None
            save_file = sample_file_template.format(sample_i, rep_i)
            print(sample_print_template.format(caption, sample_i, rep_i, save_file))
            animation_save_path = os.path.join(out_path, save_file)
            plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps, hint=hint)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)

        sample_files = save_multiple_samples(args, out_path,
                                               row_print_template, all_print_template, row_file_template, all_file_template,
                                               caption, num_samples_in_out_file, rep_files, sample_files, sample_i)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='train')
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
