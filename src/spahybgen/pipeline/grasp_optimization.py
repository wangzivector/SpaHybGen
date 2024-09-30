import pickle 
import plotly
import torch
from tqdm import tqdm
from datetime import datetime
from spahybgen.optimization import SpactialOptimization
import spahybgen.utils.utils_plotly as ut_plotly
import numpy as np
import os

class GraspOptimization(SpactialOptimization):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def set_optimization_weights(self, weight_list):
        if weight_list is None: return
        for k, v in weight_list.items(): setattr(self, k, v)

    def run_optimization(self, scene_infer_map, target_appr_matrix=[[0, 1, 0], [1, 0, 0], [0, 0, -1]], 
            pent_check_split=.80, max_iter=300, tb_writer=None, running_name='trail', tqdm_disable=False):
        scene_infer_map = torch.from_numpy(scene_infer_map).float()
        self.reset_scene(scene_infer_map, target_appr_matrix)
        q_trajectory = []
        with torch.no_grad():
            opt_q = self.get_opt_q()
            q_trajectory.append(opt_q.clone().detach())
        
        losses_dict = None
        for i_iter in tqdm(range(max_iter), desc=running_name, bar_format="{l_bar} {bar} [{elapsed_s:.3f}s, {rate_fmt}{postfix}]", disable=tqdm_disable):
            penetration_check = i_iter > (max_iter * pent_check_split)
            self.step(penetration_check)

            with torch.no_grad():
                opt_q = self.get_opt_q()
                q_trajectory.append(opt_q.clone().detach())

                if losses_dict is None: 
                    losses_dict = {}
                    for key, value in self.losses.items(): 
                        losses_dict[key] = np.expand_dims(value.detach().cpu().clone().numpy(), axis=-1)
                else:
                    for key, value in self.losses.items():
                        pre_stack = losses_dict[key]
                        losses_dict[key] = np.concatenate((pre_stack, np.expand_dims(value.detach().cpu().clone().numpy(), axis=-1)), axis=1)

                if tb_writer is not None:
                    for key, value in self.losses.items():
                        tb_writer.add_scalar(tag=f'overall/{running_name}/{key}', scalar_value=value.mean(), global_step=i_iter)
                    loss = self.losses['loss_all'].detach().cpu().tolist()
                    tb_writer.add_scalar(tag=f'overall/{running_name}/index', scalar_value=loss.index(min(loss)), global_step=i_iter)
                    tb_writer.add_scalar(tag=f'overall/{running_name}/loss_min', scalar_value=min(loss), global_step=i_iter)
                    tb_writer.add_scalar(tag=f'overall/{running_name}/loss_mean', scalar_value=torch.tensor(loss).mean(), global_step=i_iter)
                    for i_loss in range(len(loss)):
                        tb_writer.add_scalar(tag=f'loss/{running_name}/{i_loss}', scalar_value=loss[i_loss], global_step=i_iter)

        q_trajectory = torch.stack(q_trajectory, dim=0).transpose(0, 1).detach().cpu().clone().numpy()
        losses_dict['sort_ids'] = torch.sort(self.losses['loss_all'])[1].cpu().numpy()
        return q_trajectory, losses_dict


    def visualize_optimization(self, visulize_mode, losses_dict, q_trajectory=None, filedir='real-grasp', trial_id=0):
        if visulize_mode == 'SAVE':
            filename = filedir + '/grasp-{}-{}'.format(trial_id, datetime.now().strftime("%m-%d-%H:%M:%S"))
            os.makedirs(filename)
            with open(os.path.join(filename, 'losses_dict.dump'), 'wb') as f: pickle.dump(losses_dict, f)
            np.save(os.path.join(filename, 'qtrajectory.npy'), q_trajectory)
            np.save(os.path.join(filename, 'inference.npy'), self.scene_infer_map.detach().cpu().numpy())
            return
        
        ## Visualization Results
        vis_data = []
        ## -- Voxel -- 
        scene_grid = self.scene_grid.detach().cpu().numpy()
        vis_data.append(ut_plotly.plot_volume_mesh(scene_grid, volume_length = self.voxel_size*self.voxel_num, alpha=1.0, color_length=45,light=dict(ambient=0.6, diffuse=0.6, roughness = 1.0, specular=0.05, fresnel=4.0)))
        ## -- Scene Surface Points -- 
        # vis_data.append(ut_plotly.plot_point_cloud(
        #     self.batch_scene_surface_points[0][0].detach().cpu().numpy(), color='black', size=2))
        ## -- Tip Focal Points -- 
        vis_data.append(ut_plotly.plot_point_cloud(self.batch_tip_focal_points[0][0].detach().cpu().numpy(), color='rgb(153, 153, 153)', size=3))
        ## -- Wrench selected -- 
        # wren_volume, idw = torch.zeros_like(self.scene_grid.detach()), self.index_good_wrenches_focal
        # wren_volume[idw[:, 0], idw[:, 1], idw[:, 2]] = self.scene_wrench_full[idw[:, 0], idw[:, 1], idw[:, 2]]
        # vis_data.append(ut_plotly.plot_volume_heat(wren_volume.detach().cpu().numpy(), volume_length=0.4, colormap='Blues', light=dict(ambient=0.6, diffuse=0.6, roughness = 1.0, specular=0.05, fresnel=4.0), alpha=0.60))
        ## -- Wrench Focal Points -- 
        # vis_data.append(ut_plotly.plot_point_cloud(self.wrench_focal_points.detach().cpu().numpy(), color='red', size=12))
        ## -- Selected Wrench Centers for Force Closure -- 
        # batch_WrenCent_points_visualization = self.batch_WrenCent_points_visualization
        # vis_data.append(ut_plotly.plot_point_cloud(pts=batch_WrenCent_points_visualization.detach().cpu().numpy(), color='green', size=16))
        ## -- Good Grasp Poses of Lower Overall Loss [default] -- 
        vis_size = self.batch//5
        indx_good = losses_dict['sort_ids'][:vis_size]
        for i, id_sort in enumerate(indx_good): 
            vis_data += self.get_current_plotly_data(
                index=id_sort, opacity=1.0, color=f'rgb({int(250*(1 - i/vis_size))}, {0}, 200)', 
                text="RANK-{} #All:{:.3} #FQH:{:.3} #QH:{:.3} #RH:{:.3} #WH:{:.3} #Pen:{:.3} #JR:{:.3} #CK:{:.3} #FC:{:.3} #ApB:{:.3} #CTS:{:.3}".format( 
                    i,
                    losses_dict['loss_all'][id_sort][-1],
                    losses_dict['loss_FQH'][id_sort][-1],
                    losses_dict['loss_QH'][id_sort][-1],
                    losses_dict['loss_RH'][id_sort][-1],
                    losses_dict['loss_WH'][id_sort][-1],
                    losses_dict['loss_penet'][id_sort][-1],
                    losses_dict['loss_joint_range'][id_sort][-1],
                    losses_dict['loss_custom_kine'][id_sort][-1],
                    losses_dict['loss_force_closure'][id_sort][-1],
                    losses_dict['loss_approach_bias'][id_sort][-1],
                    losses_dict['CTS'][id_sort][-1],
                )
            )

        ## Contact Point's Normals and its Current Matches [default]
        # (batch_contact_points, tip_rotvet_at_CTs, batch_contact_normals) = self.tip_rotvet_at_CTs_visulization # [batch, cp, 3]
        # for i in range(5): vis_data.append(ut_plotly.plot_point_cloud(
        #     pts=(batch_contact_points.reshape(-1, 3) + 0.003 * i * tip_rotvet_at_CTs.reshape(-1, 3)).detach().cpu().numpy(), color='red'))
        # for i in range(5): vis_data.append(ut_plotly.plot_point_cloud(
        #     pts=(batch_contact_points.reshape(-1, 3) + 0.003 * i * batch_contact_normals.reshape(-1, 3)).detach().cpu().numpy(), color='blue'))
            
        best_id = losses_dict['sort_ids'][0]
        for key, value in losses_dict.items(): 
            if key != 'sort_ids': print("[Best Grasp]: [{}]: {}".format(key, value[best_id][-1]))

        fig = plotly.graph_objects.Figure(data=vis_data)

        filename = filedir + '/grasp-{}-{}'.format(trial_id, datetime.now().strftime("%m-%d-%H:%M:%S"))
        if visulize_mode == 'NONE':
            print("No optimization visulization.")
            return fig
        elif visulize_mode == 'ONLINE':
            fig.show()
            return fig
        if visulize_mode == 'OFFLINE':
            os.mkdir(filename)
            plotly.offline.plot(fig, auto_open= False, filename=os.path.join(filename, 'vis-grasp.html'))
        elif visulize_mode == 'ONOFFLINE':
            os.mkdir(filename)
            plotly.offline.plot(fig, auto_open= True, filename=os.path.join(filename, 'vis-grasp.html'))
        elif visulize_mode == 'SAVE':
            os.mkdir(filename)
            with open(os.path.join(filename, 'losses_dict.dump'), 'wb') as f: pickle.dump(losses_dict, f)
            np.save(os.path.join(filename, 'qtrajectory.npy'), q_trajectory)
            np.save(os.path.join(filename, 'inference.npy'), self.scene_infer_map.detach().cpu().numpy())
            # fig.show()
            # plotly.offline.plot(fig, auto_open= False, filename=filename + '.html')
            # fig.write_image(filename + '.png')

