import numpy as np


class CubeDataset:
    def __init__(self, opt, mu, A, new_noise=False, norm=True, num_examples=3000):
        super().__init__()
        self.opt = opt
        self.mu = np.reshape(mu, (3, 1))
        self.A = A
        self.num_point = opt['num_point']
        self.cube_template = opt['base_scale'] * (np.random.rand(3, self.num_point) -
                                                  0.5)
        self.new_noise = new_noise
        self.num_examples = num_examples
        self.norm = norm
        self.point_clouds = self.next_batch(self.num_examples)[0]

    def next_batch(self, batch_size):
        input_noise = np.random.randn(3, batch_size)
        param = self.mu + self.A.dot(input_noise)
        target = np.reshape(self.cube_template,
                            (1, 3, self.num_point)) * np.reshape(
                                np.transpose(param), (batch_size, 3, 1))
        if self.norm:
            target /= 2 * np.sqrt(np.max(np.sum(target ** 2, axis=1, keepdims=True)))
        target = np.transpose(target, axes=(0, 2, 1))
        if self.new_noise:
            input_noise = np.random.randn(batch_size, 3)
        else:
            input_noise = np.transpose(input_noise)

        return target, input_noise, np.squeeze(param)
    
    
class TableDataset:
    def __init__(self, opt):
        self.opt = opt
        zero = np.array([0, 0, 0])
        one = np.array([1, 1, 1])
        self.cube_desktop = gen_box_surface_pc(zero, one, opt['num_point_d'])
        self.cube_leg = gen_box_surface_pc(zero, one, opt['num_point_l'])
        self.param_list = [
            gen_table_param(opt['mu'], opt['sigma'], opt['tweak_weight'], opt['type'])
            for _ in range(opt['num_example'])
        ]
        self.norm = opt['norm'] if 'norm' in opt.keys() else False
        
        if self.norm:
            for index, (s, c) in enumerate(self.param_list):
                pc = self.deform(s, c)
                mean = np.mean(pc, axis=1, keepdims=True)
                #max_radius = 2 * (pc - mean).pow(2).sum(dim=0).max().sqrt()
                max_radius = np.sqrt(np.max(np.sum((pc - mean) ** 2, axis=0)))
                max_radius *= 2
                c = (c - mean.transpose()) / max_radius
                s /= max_radius
                self.param_list[index] = (s, c)
        self.mean, self.cov, self.A = self.calc_statistic()
        
        self.cnt = 0
        self.num_examples = len(self.param_list)
        self.point_clouds = self.next_batch(self.num_examples)[0]
                
    def next_batch(self, batch_size):
        pc_list = []
        noise_list = []
        if self.cnt + batch_size >= self.num_examples:
            self.cnt = 0
        params = self.param_list[self.cnt: self.cnt + batch_size]
        self.cnt += batch_size
        for b in range(batch_size):
            s, c = params[b]
            pc = self.deform(s, c)
            if 'noise_dim' not in self.opt.keys():
                input_noise = np.random.randn(30)
            else:
                input_noise = np.random.randn(self.opt['noise_dim'])
            pc_list.append(pc.transpose())
            noise_list.append(input_noise)
            
        return np.stack(pc_list, axis=0), params, 1

    def __len__(self):
        return len(self.param_list)

    def deform(self, s, c):
        part_box = c[0].reshape((3, 1)) + s[0].reshape(
            (3, 1)) * self.cube_desktop
        pc_list = [part_box]
        for i in range(1, 5):
            part_box = c[i].reshape((3, 1)) + s[i].reshape(
                (3, 1)) * self.cube_leg
            pc_list.append(part_box)
        pc = np.concatenate(pc_list, axis=1)
        return pc
    
    def calc_statistic(self):
        param_array = np.stack(self.param_list,
                               axis=0).transpose(0, 2, 1, 3).reshape(
                                   self.opt['num_example'], -1).transpose()
        cov = np.cov(param_array)
        mean = np.mean(param_array, axis=1)
        
        w, v = np.linalg.eigh(cov)
        w_pos = np.clip(w, 0, np.infty)
        w_pos_root = np.sqrt(w_pos)
        A = v.dot(w_pos_root * np.eye(w.shape[0]))
        return mean, cov, A
    
    
def gen_box_surface_pc(center, scale, num_points):
    center = np.reshape(center, (3, 1))
    scale = np.reshape(scale, (3,1))

    nums = [num_points // 6] * 5
    nums.append(num_points - sum(nums))
    faces = []
    for n in nums:
        faces.append(np.random.rand(2, n) - 0.5)
    pc_faces = []
    for i in range(3):
        face_tmp = np.zeros((3, faces[2 * i].shape[1]))
        axes = [0, 1, 2]
        axes.remove(i)
        face_tmp[axes, :] = faces[2 * i]
        face_tmp[i, :] = 0.5
        pc_faces.append(face_tmp)
        face_tmp = np.zeros((3, faces[2 * i + 1].shape[1]))
        face_tmp[axes, :] = faces[2 * i + 1]
        face_tmp[i, :] =  -0.5
        pc_faces.append(face_tmp)
    pc = np.concatenate(pc_faces, axis=1)
    pc = center + pc * scale
    return pc

def gen_table_param(mu_s0, sigma, tweak=0.0, param_type=None):
    def _pseudo_trunc_norm(size, th = 2.0):
        return np.clip(np.random.randn(*size), -th, th)
    
    if param_type == None:
        s = np.zeros((5, 3))
        c = np.zeros((5, 3))
        # s_0x ~ N()
        s[0, 0] = np.max((np.random.randn(1) * sigma + mu_s0, 0.1 * mu_s0))
        # 0.5 s_0x < s_0y < 1.5 s_0x
        s[0, 1] = np.random.rand(1) * s[0, 0] + 0.5 * s[0, 0]
        # 0.05 * (s_0x + s_0y) < s_0z < 0.125 * (s_0x + s_0y)
        s[0, 2] = np.max((np.random.rand(1) * (s[0, 0] + s[0, 1]) / 8, (s[0, 0] + s[0, 1]) / 8))
        # s_1x < 0.25 s_0x, s_1y < 0.25 s_0y
        s[1, 0] = np.random.rand(1) * s[0, 0] / 4
        s[1, 0] = np.max((s[1, 0], s[0, 0] / 16))
        s[1, 1] = np.random.rand(1) * s[0, 1] / 4
        s[1, 1] = np.max((s[1, 1], s[0, 1] / 16))
        # c_0z ~ N(2s_0z, 0.25 s_0z^2), c_0z > s_0z
        c[0, 2] = np.random.randn(1) * 0.25 * s[0, 2] * s[0, 2] + 2 * s[0, 2]
        c[0, 2] = np.max((c[0, 2], s[0, 2]))

        # equations
        c[0, 0:2] = 0 + tweak * s[0, 2] * _pseudo_trunc_norm((1,))
        s[1, 2] = 2 * c[0, 2] * (1 + tweak * _pseudo_trunc_norm((1,)))
        c[1:, 2] = -0.5 * s[0, 2] *( 1 + tweak * _pseudo_trunc_norm((4,)))
        s[2:, :] = s[1, :] * (1 + tweak * _pseudo_trunc_norm((3, 3)))

        c[[1, 3], 0] = - (0.5 * s[0, 0] - 0.5 * s[1, 0]) * (1 + tweak * _pseudo_trunc_norm((2,)))
        c[[2, 4], 0] = (0.5 * s[0, 0] - 0.5 * s[1, 0]) * (1 + tweak * _pseudo_trunc_norm((2,)))
        c[[1, 2], 1] = - (0.5 * s[0, 1] - 0.5 * s[1, 1]) * (1 + tweak * _pseudo_trunc_norm((2,)))
        c[[3, 4], 1] = (0.5 * s[0, 1] - 0.5 * s[1, 1]) * (1 + tweak * _pseudo_trunc_norm((2,)))
        return s, c

    elif param_type == 'free':
        s = np.zeros((5, 3))
        c = np.zeros((5, 3))
        # s_0x ~ N()
        s[0, 0] = np.max((np.random.randn(1) * sigma + mu_s0, 0.5 * mu_s0))
        # 0.5 s_0x < s_0y < 1.5 s_0x
        s[0, 1] = np.max((np.random.randn(1) * sigma + mu_s0, 0.5 * mu_s0))
        # 0.05 * (s_0x + s_0y) < s_0z < 0.125 * (s_0x + s_0y)
        s[0, 2] = 0.5 * np.max((np.random.randn(1) * sigma + mu_s0, 0.5 * mu_s0))
        # s_1x < 0.25 s_0x, s_1y < 0.25 s_0y
        s[1, 0] = np.random.rand(1) * s[0, 0] / 4
        s[1, 0] = np.max((s[1, 0], s[0, 0] / 16))
        s[1, 1] = np.random.rand(1) * s[0, 1] / 4
        s[1, 1] = np.max((s[1, 1], s[0, 1] / 16))
        # c_0z ~ N(2s_0z, 0.25 s_0z^2), c_0z > s_0z
        c[0, 2] = 0.55 * s[0, 2]

        # equations
        c[0, 0:2] = 0 + tweak * s[0, 2] * _pseudo_trunc_norm((1,))
        s[1, 2] = 2 * c[0, 2] * (1 + tweak * _pseudo_trunc_norm((1,)))
        c[1:, 2] = -0.5 * s[0, 2] *( 1 + tweak * _pseudo_trunc_norm((4,)))
        s[2:, :] = s[1, :] * (1 + tweak * _pseudo_trunc_norm((3, 3)))

        min_tmp_x = 0.1 * s[0, 0] + 0.5 * s[1, 0]
        max_tmp_x = 0.5 * s[0, 0] - 0.5 * s[1, 0]
        min_tmp_y = 0.1 * s[0, 1] + 0.5 * s[1, 1]
        max_tmp_y = 0.5 * s[0, 1] - 0.5 * s[1, 1]
        leg_c_x = np.random.rand(1) * (max_tmp_x - min_tmp_x) + min_tmp_x
        leg_c_y = np.random.rand(1) * (max_tmp_y - min_tmp_y) + min_tmp_y
        c[[1, 3], 0] = - leg_c_x * (1 + tweak * _pseudo_trunc_norm((2,)))
        c[[2, 4], 0] = leg_c_x * (1 + tweak * _pseudo_trunc_norm((2,)))
        c[[1, 2], 1] = - leg_c_y * (1 + tweak * _pseudo_trunc_norm((2,)))
        c[[3, 4], 1] = leg_c_y * (1 + tweak * _pseudo_trunc_norm((2,)))
        return s, c
    else:
        raise NotImplementedError('Table dataset type %s not implemented!' % (param_type))