import sys, os
import numpy as np
import argparse, json
from types import SimpleNamespace as SN
import time
import bmesh
import bpy
from os.path import join, dirname, realpath, exists
from mathutils import Matrix, Vector, Quaternion, Euler
import math
from pickle import load
sys.path.append('/data1/liuguanze/anaconda3/lib/python3.7/site-packages/')
import h5py
import scipy.io
start_time = None

def mkdir_safe(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

def setState0():
    for ob in bpy.data.objects.values():
        ob.select=False
    bpy.context.scene.objects.active = None

def log_message(message):
    elapsed_time = time.time() - start_time
    print("[%.2f s] %s" % (elapsed_time, message))

sorted_parts = ['hips','leftUpLeg','rightUpLeg','spine','leftLeg','rightLeg',
                'spine1','leftFoot','rightFoot','spine2','leftToeBase','rightToeBase',
                'neck','leftShoulder','rightShoulder','head','leftArm','rightArm',
                'leftForeArm','rightForeArm','leftHand','rightHand','leftHandIndex1' ,'rightHandIndex1']
# order
part_match = {'root':'root', 'bone_00':'Pelvis', 'bone_01':'L_Hip', 'bone_02':'R_Hip',
              'bone_03':'Spine1', 'bone_04':'L_Knee', 'bone_05':'R_Knee', 'bone_06':'Spine2',
              'bone_07':'L_Ankle', 'bone_08':'R_Ankle', 'bone_09':'Spine3', 'bone_10':'L_Foot',
              'bone_11':'R_Foot', 'bone_12':'Neck', 'bone_13':'L_Collar', 'bone_14':'R_Collar',
              'bone_15':'Head', 'bone_16':'L_Shoulder', 'bone_17':'R_Shoulder', 'bone_18':'L_Elbow',
              'bone_19':'R_Elbow', 'bone_20':'L_Wrist', 'bone_21':'R_Wrist', 'bone_22':'L_Hand', 'bone_23':'R_Hand'}

part2num = {part:(ipart+1) for ipart,part in enumerate(sorted_parts)}

class Misha():
    def __init__(self):
        self.kintree = {
           -1 : (-1,'root'),
            0 : (-1, 'Pelvis'),
            1 : (0, 'L_Hip'),
            2 : (0, 'R_Hip'),
            3 : (0, 'Spine1'),
            4 : (1, 'L_Knee'),
            5 : (2, 'R_Knee'),
            6 : (3, 'Spine2'),
            7 : (4, 'L_Ankle'),
            8 : (5, 'R_Ankle'),
            9 : (6, 'Spine3'),
            10 : (7, 'L_Foot'),
            11 : (8, 'R_Foot'),
            12 : (9, 'Neck'),
            13 : (9, 'L_Collar'),
            14 : (9, 'R_Collar'),
            15 : (12, 'Head'),
            16 : (13, 'L_Shoulder'),
            17 : (14, 'R_Shoulder'),
            18 : (16, 'L_Elbow'),
            19 : (17, 'R_Elbow'),
            20 : (18, 'L_Wrist'),
            21 : (19, 'R_Wrist'),
            22 : (20, 'L_Hand'),
            23 : (21, 'R_Hand')
        }
        self.n_bones = 24
        with open('/data1/liuguanze/depth_point_cloud/data/male_model.pkl', 'rb') as f:
            male_model = load(f, encoding='latin1')
        self.faces = male_model['f']

    def create_bpy_mesh(self, vertice, scene):
        # vertice [6890, 3]
        mesh = bpy.data.meshes.new("mesh")
        obj = bpy.data.objects.new("MyObject", mesh)
        scene.objects.link(obj)
        scene.objects.active = obj
        vertice = [(vertex[0], vertex[1], vertex[2]) for vertex in vertice]
        obj.select = True
        mesh = bpy.context.object.data
        bm = bmesh.new()
        for v in vertice:
            bm.verts.new(v)
        bm.to_mesh(mesh)
        bm.free()

    def change_existing_mesh(self, vertices, scene, mesh):
        vertices = [(vertex[0], vertex[1], vertex[2]) for vertex in vertices]
        num_verts = len(vertices)
        for i in range(num_verts):
            mesh.vertices[i].co = vertices[i]
        """
        bm = bmesh.new()
        bpy.ops.object.mode_set(mode='EDIT')
        bm.from_mesh(mesh)
        bpy.ops.object.mode_set(mode='OBJECT')
        # print('the type of the bmesh data:', type(bm.verts))
        # BMVertSeq
        bm.verts.ensure_lookup_table()
        for index, vertex in enumerate(vertices):
            bm.verts[index].co = vertex
        bm.to_mesh(mesh)
        """
        return mesh

    def render(self, opt):
        global start_time
        idx = opt.idx
        ishape = opt.ishape
        stride = opt.stride
        start_time = time.time()
        # import configuration
        sys.path.append('/data1/liuguanze/depth_point_cloud/src/datasets/')
        import config
        params = config.load_file('config.copy', 'SYNTH_DATA')
        resy = params['resy']
        resx = params['resx']
        tmp_path = params['tmp_path']
        output_path = params['output_path']
        output_types = params['output_types']
        openexr_py2_path = params['openexr_py2_path']
        dfaust_path = params['dfaust_path']
        # try to separate the dfaust dataset and render in multiple times
        fbegin = opt.fbegin
        fend = opt.fend
        log_message("Setup Blender")
        sh_dir = join(tmp_path, 'spher_harm')
        if not exists(sh_dir):
            mkdir_safe(sh_dir)
        # (runpass, idx) = divmod(idx, len(idx_info))
        sh_dst = join(sh_dir, 'sh.osl')
        # sh_dst = join(sh_dir, 'sh_%02d_%05d.osl' % (runpass, idx))
        # os.system('cp spher_harm/sh.osl %s' % sh_dst)
        scene = bpy.data.scenes['Scene']
        scene.render.engine = 'CYCLES'
        bpy.data.materials['Material'].use_nodes = True
        scene.cycles.shading_system = True
        scene.use_nodes = True
        log_message("Building materials tree")
        mat_tree = bpy.data.materials['Material'].node_tree
        self.create_sh_material(mat_tree, sh_dst, img=None)
        res_paths, depth_out, segm_out = self.create_composite_nodes(scene.node_tree, params, img=None, idx=idx)
        log_message("Initializing scene")
        camera_distance = np.random.normal(8.0, 1)
        params['camera_distance'] = camera_distance
        ob, obname, arm_ob, cam_ob = self.init_scene(scene, params)
        setState0()
        ob.select = True
        bpy.context.scene.objects.active = ob
        segmented_materials = True  # True: 0-24, False: expected to have 0-1 bg/fg
        log_message("Creating materials segmentation")
        # create material segmentation
        if segmented_materials:
            materials = self.create_segmentation(ob, params)
            prob_dressed = {'leftLeg': .5, 'leftArm': .9, 'leftHandIndex1': .01,
                            'rightShoulder': .8, 'rightHand': .01, 'neck': .01,
                            'rightToeBase': .9, 'leftShoulder': .8, 'leftToeBase': .9,
                            'rightForeArm': .5, 'leftHand': .01, 'spine': .9,
                            'leftFoot': .9, 'leftUpLeg': .9, 'rightUpLeg': .9,
                            'rightFoot': .9, 'head': .01, 'leftForeArm': .5,
                            'rightArm': .5, 'spine1': .9, 'hips': .9,
                            'rightHandIndex1': .01, 'spine2': .9, 'rightLeg': .5}
        orig_pelvis_loc = (arm_ob.matrix_world.copy() * arm_ob.pose.bones[obname + '_Pelvis'].head.copy()) - Vector(
            (-1., 1., 1.))
        orig_cam_loc = cam_ob.location.copy()
        get_real_frame = lambda ifr: ifr
        # unblocking both the pose and the blendshape limits
        for k in ob.data.shape_keys.key_blocks.keys():
            bpy.data.shape_keys["Key"].key_blocks[k].slider_min = -10
            bpy.data.shape_keys["Key"].key_blocks[k].slider_max = 10
        log_message("Loading body data")
        cmu_parms, names = self.load_body_data(dfaust_path, ob, obname, idx=idx)
        names = ['female50021_knees', 'female50021_light_hopping_stiff', 'female50021_one_leg_jump', 'female50021_one_leg_loose', 'female50021_punching', 'female50021_running_on_spot', 'female50021_shake_arms', 'female50021_shake_hips', 'female50021_shake_shoulders', 'female50022_hips', 'female50022_jiggle_on_toes', 'female50022_jumping_jacks', 'female50022_knees', 'female50022_light_hopping_loose', 'female50022_light_hopping_stiff', 'female50022_one_leg_jump', 'female50022_one_leg_loose', 'female50022_punching', 'female50022_running_on_spot', 'female50022_shake_arms', 'female50022_shake_hips', 'female50022_shake_shoulders', 'female50025_chicken_wings', 'female50025_hips', 'female50025_jiggle_on_toes', 'female50025_knees', 'female50025_light_hopping_loose', 'female50025_light_hopping_stiff', 'female50025_one_leg_jump', 'female50025_one_leg_loose', 'female50025_punching', 'female50025_running_on_spot', 'female50025_shake_arms', 'female50025_shake_hips']
        scene.objects.active = arm_ob
        orig_trans = np.asarray(arm_ob.pose.bones[obname + '_Pelvis'].location).copy()
        # create output directory
        if not exists(output_path):
            mkdir_safe(output_path)
        scs = []
        for mname, material in materials.items():
            scs.append(material.node_tree.nodes['Script'])
            scs[-1].filepath = sh_dst
            scs[-1].update()
        mesh = ob.to_mesh(scene, True, 'PREVIEW')
        scene.objects.unlink(ob)
        ob.select = False
        bpy.data.objects.remove(ob)
        for name in names:
            log_message("Processing Sequence Data %s" % name)
            data = cmu_parms[name]
            sequence_length = data.shape[-1]
            log_message("Computing how many frames to allocate")
            log_message("Allocating %d frames in mat file" % sequence_length)
            matfile_info = join(output_path, name + "_c%04d_info.mat" % (ishape + 1))
            matfile_depth = join(output_path, name + "_c%04d_depth.mat" % (ishape + 1))
            matfile_segm = join(output_path, name + "_c%04d_segm.mat" % (ishape + 1))
            log_message('Working on %s' % matfile_info)
            # allocate
            dict_info = {}
            dict_depth = {}
            dict_segm = {}
            dict_info['camLoc'] = np.empty(3)  # (1, 3)
            # dict_info['gender'] = np.empty(N, dtype='uint8')  # 0 for male, 1 for female
            dict_info['sequence'] = name.replace(" ", "") + "_c%04d" % (ishape + 1)
            dict_info['vertice'] = np.empty((6890, 3, sequence_length), dtype='float32')
            dict_info['camDist'] = camera_distance
            dict_info['stride'] = stride
            dict_info['source'] = 'dfaust'
            reset_loc = False
            bpy.ops.wm.memory_statistics()
            # the shape of the vertices: <bpy_collection[6890], MeshVertices>
            random_zrot = 2 * np.pi * np.random.rand()
            arm_ob.animation_data_clear()
            cam_ob.animation_data_clear()
            num_frames = sequence_length
            for seq_frame in range(num_frames):
                index = seq_frame
                vertice = data[:, :, seq_frame]
                scene.frame_set(get_real_frame(seq_frame))
                arm_ob.pose.bones[obname + '_root'].rotation_quaternion = Quaternion(Euler((0, 0, random_zrot), 'XYZ'))
                arm_ob.pose.bones[obname + '_root'].keyframe_insert('rotation_quaternion', frame=get_real_frame(seq_frame))
                scene.update()
                if index == 0 or reset_loc:
                    reset_loc = False
                    cam_ob.keyframe_insert('location', frame=get_real_frame(seq_frame))
                    dict_info['camLoc'] = np.array(cam_ob.location)
                    dict_info['vertice'][:, :, index] = vertice
                    # set the camera locations of the corresponding frames
            for part, material in materials.items():
                material.node_tree.nodes['Vector Math'].inputs[1].default_value[:2] = (0, 0)
            sh_coeffs = .7 *  (2 * np.random.rand(9) - 1)
            sh_coeffs[
                0] = .5 + .9 * np.random.rand()  # Ambient light (first coeff) needs a minimum  is ambient. Rest is uniformly distributed, higher means brighter.
            sh_coeffs[1] = -.7 * np.random.rand()

            for ish, coeff in enumerate(sh_coeffs):
                for sc in scs:
                    sc.inputs[ish + 1].default_value = coeff
            for seq_frame in range(num_frames):
                index = seq_frame
                bpy.ops.object.mode_set(mode='EDIT')
                vertice = data[:, :, seq_frame]
                scene.frame_set(get_real_frame(seq_frame))
                mesh = self.change_existing_mesh(vertice, scene, mesh)
                scene.render.use_antialiasing = False
                bpy.data.objects.new('current_mesh', mesh)
                current_mesh = bpy.data.objects['current_mesh']
                scene.objects.link(current_mesh)
                log_message("Rendering frame %d" % seq_frame)

                # disable render output
                logfile = '/dev/null'
                open(logfile, 'a').close()
                old = os.dup(1)
                sys.stdout.flush()
                os.close(1)
                os.open(logfile, os.O_WRONLY)
                depth_out.file_slots[0].path = '%s_depth' % seq_frame
                segm_out.file_slots[0].path = '%s_segm' % seq_frame
                # Render
                bpy.ops.render.render(write_still=True)

                # disable output redirection
                os.close(1)
                os.dup(old)
                os.close(old)
                # cmd_tar = 'tar -czvf %s/%s.tar.gz -C %s %s' % (output_path, rgb_dirname, tmp_path, rgb_dirname)
                # log_message("Tarballing the images (%s)" % cmd_tar)
                # os.system(cmd_tar)
                for k, folder in res_paths.items():
                    if not k == 'vblur' and not k == 'fg':
                        file_name = str(seq_frame) + '_' + k + '%04d.exr' % (seq_frame)
                        path = join(folder, file_name)
                        render_img = bpy.data.images.load(path)
                        # render_img.pixels size is width * height * 4 (rgba)
                        arr = np.array(render_img.pixels[:]).reshape(resx, resy, 4)[::-1, :, :]
                        if k == 'depth':
                            mat = arr[:, :, 0]
                            dict_depth['depth_%d' % (index + 1)] = mat.astype(np.float32, copy=False)
                        elif k == 'segm':
                            mat = arr[:, :, 0]
                            dict_segm['segm_%d' % (index + 1)] = mat.astype(np.uint8, copy=False)
                scene.objects.unlink(current_mesh)
                bpy.data.objects.remove(current_mesh)
                # save annotation excluding png/exr data to _info.mat file
            scipy.io.savemat(matfile_info, dict_info, do_compression=True)
            scipy.io.savemat(matfile_depth, dict_depth, do_compression=True)
            scipy.io.savemat(matfile_segm, dict_segm, do_compression=True)

    def deselect(self):
        for o in bpy.data.objects.values():
            o.select_set(False)
        bpy.context.view_layer.objects.active = None

    def get_bname(self, i, obname='f_avg'):
        return obname+'_'+ self.kintree[i][1]

    def create_segmentation(self, ob, params):
        materials = {}
        vgroups = {}
        tmp_path = params['tmp_path']
        # /data2/liuguanze/dfaust/tmp/run0
        with open(tmp_path + '/pkl/segm_per_v_overlap.pkl', 'rb') as f:
            vsegm = load(f)
        bpy.ops.object.material_slot_remove()
        parts = sorted(vsegm.keys())
        for part in parts:
            vs = vsegm[part]
            vgroups[part] = ob.vertex_groups.new(part)
            vgroups[part].add(vs, 1.0, 'ADD')
            bpy.ops.object.vertex_group_set_active(group=part)
            materials[part] = bpy.data.materials['Material'].copy()
            materials[part].pass_index = part2num[part]
            bpy.ops.object.material_slot_add()
            ob.material_slots[-1].material = materials[part]
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.vertex_group_select()
            bpy.ops.object.material_slot_assign()
            bpy.ops.object.mode_set(mode='OBJECT')
        return materials

    def create_composite_nodes(self, tree, params, img=None, idx=0):
        res_paths = {k: join(params['tmp_path'], '%04d_%s' % (idx, k)) for k in params['output_types'] if
                     params['output_types'][k]}

        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)

        # create node for foreground image
        layers = tree.nodes.new('CompositorNodeRLayers')
        layers.location = -300, 400

        # create node for background image
        bg_im = tree.nodes.new('CompositorNodeImage')
        bg_im.location = -300, 30

        # create node for mixing foreground and background images
        mix = tree.nodes.new('CompositorNodeMixRGB')
        mix.location = 40, 30
        mix.use_alpha = True

        # create node for the final output
        composite_out = tree.nodes.new('CompositorNodeComposite')
        composite_out.location = 240, 30

        # create node for saving depth
        if params['output_types']['depth']:
            depth_out = tree.nodes.new('CompositorNodeOutputFile')
            depth_out.location = 40, 700
            depth_out.format.file_format = 'OPEN_EXR'
            depth_out.base_path = res_paths['depth']

        # create node for saving segmentation
        if params['output_types']['segm']:
            segm_out = tree.nodes.new('CompositorNodeOutputFile')
            segm_out.location = 40, 400
            segm_out.format.file_format = 'OPEN_EXR'
            segm_out.base_path = res_paths['segm']

        # merge fg and bg images
        tree.links.new(bg_im.outputs[0], mix.inputs[1])
        tree.links.new(layers.outputs['Image'], mix.inputs[2])

        tree.links.new(mix.outputs[0], composite_out.inputs[0])  # bg+fg image
        if params['output_types']['depth']:
            tree.links.new(layers.outputs['Z'], depth_out.inputs[0])  # save depth
        if params['output_types']['segm']:
            tree.links.new(layers.outputs['IndexMA'], segm_out.inputs[0])  # save segmentation

        return res_paths, depth_out, segm_out

    def rodrigues2bshapes(self, pose, mat_pose=False):
        if mat_pose:
            mat_rots = np.zeros((self.n_bones, 3, 3))
            mat_rots[1:] = pose[1:]
        else:
            rod_rots = np.asarray(pose).reshape(self.n_bones, 3)
            mat_rots = [self.rodrigues(rod_rot) for rod_rot in rod_rots]
        bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel() for mat_rot in mat_rots[1:]])
        return mat_rots, bshapes

    def create_sh_material(self, tree, sh_path, img=None):
        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)

        uv = tree.nodes.new('ShaderNodeTexCoord')
        uv.location = -800, 400

        uv_xform = tree.nodes.new('ShaderNodeVectorMath')
        uv_xform.location = -600, 400
        uv_xform.inputs[1].default_value = (0, 0, 1)
        uv_xform.operation = 'AVERAGE'

        uv_im = tree.nodes.new('ShaderNodeTexImage')
        uv_im.location = -400, 400
        if img is not None:
            uv_im.image = img

        rgb = tree.nodes.new('ShaderNodeRGB')
        rgb.location = -400, 200

        script = tree.nodes.new('ShaderNodeScript')
        script.location = -230, 400
        script.mode = 'EXTERNAL'
        script.filepath = sh_path  # 'spher_harm/sh.osl' #using the same file from multiple jobs causes white texture
        script.update()

      # the emission node makes it independent of the scene lighting
        emission = tree.nodes.new('ShaderNodeEmission')
        emission.location = -60, 400

        mat_out = tree.nodes.new('ShaderNodeOutputMaterial')
        mat_out.location = 110, 400

        tree.links.new(uv.outputs[2], uv_im.inputs[0])
        tree.links.new(uv_im.outputs[0], script.inputs[0])
        tree.links.new(script.outputs[0], emission.inputs[0])
        tree.links.new(emission.outputs[0], mat_out.inputs[0])

    def init_scene(self, scene, params):
        # load fbx model
        bpy.ops.import_scene.fbx(
          filepath=join(params['smpl_data_folder'], 'basicModel_m_lbs_10_207_0_v1.0.2.fbx'),
          axis_forward='Y', axis_up='Z', global_scale=100)

        obname = 'm_avg'
        ob = bpy.data.objects[obname]
        ob.data.use_auto_smooth = False  # autosmooth creates artifacts
        # print('the type of the object:', type(ob))
        # assign the existing spherical harmonics material
        ob.active_material = bpy.data.materials['Material']

        # delete the default cube (which held the material)
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Cube'].select = True
        bpy.ops.object.delete(use_global=False)

        # set camera properties and initial position
        bpy.ops.object.select_all(action='DESELECT')
        cam_ob = bpy.data.objects['Camera']
        scn = bpy.context.scene
        scn.objects.active = cam_ob

        cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),
                                        (0., -1, 0., -1.0),
                                        (-1., 0., 0., 0.),
                                        (0.0, 0.0, 0.0, 1.0)))
        cam_ob.data.angle = math.radians(40)
        cam_ob.data.lens = 60
        cam_ob.data.clip_start = 0.1
        cam_ob.data.sensor_width = 32
        print('the locations of the cam_ob:', cam_ob.location)
        # setup an empty object in the center which will be the parent of the Camera
        # this allows to easily rotate an object around the origin
        scn.cycles.film_transparent = True
        scn.render.layers["RenderLayer"].use_pass_vector = True
        scn.render.layers["RenderLayer"].use_pass_normal = True
        scene.render.layers['RenderLayer'].use_pass_emit = True
        scene.render.layers['RenderLayer'].use_pass_emit = True
        scene.render.layers['RenderLayer'].use_pass_material_index = True

        # set render size
        scn.render.resolution_x = params['resy']
        scn.render.resolution_y = params['resx']
        scn.render.resolution_percentage = 100
        scn.render.image_settings.file_format = 'PNG'

        # clear existing animation data
        ob.data.shape_keys.animation_data_clear()
        arm_ob = bpy.data.objects['Armature']
        arm_ob.animation_data_clear()

        return ob, obname, arm_ob, cam_ob

    def reset_joint_positions(self, orig_trans, vertices, ob, arm_ob, obname, scene, cam_ob, reg_ivs, joint_reg):
        # since the regression is sparse, only the relevant vertex
        #     elements (joint_reg) and their indices (reg_ivs) are loaded
        reg_vs = np.empty((len(reg_ivs), 3))  # empty array to hold vertices to regress from
        # zero the pose and trans to obtain joint positions in zero pose
        # obtain a mesh after applying modifiers
        bpy.ops.wm.memory_statistics()
        # me holds the vertices after applying the shape blendshapes
        mesh = ob.to_mesh(scene, True, 'PREVIEW')
        self.apply_current_vertice(mesh, orig_trans, vertices, arm_ob, obname)
        # fill the regressor vertices matrix
        for iiv, iv in enumerate(reg_ivs):
            reg_vs[iiv] = mesh.vertices[iv].co
        # regress joint positions in rest pose
        joint_xyz = joint_reg.dot(reg_vs)
        # adapt joint positions in rest pose
        arm_ob.hide = False
        arm_ob.hide = True
        for ibone in range(24):
            bb = arm_ob.data.edit_bones[obname + '_' + part_match['bone_%02d' % ibone]]
            bboffset = bb.tail - bb.head
            bb.head = joint_xyz[ibone]
            bb.tail = bb.head + bboffset
        bpy.ops.object.mode_set(mode='OBJECT')
        return mesh

    def load_body_data(self, dfaust_path, ob, obname, dfaust_use_male=True, dfaust_use_female=True, idx=0):
        # load MoSHed data from CMU Mocap (only the given idx is loaded)
        dfaust_female_path = dfaust_path + 'registrations_f.hdf5'
        female_file = h5py.File(dfaust_female_path, 'r')
        # create a dictionary with key the sequence name and values the pose and trans
        cmu_keys = []
        cmu_params = {}
        if dfaust_use_female:
            for name in list(female_file.keys()):
                if not name.endswith('faces'):
                    sequence_data = female_file[name]
                    name = 'female' + name
                    cmu_params[name] = sequence_data
                    cmu_keys.append(name)

        return cmu_params, cmu_keys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synth dataset images.')
    parser.add_argument('--idx', type=int,
                        help='idx of the requested sequence')
    parser.add_argument('--ishape', type=int,
                        help='requested cut, according to the stride')
    parser.add_argument('--stride', type=int,
                        help='stride amount, default 50')
    parser.add_argument('--output_dir', type=str, default='/data2/liuguanze/dfaust/depth_image')
    parser.add_argument('--fbegin', type=int, default=0)
    parser.add_argument('--fend', type=int, default=5)
    parser.add_argument('--number_files', type=int, default=100)
    opt = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    dfaust_renderer = Misha()
    dfaust_renderer.render(opt)