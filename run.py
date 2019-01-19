import os
import doodad as dd
import doodad.mode as mode
import doodad.mount as mount
import doodad.ssh as ssh

mode_local = mode.LocalDocker(
    image='kylehsu/umrl:latest'
)

os.environ['CLOUDSDK_CORE_PROJECT'] = 'umrl-228204'
mode_gcp = mode.GCPDocker(
    zone='us-west1-b',
    gcp_bucket_name='umrl',
    instance_type='n1-standard-8',
    image_name='gpu-docker',
    image_project='umrl-228204',
    disk_size='100',
    terminate=True,
    preemptible=False,
    gcp_log_prefix='experiment',
    gpu=True,
    gpu_kwargs=dict(num_gpu=1, gpu_model='nvidia-tesla-k80'),
    image='kylehsu/umrl:latest'
)

mode_ssh = dd.mode.SSHDocker(
    image='kylehsu/umrl:latest',
    credentials=ssh.SSHCredentials(hostname='alan.ist.berkeley.edu', username='kylehsu', identity_file='~/.ssh/id_rsa'),
)

mode = mode_gcp

mounts = [
    mount.MountLocal(local_dir=os.path.dirname(os.path.realpath(__file__)),
                     filter_dir=('output',))
]

output_dir = '/home/docker/store/umrl/output'

if mode == mode_gcp:
    output_mount = mount.MountGCP(gcp_path='output', gcp_bucket_name='umrl', mount_point=output_dir, output=True,
                                  include_types=('*.txt', '*.csv', '*.json', '*.gz', '*.tar', '*.log', '*.pkl',
                                                 '*.png', '*.html', '*.mp4'))
else:
    output_mount = mount.MountLocal(local_dir='/home/kylehsu/experiments/umrl/output', mount_point=output_dir,
                                    output=True)

mounts.append(output_mount)

dd.launch_python(
    target=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'main_contextual.py'),
    mode=mode,
    mount_points=mounts,
    args=dict(log_dir_root=output_dir),
    python_cmd='source activate umrl && python -m ipdb -c continue',
    fake_display=False
)

# dd.launch_shell(
#     command='ls',
#     mode=mode,
#     dry=False,
#     mount_points=[]
#     # mount_points=mounts
# )
