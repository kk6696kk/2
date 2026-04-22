import os

cmd0 = 'python "/disk1/liuzy/lung_cancer_project/VGG_egfr9_1_Dialated_AC_in_Dense_K_fold/main.py"'
cmd1 = 'python "/disk1/liuzy/lung_cancer_project/VGG_egfr9_1_Dialated_AC_in_Dense_K_fold/main_(112_112_112).py"'
cmd2 = 'python "/disk1/liuzy/lung_cancer_project/VGG_egfr9_1_Dialated_AC_in_Dense_K_fold/main_(ori_90_112_112).py"'
cmd3 = 'python "/disk1/liuzy/lung_cancer_project/VGG_egfr9_1_Dialated_AC_in_Dense_K_fold/main_twodataset.py"'
cmd4 = 'python "/disk1/liuzy/lung_cancer_project/VGG_egfr9_1_Dialated_AC_in_Dense_K_fold/main_twodataset(112).py"'


def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('Default |')
    memort_lis = []
    for i in range(1):
        gpu_status_one = gpu_status[i].split('|')
        gpu_memory = int(gpu_status_one[2].split('/')[0].split('M')[0].strip())
        memort_lis.append(gpu_memory)
    return memort_lis


def narrow_setup():
    memort_lis = gpu_info()
    min_mem = min(memort_lis)
    while min_mem > 1000:
        print(memort_lis)
        print('No aviliable GPU, try again!')
        memort_lis = gpu_info()
        min_mem = min(memort_lis)

    num_gpu = memort_lis.index(min_mem)
    print('begin in gpu ', num_gpu)

    print('\n' + cmd0)
    os.system(cmd0)
    print('\n' + cmd1)
    os.system(cmd1)
    print('\n' + cmd2)
    os.system(cmd2)
    print('\n' + cmd3)
    os.system(cmd3)
    print('\n' + cmd4)
    os.system(cmd4)


if __name__ == '__main__':
    narrow_setup()
