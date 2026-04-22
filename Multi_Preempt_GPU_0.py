import pynvml
import os
import time

# trainpythonfile= "/data1/zhaoshijie_3/project/code/egfr/Add_Transformer/main_cifar10_swtTransformer.py"
trainpythonfile='/data2/lwy/projevt/Lung/VGG_egfr9_1_Dialated_AC_in_Dense_K_fold/main_Transformer_noDialated_112_112_112_layer2.py'

gpu_memory = 9 # 预期GPU剩余显存，如果达到这个数字就运行train脚本
interval = 2   # sleep interval
stand_dim = 256
pynvml.nvmlInit()

gpu_list = [
    'os.environ["CUDA_VISIBLE_DEVICES"] = "0"',
    'os.environ["CUDA_VISIBLE_DEVICES"] = "1"',
    'os.environ["CUDA_VISIBLE_DEVICES"] = "2"',
    'os.environ["CUDA_VISIBLE_DEVICES"] = "3"',
    'os.environ["CUDA_VISIBLE_DEVICES"] = "4"',
    'os.environ["CUDA_VISIBLE_DEVICES"] = "5"',
    'os.environ["CUDA_VISIBLE_DEVICES"] = "6"',
    'os.environ["CUDA_VISIBLE_DEVICES"] = "7"',
    'os.environ["CUDA_VISIBLE_DEVICES"] = "8"',
    'os.environ["CUDA_VISIBLE_DEVICES"] = "9"'
]

# 读写python文件在指定位置插入显卡编号
def writepythonfile(pythonfile,cline,string):
    lines = []
    with open(pythonfile,"r") as y:
        for line in y:
            lines.append(line)
        y.close()
    lines.insert(cline, string+"\n")
    s = ''.join(lines)
    with open(pythonfile,"w") as z:
        z.write(s)
        z.close()
    del lines[:]

if __name__ == '__main__':
    Flag = False    # break the while 1
    while 1:
        if Flag==True:
            break
        else:
            time.sleep(interval)
            for i in range(0,len(gpu_list)):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                print('第'+str(i)+'块GPU剩余显存:'+str(meminfo.free/(1024**3))+' GB') #第i块显卡剩余显存大小
                if meminfo.free/(1024**3) >= gpu_memory:
                    # writepythonfile(trainpythonfile,0,'import os')
                    # writepythonfile(trainpythonfile,1,gpu_list[i])
                    # return_value = os.system('python '+trainpythonfile+' -cuda '+str(i)+' --stand_dim '+str(stand_dim))    # error=256,no_error=0
                    return_value = os.system('python ' + trainpythonfile + ' -cuda ' + str(i))  # error=256,no_error=0
                    if return_value==0: # run no error
                        Flag = True
                    break
                else:
                    print("不符合剩余"+str(gpu_memory)+"GB显存需求")
    print(Flag)
    print('\nEnd the monitoring!')