import os


gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

file = '1_main.py'
filepath = os.path.join(os.getcwd(), file)

lr_init_list = [1e-3, 1e-4, 1e-5]

epochs = 10

batch_size = 64


def run_jobs():
    for lr_init in lr_init_list:

        command = 'python {0} -batch_size {1} -lr_init {2} -epochs {3} '.format(filepath,
                                                                   batch_size,
                                                                   lr_init,
                                                                   epochs,)

        print(command)
        if os.system(command):
            print('\nFailed: ', command, '\n')



def main():
    run_jobs()



if __name__ == '__main__':
    main()

