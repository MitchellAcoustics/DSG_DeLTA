import torch, os

if os.path.exists(r'D:\Yuanbo\Code\UCL\meta_data'):
    root = r'D:\Yuanbo\Code\UCL\meta_data'
elif os.path.exists(r'E:\Yuanbo\UCL\DeLTA\meta_data'):
    root = r'E:\Yuanbo\UCL\DeLTA\meta_data'
elif os.path.exists(r'C:\Users\mitch\Documents\Github\DSG_DeLTA\meta_data_v2'):
    root = r'C:\Users\mitch\Documents\Github\DSG_DeLTA\meta_data_v2'

####################################################################################################

cuda = 1

training = 1
testing = 1

if cuda:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')

mel_bins = 64
batch_size = 64
epoch = 10
lr_init = 1e-3

event_labels = ['Aircraft', 'Bells', 'Bird tweet', 'Bus', 'Car', 'Children', 'Construction',
                'Dog bark', 'Footsteps', 'General traffic', 'Horn', 'Laughter', 'Motorcycle', 'Music',
                'Non-identifiable', 'Other', 'Rail', 'Rustling leaves', 'Screeching brakes', 'Shouting',
                'Siren', 'Speech', 'Ventilation', 'Water']

endswith = '.pth'

cuda_seed = None
