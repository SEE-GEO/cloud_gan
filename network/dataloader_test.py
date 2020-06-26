import torch
n=50000
scenes = [range(0,n),range(0,n)]
batch_size =64
workers = 2
dataloader = torch.utils.data.DataLoader(scenes, batch_size=batch_size, shuffle=True,
                                             num_workers=workers)

for i, data in enumerate(dataloader,0):
    if data[0]!=data[1]:
        print(str(data[0]),' and ', str(data[1]))