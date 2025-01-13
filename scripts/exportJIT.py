import torch
from SuperPointPretrainedNetwork.demo_superpoint import SuperPointNet

net = SuperPointNet()
net.load_state_dict(torch.load('SuperPointPretrainedNetwork/superpoint_v1.pth',
                        map_location=lambda storage, loc: storage))

# torch.save(net.state_dict(), "my.pt")
# 使用 TorchScript 跟踪模型
scripted_model = torch.jit.script(net)

# 保存为 TorchScript 格式
scripted_model.save("../weights/superpoint.pt")