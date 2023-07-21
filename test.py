import torch

import torch.nn as nn
import pdb
from ultralytics import YOLO
import torch.nn.functional as F



class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """

    def __init__(self, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            print(s.shape)
            print(t.shape)
            assert s.shape == t.shape
            
            N, C, H, W = s.shape
            
            # normalize in channel diemension
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)  # [N*C, H*W]
            
            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))
        loss = sum(losses)

        return loss

class MGDLoss(nn.Module):
    def __init__(self, channels_s, channels_t, alpha_mgd=0.00002, lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = [
            nn.Sequential(
                nn.Conv2d(channel_s, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1)).to(device) for channel_s,channel in zip(channels_s,channels_t)
        ]
        # print(self.generation)
    def forward(self, y_s, y_t,layer=None):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # assert s.shape == t.shape
            if layer == "outlayer":
                idx = -1
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)
        # print(preds_S.shape)
        masked_fea = torch.mul(preds_S, mat)
        # print(masked_fea.shape)
        print(self.generation[idx])
        new_fea = self.generation[idx](masked_fea)
        print(new_fea.shape)
        print(preds_T.shape)

        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss


class Distillation_loss:
    def __init__(self, modeln,modelL,distiller="CWDLoss"):  # model must be de-paralleled

        # self.D_loss_fn   = torch.nn.MSELoss()
        # self.D_loss_fn   =  torch.nn.BCELoss()

        self.distiller = distiller
        if distiller == "MGDLoss":
            channels_s = [32,64,128,256,128,64,128,256]
            channels_t = [128,256,512,512,512,256,512,512]
            
   
            # channels_s.append(modelL.yaml['ch']+64)
            # channels_t.append(modelL.yaml['ch']+64)
            self.D_loss_fn   = MGDLoss(channels_s=channels_s,channels_t=channels_t)
        elif distiller == "CWDLoss":
            self.D_loss_fn   = CWDLoss(1)

        self.teacher_module_pairs = []
        self.student_module_pairs = []
        self.remove_handle = []
        layers = ["2","4","6","8","12","15","18","21"]

        for mname, ml in modelL.named_modules():
            if mname is not None:
                name = mname.split(".")
                if name[0] == "module":
                    name.pop(0)
                if len(name) == 3:
                    if name[1] in layers:
                        if "cv2" in mname:
                            self.teacher_module_pairs.append(ml)

        for mname, ml in modeln.named_modules():

            if mname is not None:
                name = mname.split(".")
                if name[0] == "module":
                    name.pop(0)
                if len(name) == 3:
                    # print(mname)
                    if name[1] in layers:
                        if "cv2" in mname:
                            self.student_module_pairs.append(ml)


    def register_hook(self):
        self.teacher_outputs = []
        self.student_outputs = []

        def make_layer_forward_hook(l):
            def forward_hook(m, input, output):
                l.append(output)
            return forward_hook
        
        for ml, ori in zip(self.teacher_module_pairs,self.student_module_pairs):
            # 为每层加入钩子，在进行Forward的时候会自动将每层的特征传送给model_outputs和origin_outputs
            self.remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(self.teacher_outputs)))
            self.remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(self.student_outputs))) 

    def get_loss(self):
        quant_loss = 0
        # for index, (mo, fo) in enumerate(zip(self.teacher_outputs, self.student_outputs)):
        #     print(mo.shape,fo.shape)
            # quant_loss += self.D_loss_fn(mo, fo)
        quant_loss += self.D_loss_fn(y_t=self.teacher_outputs, y_s=self.student_outputs)

        self.teacher_outputs.clear()
        self.student_outputs.clear()
        return quant_loss
        
    
    def remove_handle_(self):
        for rm in self.remove_handle:
            rm.remove()


# modelL = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/coco_v8l/weights/best.pt')
# modelL = YOLO('yolov8l.pt')

T_model = YOLO('/home/huangzm/code/mycode/pytorch/yolov8/Knowledge_Distillation/ultralytics/models/v8/yolov8l.yaml')
data = "/home/huangzm/code/mycode/pytorch/yolov5/data/yitiji/person_car.yaml"

# # modelL.train(data=data, epochs=1, imgsz=640, Distillation = None)
modeln = YOLO('yolov8s.pt')

# for m in modeln.model.modules():
#     print(m)
    # if isinstance(m, C2f_4):
    #     print(m)
# # print(modeln.model.model[-1])
# # modeln.model.model[-1].export = True
# # success = modeln.export(format="onnx")
# print(modelL.model)
T_model.model.model[-1].set_Distillation = True
D_loss = Distillation_loss(modeln.model,T_model.model,distiller="CWDLoss")
example_inputs = torch.randn(4, 3, 640, 640)
D_loss.register_hook()
modeln.model(example_inputs)

y = T_model.model(example_inputs)

print(D_loss.get_loss())
if D_loss.distiller == "MGDLoss":
    print(D_loss.D_loss_fn(y,y,layer="outlayer"))
D_loss.remove_handle_()

# modeln = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/train30/weights/best.pt')
# source="/home/huangzm/code/mycode/pytorch/yolov5/img/20221214120221.jpg"
# modeln.predict(source=source)
# modeln.train(data=data, epochs=1, imgsz=640,batch=1, Distillation = None)
# success = modeln.export(format="onnx",imgsz=(384,640),device="cpu")

