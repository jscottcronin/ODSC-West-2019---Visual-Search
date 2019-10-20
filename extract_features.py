"""Feature Extraction from Image Filename."""

from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

model = models.resnet18(pretrained=True)
model.eval()

layer = model._modules.get('avgpool')

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


def get_vector(image_name):
    """Extract vector from filename for local image."""
    img = Image.open(image_name)
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

    # Create hook to extract vector
    embedding = torch.zeros(512)

    def copy_data(m, i, o):
        embedding.copy_(o.data.resize(512))
    h = layer.register_forward_hook(copy_data)

    model(t_img)
    h.remove()

    return embedding.numpy()
