from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

# Untrained model
my_model = vgg11_bn()

# Pretrained model
my_model = vgg11_bn(pretrained=True)
my_model.eval() # for evaluation

print("a")