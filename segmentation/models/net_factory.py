from .unet import UNetAux, UNetOrg


def net_factory(net_type="unetaux", in_chns=3, class_num=2, drop=False):
    if net_type == "unetaux":
        net = UNetAux(in_chns=in_chns, class_num=class_num, dropout=drop)#.cuda()
    else:
        net = UNetOrg(in_chns=in_chns, class_num=class_num, dropout=drop)#.cuda()
    return net