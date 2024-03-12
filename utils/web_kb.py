from torch_geometric.datasets import WebKB

def texas_data():
    webkb = WebKB(root='data/WebKB', name="Texas")
    print(webkb.num_features, webkb.num_classes)
    return webkb[0]

# print(texas_data())

def cornell_data():
    webkb = WebKB(root='data/WebKB', name="Cornell")
    print(webkb.num_features, webkb.num_classes)
    return webkb[0]