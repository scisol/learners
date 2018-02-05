"""

"""
from shapely.geometry import Polygon


# code
def ret_poly(x):
    """
    Given a feature vector of shape (D,), convert it to array of (floor(D/2),2)
    and construct a Polygon object from the data.
    """
    x = x.reshape((int(float(x.shape[0])/2), 2))
    return Polygon(zip(x[:, 0].tolist(), x[:, 1].tolist()))


def polygon_overlap(p1, p2):
    """
    Given two polygons, compute the overlap of the polygons with each other.
    """
    p1_self = p1.intersection(p1).area
    p2_self = p2.intersection(p2).area
    return (p1_self + p2_self - 2*p1.intersection(p2).area)#/(p1_self + p2_self)


def metric_overlap(x, y):
    """
    Distance metric for overlap between two vectors representing 2-dimensional polygons.
    Consumed by scikit-learn's nearest neighbors class.

    :param x:
    :param y:
    :return:
    """
    return polygon_overlap(ret_poly(x), ret_poly(y))
