from typing import List, Tuple
import torch
from datetime import datetime, timedelta
import math


def euclidean_distance(loc1, loc2):
    """
    Computes the Euclidean distance between two locations.
    :param loc1: Tuple (2) (longitude, latitude)
    :param loc2: Tuple (2) (longitude, latitude)
    :return distance: float
    """

    sqr_delta = (loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2

    return math.sqrt(sqr_delta)


def haversine_distance(loc1, loc2, radius: float = 6371):
    """
    Computes the Euclidean distance between two locations.
    :param loc1: Tuple (2) (longitude, latitude)
    :param loc2: Tuple (2) (longitude, latitude)
    :param radius: float. Default is radius of the Earth (6371 km)

    :return distance: float. Km unit
    """

    delta_lon = (loc1[0] - loc2[0]) * math.pi / 180
    delta_lat = (loc1[1] - loc2[1]) * math.pi / 180

    lat1_rad = loc1[1] * math.pi / 180
    lat2_rad = loc2[1] * math.pi / 180

    a = math.sin(delta_lat / 2) ** 2 + \
                math.sin(delta_lon / 2) ** 2 * math.cos(lat1_rad) * math.cos(lat2_rad)

    dist = 2 * math.asin(math.sqrt(a)) * radius

    return dist


def get_solar_term(date: datetime):
    def in_range(date: datetime, d1: Tuple[int, int], d2: Tuple[int, int]):
        d1, m1 = d1
        d2, m2 = d2

        cond1 = date.day >= d1 and date.month == m1
        cond2 = date.month > m1
        cond3 = date.month < m2
        cond4 = date.day < d2 and date.month == m2

        return (cond1 or cond2) and (cond3 or cond4)

    LAP_XUAN, VU_THUY, KINH_TRAP, XUAN_PHAN, THANH_MINH, COC_VU,\
        LAP_HA, TIEU_MAN, MANG_CHUNG, HA_CHI, TIEU_THU, DAI_THU,\
        LAP_THU, XU_THU, BACH_LO, THU_PHAN, HAN_LO, SUONG_GIANG,\
        LAP_DONG, TIEU_TUYET, DAI_TUYET, DONG_CHI, TIEU_HAN, DAI_HAN,\
        = range(24)

    if in_range(date, (4, 2), (18, 2)):
        return LAP_XUAN
    elif in_range(date, (18, 2), (5, 3)):
        return VU_THUY
    elif in_range(date, (5, 3), (20, 3)):
        return KINH_TRAP
    elif in_range(date, (20, 3), (4, 4)):
        return XUAN_PHAN
    elif in_range(date, (4, 4), (20, 4)):
        return THANH_MINH
    elif in_range(date, (20, 4), (5, 5)):
        return COC_VU
    elif in_range(date, (5, 5), (21, 5)):
        return LAP_HA
    elif in_range(date, (21, 5), (5, 6)):
        return TIEU_MAN
    elif in_range(date, (5, 6), (21, 6)):
        return MANG_CHUNG
    elif in_range(date, (21, 6), (7, 7)):
        return HA_CHI
    elif in_range(date, (7, 7), (22, 7)):
        return TIEU_THU
    elif in_range(date, (22, 7), (7, 8)):
        return DAI_THU
    elif in_range(date, (7, 8), (23, 8)):
        return LAP_THU
    elif in_range(date, (23, 8), (7, 9)):
        return XU_THU
    elif in_range(date, (7, 9), (23, 9)):
        return BACH_LO
    elif in_range(date, (23, 9), (8, 10)):
        return THU_PHAN
    elif in_range(date, (8, 10), (23, 10)):
        return HAN_LO
    elif in_range(date, (23, 10), (7, 11)):
        return SUONG_GIANG
    elif in_range(date, (7, 11), (22, 11)):
        return LAP_DONG
    elif in_range(date, (22, 11), (7, 12)):
        return TIEU_TUYET
    elif in_range(date, (7, 12), (21, 12)):
        return DAI_TUYET
    elif in_range(date, (21, 12), (32, 12)) or in_range(date, (1, 1), (5, 1)):
        return DONG_CHI
    elif in_range(date, (5, 1), (20, 1)):
        return TIEU_HAN
    elif in_range(date, (20, 1), (4, 2)):
        return DAI_HAN
    else:
        raise ValueError


def extract_wind(u: List[float], v: List[float]):
    """
    Extract wind from u, v vector

    Returns:
        direction: Direction of wind
        magnitude:  Magnitude of wind
    """
    direction = []
    magnitude = []
    for i in range(len(u)):
        x = u[i]
        y = v[i]
        direction.append(math.atan2(y, x))

        magnitude.append(math.sqrt(y ** 2 + x ** 2))

    return direction, magnitude
