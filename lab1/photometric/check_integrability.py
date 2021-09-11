import numpy as np

def check_integrability(normals):
    #  CHECK_INTEGRABILITY check the surface gradient is acceptable
    #   normals: normal image
    #   p : df / dx
    #   q : df / dy
    #   SE : Squared Errors of the 2 second derivatives

    # initalization
    p = np.zeros(normals.shape[:2])
    q = np.zeros(normals.shape[:2])
    SE = np.zeros(normals.shape[:2])

    """
    ================
    Your code here
    ================
    Compute p and q, where
    p measures value of df / dx
    q measures value of df / dy
    """

    n1 = normals[:, :, 0]
    n2 = normals[:, :, 1]
    n3 = normals[:, :, 2]

    p = np.divide(n1, n3, out = np.zeros_like(p), where = (n3 != 0))
    q = np.divide(n2, n3, out = np.zeros_like(q), where = (n3 != 0))

    # change nan to 0
    p[p!=p] = 0
    q[q!=q] = 0

    """
    ================
    Your code here
    ================
    approximate second derivate by neighbor difference
    and compute the Squared Errors SE of the 2 second derivatives SE
    """

    dp_dy = np.diff(p, axis = 1, append = 0)
    dq_dx = np.diff(q, axis = 0, append = 0)
    SE = np.power(dp_dy - dq_dx, 2)

    return p, q, SE


if __name__ == '__main__':
    normals = np.zeros([10,10,3])
    check_integrability(normals)
