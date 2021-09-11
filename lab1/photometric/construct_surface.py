import numpy as np

def __constr_col_major(p, q):
    """
        ================
        Your code here
        ================
        % top left corner of height_map is zero
        % for each pixel in the left column of height_map
        %   height_value = previous_height_value + corresponding_q_value

        % for each row
        %   for each element of the row except for leftmost
        %       height_value = previous_height_value + corresponding_p_value

    """
    h, w = p.shape
    height_map = np.zeros([h, w])

    height_map[:, 0] = np.cumsum(q[:, 0])

    for row in range(h):
        height_map[row, 1:] = np.cumsum(p[row, 1:]) + np.repeat(q[row, 0], w - 1)

    return height_map

def __constr_row_major(p, q):
    """
        ================
        Your code here
        ================
        % top left corner of height_map is zero
        % for each pixel in the top row of height_map
        %   height_value = previous_height_value + corresponding_p_value

        % for each column
        %   for each element of the column except for topmost
        %       height_value = previous_height_value + corresponding_q_value

    """
    h, w = p.shape
    height_map = np.zeros([h, w])

    height_map[0, :] = np.cumsum(p[0, :])

    for col in range(w):
        height_map[1:, col] = np.cumsum(q[1:, col]) + np.repeat(q[0, col], h - 1)

    return height_map

def construct_surface(p, q, path_type='column'):
    '''
    CONSTRUCT_SURFACE construct the surface function represented as height_map
       p : measures value of df / dx
       q : measures value of df / dy
       path_type: type of path to construct height_map, either 'column',
       'row', or 'average'
       height_map: the reconstructed surface
    '''

    h, w = p.shape
    height_map = np.zeros([h, w])

    if path_type=='column':
        height_map = __constr_col_major(p, q)

    elif path_type=='row':
        height_map = __constr_row_major(p, q)


    elif path_type=='average':
        hm_c = __constr_col_major(p, q)
        hm_r = __constr_row_major(p, q)
        height_map = (hm_c + hm_r) / 2

    return height_map

