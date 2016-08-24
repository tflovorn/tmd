def parse_inner_window(win_path):
    '''Return inner_win = (inner_window_min, inner_window_max) given in
    the Wannier90 input file and win_path.
    '''
    lines = None
    with open(win_path, 'r') as fp:
        lines = fp.readlines()

    inner_win = [None, None]
    for line in lines:
        lsp = line.strip().split('=')
        if lsp[0].strip() == 'dis_froz_min':
            inner_win[0] = float(lsp[-1])
        elif lsp[0].strip() == 'dis_froz_max':
            inner_win[1] = float(lsp[-1])

    if None in inner_win:
        raise ValueError("Failed to find dis_froz_min and dis_froz_max in Wannier90 input file")

    return inner_win
