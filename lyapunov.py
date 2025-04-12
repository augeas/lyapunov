import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Lyapunov fractals

        The [Lyapunov Fractal](https://en.wikipedia.org/wiki/Lyapunov_fractal) is calculated
        from repeated iterations of the [Logistic Map](https://en.wikipedia.org/wiki/Logistic_map):

        $$ \Large{x_{n+1} = r_{n}x_{n}(x_{n}-1)} $$

        where at each iteration $r_{n}$ takes values from some repeated sequence, for 
        example $AABAB$. For each point in an image $A$ and $B$ take the values of
        the $x$ and $y$ coordinates. For a large number of iterations $N$, the Lyapunov
        exponent $\lambda$ is found for each point, and coloured accordingly:

        $$ \Large{\lambda = \dfrac{1}{N}\sum_{n=1}^{N}|r_{n}(1-2x_{n})|} $$
        """
    )
    return


@app.cell
def _():
    from datetime import datetime
    import io
    import itertools
    from functools import partial
    import math
    try:
        from multiprocessing import Pool, shared_memory
    except:
        pass
    import os
    import subprocess as sp
    import sys
    from typing import Iterable

    from matplotlib import colormaps
    import numpy as np
    from numpy import typing as npt
    from PIL import Image
    return (
        Image,
        Iterable,
        Pool,
        colormaps,
        datetime,
        io,
        itertools,
        math,
        np,
        npt,
        os,
        partial,
        shared_memory,
        sp,
        sys,
    )


@app.cell
def _(np, npt):
    def seq_vector(seq: str, size: int=128) -> npt.ArrayLike:
        """Take a string of letters and return a repeating array of ints."""
        assert str.isalpha(seq)
        vec = np.fromiter(map(ord, seq.upper()), dtype=np.int32) - 65
        reps = size // len(seq)
        if size % len(seq) == 0:
            return np.tile(vec, reps)
        else:
            return np.tile(vec, reps + 1)[:size]
    return (seq_vector,)


@app.cell
def _(np, npt):
    def lyapunov(seq: npt.ArrayLike, *points: npt.NDArray) -> npt.NDArray:
        """Compute a Lyapunov fractal.

        seq: Coefficient sequence as a Numpy array of ints in [0..N].
        points: List of N arrays, one for each coefficient, giving
        the coefficent value at each point in the image.
        """
        coeffs = np.stack(points)[seq]
        n_its = len(seq)
        iterates = np.zeros(coeffs.shape)
        iterates[0] = 0.5
        for i in range(1, n_its):
            prev = iterates[i-1]
            iterates[i] = coeffs[i] * prev * (1.0 - prev)
        return np.log(
            np.abs(coeffs[1:] * (1.0 - 2 * iterates[1:]))
        ).sum(axis=0) / (n_its-1)
    return (lyapunov,)


@app.cell
def _(Image, colormaps, np, npt):
    def sigmoid(x: npt.NDArray) -> npt.NDArray:
        """Hyperboloic tangent normalized from 0 to 1."""
        return 0.5 * (1 + np.tanh(x))

    def render_image(img: npt.NDArray, palette: str='Spectral') -> Image.Image:
        """Colour a Numpy array according to a Matplotlib palette."""
        colours = colormaps[palette]
        return Image.fromarray(
            (255 * colours(sigmoid(img))).astype(np.uint8)
        )
    return render_image, sigmoid


@app.cell
def _(colormaps, mo):
    seq_box = mo.ui.text(value='AABAB', label='coefficient sequence')
    its_box = mo.ui.number(start=20, stop=200, step=20, value=100, 
                           label="number of iterations")
    x_img_slider = mo.ui.range_slider(start=2.0, stop=4.0, step=0.1, value=[2.0, 4.0],
                                   label='x range')
    y_img_slider = mo.ui.range_slider(start=2.0, stop=4.0, step=0.1, value=[2.0, 4.0],
                                   label='y range')
    cmap_names = ['seismic', 'vanimo', 'managua', 'berlin', 'Spectral',
        'twilight', 'ocean', 'cubehelix', 'turbo', 'plasma', 'magma']

    palettes = [name for name in cmap_names if name in colormaps]

    colour_box = mo.ui.dropdown(palettes, value='managua', label='palette')
    return (
        cmap_names,
        colour_box,
        its_box,
        palettes,
        seq_box,
        x_img_slider,
        y_img_slider,
    )


@app.cell
def _(
    colour_box,
    its_box,
    lyapunov,
    mo,
    np,
    render_image,
    seq_box,
    seq_vector,
    x_img_slider,
    y_img_slider,
):
    __IMG_SIZE__ = 400

    img_x_min, img_x_max = x_img_slider.value
    img_y_min, img_y_max = y_img_slider.value

    img_x_points, img_y_points = np.meshgrid(
        np.linspace(img_x_min, img_x_max, __IMG_SIZE__),
        np.linspace(img_y_max, img_y_min, __IMG_SIZE__),
        indexing='xy'
    )

    img_seq = seq_vector(
        ''.join(filter(lambda char: char in 'AB', seq_box.value.upper())),
        its_box.value
    )

    def ui_image():
        if mo.running_in_notebook():
             return render_image(lyapunov(
                 img_seq, img_x_points, img_y_points),
            palette=colour_box.value)
        else:
            # No point in generating an image for the UI if we're not in a notebook.
            return None

    img = ui_image()
    return (
        __IMG_SIZE__,
        img,
        img_seq,
        img_x_max,
        img_x_min,
        img_x_points,
        img_y_max,
        img_y_min,
        img_y_points,
        ui_image,
    )


@app.cell
def _(colour_box, img, its_box, mo, seq_box, x_img_slider, y_img_slider):
    mo.vstack(
        [
            img,
            mo.hstack([x_img_slider, y_img_slider]),
            seq_box,
            its_box,
            colour_box,
        ],
        align="center",
    )
    return


@app.cell
def _(mo):
    mo.md("""The repeated sequence of coeffecients can extended beyond $A$ and $B$. If a third, $C$, that varies over time is added, then an animation can be produced. More pleasingly, if there are $C$ and $D$ coefficients in a sequence, they can rotate in a circle so the animation can return to the start and repeat.""")
    return


@app.cell
def _(np, npt):
    def rot_coeffs(x: float, y: float, radius: float, n: int) -> npt.NDArray:
        """Return an array of n pairs of coefficients centred at (x, y) with radius r."""
        theta = np.linspace(-np.pi, np.pi, n)
        rot = np.zeros(shape=(n, 2, 2))
        rot[:, 0, 0] = np.cos(theta)
        rot[:, 0, 1] = np.sin(theta)
        rot[:, 1, 0] = -rot[:, 0, 1]
        rot[:, 1, 1] = rot[:, 0, 0]
        point = np.zeros(shape=(1, 1, 2))
        point[:, :, -1] = radius
        return (
            np.array([x, y]).reshape((1, 1, 2)) + point @ rot
        ).reshape(n, 2)

    def extra_coeffs(point: npt.ArrayLike, shape: tuple[int, int]) -> tuple[npt.NDArray, npt.NDArray]:
        """Return two arrays of coefficients with the given shape"""
        c, d = point
        c_coeff = np.zeros(shape)
        c_coeff.fill(c)
        d_coeff = np.zeros(shape)
        d_coeff.fill(d)
        return (c_coeff, d_coeff)
    return extra_coeffs, rot_coeffs


@app.cell
def _(mo, palettes):
    rot_seq_box = mo.ui.text(value='AACBABD', label='coefficient sequence')
    x_rot_range = mo.ui.range_slider(start=2.0, stop=4.0, step=0.1, value=[2.0, 4.0],
                                   label='A range')
    y_rot_range = mo.ui.range_slider(start=2.0, stop=4.0, step=0.1, value=[2.0, 4.0],
                                   label='B range')
    rad_box = mo.ui.slider(start=0.1, stop=1.0, step=0.05, value=0.25, label='CD radius')
    rot_colour_box = mo.ui.dropdown(palettes, value='managua', label='palette')
    play_pause = mo.ui.button(label='⏯', value=False, on_click=lambda v: not v)
    return (
        play_pause,
        rad_box,
        rot_colour_box,
        rot_seq_box,
        x_rot_range,
        y_rot_range,
    )


@app.cell
def _(mo, play_pause, rad_box):
    def tock():
        if play_pause.value:
            return mo.ui.refresh(default_interval=1, options=[0.5, 1, 2])
        else:
            return ''

    c_centre_slider = mo.ui.slider(start=2+rad_box.value, stop=4-rad_box.value,
                                   step=0.05, value=3, label='C centre')
    d_centre_slider = mo.ui.slider(start=2+rad_box.value, stop=4-rad_box.value,
                                   step=0.05, value=3, label='D centre')
    return c_centre_slider, d_centre_slider, tock


@app.cell
def _(
    __IMG_SIZE__,
    c_centre_slider,
    d_centre_slider,
    itertools,
    mo,
    np,
    rad_box,
    rot_coeffs,
    tock,
    x_rot_range,
    y_rot_range,
):
    tick = tock()

    rot_x_min, rot_x_max = x_rot_range.value
    rot_y_min, rot_y_max = y_rot_range.value

    rot_img_x_points, rot_img_y_points = np.meshgrid(
        np.linspace(rot_x_min, rot_x_max, __IMG_SIZE__),
        np.linspace(rot_y_max, rot_y_min, __IMG_SIZE__),
        indexing='xy'
    )

    rot_img_coeffs = rot_coeffs(c_centre_slider.value, d_centre_slider.value,
                                rad_box.value, 360)

    rot_img_slider = mo.ui.slider(start=0, stop=359, step=1, label='CD rotation')

    cycle = itertools.cycle(range(0, 360, 2))
    return (
        cycle,
        rot_img_coeffs,
        rot_img_slider,
        rot_img_x_points,
        rot_img_y_points,
        rot_x_max,
        rot_x_min,
        rot_y_max,
        rot_y_min,
        tick,
    )


@app.cell
def _(cycle, play_pause, rot_img_slider, tick):
    tick
    if play_pause.value:
        rotation = cycle.__next__()
    else:
        rotation = rot_img_slider.value
    return (rotation,)


@app.cell
def _(
    extra_coeffs,
    lyapunov,
    render_image,
    rot_colour_box,
    rot_img_coeffs,
    rot_img_x_points,
    rot_img_y_points,
    rot_seq_box,
    rotation,
    seq_vector,
):
    rot_img_c, rot_img_d = extra_coeffs(
        rot_img_coeffs[rotation],
        rot_img_x_points.shape
    )

    rot_img_seq = seq_vector(
        ''.join(filter(lambda char: char in 'ABCD', rot_seq_box.value.upper())), 100
    )

    rot_img = render_image(
        lyapunov(rot_img_seq, rot_img_x_points, rot_img_y_points, rot_img_c, rot_img_d),
        palette=rot_colour_box.value
    )
    return rot_img, rot_img_c, rot_img_d, rot_img_seq


@app.cell
def _(
    c_centre_slider,
    d_centre_slider,
    mo,
    play_pause,
    rad_box,
    rot_colour_box,
    rot_img,
    rot_img_slider,
    rot_seq_box,
    tick,
    x_rot_range,
    y_rot_range,
):
    mo.vstack([
        rot_img, rot_seq_box,
        mo.hstack([x_rot_range, y_rot_range], justify='start'),
        mo.hstack([c_centre_slider, d_centre_slider], justify='start'),
        mo.hstack([rad_box, rot_img_slider], justify='start'),
        mo.hstack([rot_colour_box, play_pause, tick], justify='start')
    ], align='center')
    return


@app.cell
def _(np, shared_memory):
    def get_shared_np(shape: tuple[int, ...], dtype: str='float64', name=None):
        """Return a SharedMemory instance, and a numpy array of the given
        shape that points to it."""
        dtype = np.dtype(dtype)
        size=dtype.itemsize * np.prod(shape)
        if name is None:
            buff = shared_memory.SharedMemory(create=True, size=size)
        else:
            buff = shared_memory.SharedMemory(name=name, create=False, size=size)
        arr = np.ndarray(shape, dtype, buffer=buff.buf)
        return buff, arr
    return (get_shared_np,)


@app.cell
def _(extra_coeffs, get_shared_np, lyapunov):
    def lyapunov_mp(cd: tuple[float, float], out_name: str, shape: tuple[int, int],
        its: int, seq_name: str, x_name: str, y_name: str):
        """Compute a Lyapunov fractal using arrays backed by shared memory.

        cd: Tuple of floats giving the C, D coefficients constant across the image.
        out_name: Name of a SharedMemory buffer to hold the Lyapunov exponents.
        shape: Tuple of ints giving the shape of the A, B coefficient and output arrays.
        its: Number of iterations, the length of the sequence array.
        seq_name: Name of a SharedMemory buffer pointing to the coefficient sequence array.
        x_name, y_name: Names of SharedMemory buffers pointing to the A and B coefficient arrays.
        """
        seq_buff, seq_vec = get_shared_np((its,), dtype='int32', name=seq_name)
        x_buff, x_coeff = get_shared_np(shape, name=x_name)
        y_buff, y_coeff = get_shared_np(shape, name=y_name)
        out_buff, out = get_shared_np(shape, name=out_name)
        c_coeff, d_coeff = extra_coeffs(cd, shape)
        out[:, :] = lyapunov(seq_vec, x_coeff, y_coeff, c_coeff, d_coeff)
        for buff in (seq_buff, x_buff, y_buff, out_buff):
            buff.close()
        return out_name
    return (lyapunov_mp,)


@app.cell
def _(
    Pool,
    get_shared_np,
    io,
    lyapunov_mp,
    np,
    partial,
    render_image,
    rot_coeffs,
    seq_vector,
):
    def video_seq_mp(seq: str, x_mi: float, x_mx: float, y_mi: float, y_mx: float,
        x: float, y: float, r: float, n: int,
        cores: int,
        its: int=100, pal: str='managua', w: int=512, h: int=512):
        """Yield an animated sequence of Lyapunov images using multiprocessing and SharedMemory.

        seq: String representing the repeating coefficient sequence, e.g: "AACBABD".
        x_mi, y_mi, x_mx, y_mx: Floats describing the boundaries of the images, the ranges
        of the A and B coefficients.
        x, y, r: Floats giving the centre and radius of a circle on which the C, D coefficients lie.
        n: Number of points around the circle, the number of frames.
        cores: Number of cores to use in the Pool, not including the calling process which
        compresses the image and the ffmpeg process that consumes them.
        its: Number of iterations for each image.
        pal: Name of the Matplotlib palette to use.
        w, h: Image width and height.
        """

        img_shape = (h, w)
        """The sequence of images is divided into n_chunks chunks of size chunk_size,
        each with a block of SharedMemory to hold each image before it is yielded.
        pool_chunk_size images are computed by the Pool at a time.
        """    
        chunk_size = 16 * cores
        n_chunks = n // chunk_size
        pool_chunk_size = 4 * cores

        # Reserve SharedMemory for the coefficient sequence and A, B coefficients:
        seq_buff, seq_vec = get_shared_np((its,), 'int32')
        x_buff, x_coeff = get_shared_np(img_shape)
        y_buff, y_coeff = get_shared_np(img_shape)

        x_coeff[:], y_coeff[:] = np.meshgrid(
            np.linspace(x_mi, x_mx, w), np.linspace(y_mx, y_mi, h),
        indexing='xy')
        seq_vec[:] = seq_vector(seq, its)

        # Reserve SharedMemory for the images.
        out_buffs, out_arrays = zip(*[get_shared_np(img_shape) for _ in range(chunk_size)])
        out_map = {buff.name: array for buff, array in zip(out_buffs, out_arrays)}

        cd_coeff = rot_coeffs(x, y, r, n)

        """All the arguments of lyapunov_mp remain constant, except for the C, D coefficients
        and the name of the output buffer to hold the image. Thus, no large numpy arrays will
        be serialized when the function is passed to the Pool."""
        lyap = partial(lyapunov_mp,
            shape=img_shape, its=its,
            seq_name=seq_buff.name, x_name=x_buff.name, y_name=y_buff.name,
        )

        # Divide the C, D coefficients among the chunks.
        chunks = np.array_split(cd_coeff, n_chunks)

        with Pool(cores) as pool:
            for chunk in chunks:
                for out_name in pool.starmap(lyap, zip(chunk, out_map.keys()), pool_chunk_size):
                    buff = io.BytesIO()
                    """The job of turning the returned images is left to the calling process.
                    There's no point compressing the images when ffmpeg would have to uncompress
                    them afterwards."""
                    render_image(out_map[out_name]).save(buff, format='PNG', compress_level=0)
                    yield buff.getvalue()

        for shm in (seq_buff, x_buff, y_buff):
            shm.close()
            shm.unlink

        for shm in out_buffs:
            shm.close()
            shm.unlink()

        # Need to yield something so that the SharedMemory is freed.
        yield None
    return (video_seq_mp,)


@app.cell
def _(sp):
    def render_video(fname: str, im_seq, fps: int=30, quiet: bool=True):
        """Stream a sequence of .png images to ffmpeg, and turn them into an .mp4 video.
        fname: Filename for the video.
        fps: Frames per second, defaults to 30.
        quiet: Whether to suppress ffmpeg's rather verbose output.
        """
        ffmpeg_cmd = [
            'ffmpeg', '-threads', '1', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps),
            '-i', '-', '-vcodec', 'libx264', '-q:a', '0', fname
        ]
        if quiet:
            ffmpeg_out = sp.DEVNULL
        else:
            ffmpeg_out = None
        proc = sp.Popen(ffmpeg_cmd, stdin=sp.PIPE, stdout=ffmpeg_out, stderr=ffmpeg_out)
        # Filter the sequence to omit the finall "None".
        for im in filter(None, im_seq):
            proc.stdin.write(im)
        proc.stdin.close()
        proc.wait()
    return (render_video,)


@app.cell
def _(os):
    __TOTAL_CORES__ = os.environ.get('OMP_NUM_THREADS', len(os.sched_getaffinity(0)))
    if __TOTAL_CORES__ > 2:
        __MAX_VID_SEQ_CORES__ =__TOTAL_CORES__ - 2
    else:
        __MAX_VID_SEQ_CORES__ = 1
    return __MAX_VID_SEQ_CORES__, __TOTAL_CORES__


@app.cell
def _(__MAX_VID_SEQ_CORES__, mo, rot_seq_box):
    duration_slider = mo.ui.slider(start=10, stop=300, value=60, label='duration (s)')
    fps_dropdown = mo.ui.dropdown(options=['24', '25', '30', '50', '60'], value='30', label='fps')
    vid_size_dropdown = mo.ui.dropdown(options=['512x512', '640x480', '1280x720', '1024x1024',
                                               '1080x1080'], value='512x512')
    cores_dropdown = mo.ui.dropdown(
        options=list(map(str, range(1, __MAX_VID_SEQ_CORES__+1))),
        value='1', label='cores'                               
    )
    video_fname = mo.ui.text(value='{}.mp4'.format(rot_seq_box.value))
    do_video = mo.ui.run_button(label='generate video')
    return (
        cores_dropdown,
        do_video,
        duration_slider,
        fps_dropdown,
        vid_size_dropdown,
        video_fname,
    )


@app.cell
def _(
    c_centre_slider,
    cores_dropdown,
    d_centre_slider,
    duration_slider,
    fps_dropdown,
    its_box,
    rad_box,
    rot_colour_box,
    rot_seq_box,
    rot_x_max,
    rot_x_min,
    rot_y_max,
    rot_y_min,
    vid_size_dropdown,
    video_seq_mp,
):
    video_fps = int(fps_dropdown.value)
    video_frames = duration_slider.value * video_fps
    vid_width, vid_height = map(int, vid_size_dropdown.value.split('x'))

    vs = video_seq_mp(
        rot_seq_box.value.upper(),
        rot_x_min, rot_x_max, rot_y_min, rot_y_max,
        c_centre_slider.value, d_centre_slider.value,
        rad_box.value, video_frames, int(cores_dropdown.value),
        its_box.value, rot_colour_box.value,
        vid_width, vid_height
    )
    return vid_height, vid_width, video_fps, video_frames, vs


@app.cell
def _(do_video, mo, render_video, video_fname, video_fps, video_frames, vs):
    if do_video.value and mo.running_in_notebook():
        def vid_seq():
            with mo.status.progress_bar(total=video_frames) as prog:
                for frame in vs:
                    yield frame
                    prog.update()
        render_video(video_fname.value, vid_seq(), fps=video_fps)
    return (vid_seq,)


@app.cell
def _(
    cores_dropdown,
    do_video,
    duration_slider,
    fps_dropdown,
    mo,
    sys,
    vid_size_dropdown,
    video_fname,
):
    if 'pyodide' not in sys.modules:
        video_ui = [
            mo.hstack([vid_size_dropdown, duration_slider, fps_dropdown], justify='start'),
            mo.hstack([cores_dropdown, video_fname, do_video], justify='start')
        ]
    else:
        video_ui = [
            'No video generation in web-assembly.',
            'Needs multiprocessing, subprocess and ffmpeg.'
        ]

    mo.vstack(video_ui, align='center')
    return (video_ui,)


@app.cell
def _():
    __DEFAULT_ARGS__ = {
        'seq': 'AACBABD',
        'xmin': 2.0, 'xmax': 4.0, 'ymin': 2.0, 'ymax': 4.0,
        'xc': 3.0, 'yc': 3.0, 'rad': 0.2, 'its': 100,
        'width': 512, 'height': 512, 'dur': 60, 'fps': 30,
        'cores': 1, 'pal': 'managua'
    }
    return (__DEFAULT_ARGS__,)


@app.cell
def _(__DEFAULT_ARGS__, datetime, mo, render_video, video_seq_mp):
    # If this is being run as a script:
    if not mo.running_in_notebook():
        args = mo.cli_args()
        def get_arg(arg):
            return args.get(arg, __DEFAULT_ARGS__.get(arg))
        sq = args.get('seq')
        fname = args.get('fname')

        fps = get_arg('fps')
        n_frames = fps * get_arg('dur')

        xmin, xmax, ymin, ymax, xc, yc, rad = map(
            get_arg, ['xmin', 'xmax', 'ymin', 'ymax', 'xc', 'yc', 'rad'])

        its, cores, pal, width, height = map(get_arg, ['its', 'cores', 'pal', 'width', 'height'])

        img_sq = video_seq_mp(sq, xmin, xmax, ymin, ymax, xc, yc, rad, n_frames, cores,
            its=its, pal=pal, w=width, h=height)

        right_now = datetime.now()
        render_video(fname, img_sq, fps=fps, quiet=False)
        print('Wrote {} in {}s.'.format(
            fname, (datetime.now()-right_now).total_seconds()
        ))
    return (
        args,
        cores,
        fname,
        fps,
        get_arg,
        height,
        img_sq,
        its,
        n_frames,
        pal,
        rad,
        right_now,
        sq,
        width,
        xc,
        xmax,
        xmin,
        yc,
        ymax,
        ymin,
    )


if __name__ == "__main__":
    app.run()
