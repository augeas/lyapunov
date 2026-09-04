# lyapunov
## Experiments with Marimo, Numpy and Multiprocessing SharedMemory

The [Lyapunov Fractal](https://en.wikipedia.org/wiki/Lyapunov_fractal) is calculated
from repeated iterations of the [Logistic Map](https://en.wikipedia.org/wiki/Logistic_map):

$$ \Large{x_{n+1} = r_{n}x_{n}(x_{n}-1)} $$

where at each iteration $r_{n}$ takes values from some repeated sequence, for 
example $AABAB$. For each point in an image $A$ and $B$ take the values of
the $x$ and $y$ coordinates. For a large number of iterations $N$, the Lyapunov
exponent $\lambda$ is found for each point, and coloured accordingly:

$$ \Large{\lambda = \dfrac{1}{N}\sum_{n=1}^{N}|r_{n}(1-2x_{n})|} $$

Code to play with Lyapunov fractals and animate them has been implemented as a [Marimo notebook](https://marimo.io/).
To experiment with it, either start a [virtualenv](https://docs.python.org/3/library/venv.html),
clone this repo, install the dependencies and then run the notebook...

```bash
mkdir marimo_env
python3 -m venv marimo_env
source ./marimo_env/bin/activate
git clone https://github.com/augeas/lyapunov.git
cd lyapunov
pip install -r requirements.txt
marimo edit
```

...or, since Marimo enables [export to web-assembly](https://docs.marimo.io/guides/wasm/) via
[Pyodide](https://pyodide.org/en/stable/), you play with a web-app
based on the code at [https://augeas.github.io/lyapunov](https://augeas.github.io/lyapunov).
Since the code uses Numpy, Pillow and Matplotlib, all of which are packaged by default by [Pyodide](https://pyodide.org/en/stable/),
there is no need to resort to [micropip](https://micropip.pyodide.org/en/latest/project/usage.html).

One might generate the "[Zircon Zity](https://en.wikipedia.org/wiki/Lyapunov_fractal#/media/File:Lyapunov-fractal.png)"
image on the [Wikipedia page](https://en.wikipedia.org/wiki/Lyapunov_fractal) with:

```python
from lyapunov import lyapunov_img
lyapunov_img('BBBBBBAAAAAA',
    x_min=2.5, x_max=3.4, y_min=3.4, y_max=4.0, its=400, width=900, height=600,
palette='managua').save('zircon_zity.png')
```

![Zircon Zity](img/zircon_zity.png)

The repeated sequence of coeffecients can extended beyond $A$ and $B$. If a third,
$C$, that varies over time is added, then an animation can be produced. More pleasingly,
if there are $C$ and $D$ coefficients in a sequence, they can rotate in a circle so the
animation can return to the start and repeat. The individual images are joined togther
in a video via [ffmpeg](https://ffmpeg.org/). This illustrate how Marimo notebooks can
be [run as scripts](https://docs.marimo.io/guides/scripts/), including the passing of
parameters:

```bash
python lyapunov.py --seq=ACDBCD --fname=ACDBCD_1080.mp4 --width=1080  --height=1080 --xc 2.95 --yc 2.95 --rad=0.25 --pal twilight --cores=2
```

Rather than write each image to disk before generating the video, they are
streamed to [ffmpeg](https://ffmpeg.org/) via standard input and the
[`subprocess`](https://docs.python.org/3/library/subprocess.html) module from the Python
standard library. Here, `cores` refers to the number of processes used to generate the frames,
another Python process is required to pass the images to [ffmpeg](https://ffmpeg.org/),
which has a process of its own.
