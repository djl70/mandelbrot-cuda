# Mandelbrot Escape Time Algorithm in CUDA

![Example Mandelbrot visualization](https://github.com/djl70/mandelbrot-cuda/blob/fd655b0c1989a90f944048dc4c1be74e217a3cad/mandelbrot068.png)

## Requirements
- CUDA
- CUDA-capable GPU
- Linux-based OS (probably)
- (optional) ImageMagick (used to combine PPM images into an animated GIF)

## Usage
Download or clone the repository, extract it if necessary, then build and run the code:
```
nvcc -std=c++11 mandelbrot.cu -o mandelbrot
./mandelbrot 16 240 1
```
For information about the command-line arguments, run:
```
./mandelbrot
```
To combine the output PPM files into an animated GIF, run:
```
convert -delay 10 mandelbrot*ppm mandelbrot.gif
```

## Acknowledgments
- [This great source](http://warp.povusers.org/Mandelbrot/)
- [Wikipedia](https://en.wikipedia.org/wiki/Mandelbrot_set)
- [Mandelbrot Viewer](http://math.hws.edu/eck/js/mandelbrot/MB.html)
- [Wolfram](http://mathworld.wolfram.com/MandelbrotSet.html)
- Dr. Burtscher for making CUDA seem easy
