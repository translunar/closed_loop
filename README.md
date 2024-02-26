# closed_loop

This demonstrates a closed loop, single-threaded mass-spring-damper with a
PID controller. I've also demonstrated some zero-mean Gaussian noise, and
plotting is carried out by a separate script. Models can run at different
rates, and the example includes two logging rate groups as well.

## Running the example

    source .venv/bin/activate
    pip install -r requirements.txt
    python world.py && python plotter.py

To run unit tests,

    pytest test/

## A note about typing

I decided against using mypy to check typing because it would be difficult
to finish this task in five hours with typing.

## Copyright

Copyright Juno Woods, Ph.D., Translunar LLC, 2024. All rights reserved.