# robotodo-isaac
Robotodo NVIDIA Omniverse simulation engine.

## Usage
Audience: general public.

Refer to the documentation published 
[here](https://robotodo-isaac.readthedocs.io)
or the `docs` directory.

## Development
Audience: package developers/contributors.

Directory structure:
- `packages`: Package source code.
- `docs`: Package documentation.

Setup:
- Clone this repository and `cd` into it.
- Run `pip install --editable .`.
- (Optional) Start unit testing: 
    - Run `pip install --editable .[test]`.
    - Run `pytest packages/`.
- (Optional) Build the documentation:
    - Run `pip install --editable .[docs]`.
    - Run `jupyter-book build docs/`.
