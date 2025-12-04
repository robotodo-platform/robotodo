# robotodo-isaac
Robotodo NVIDIA Omniverse simulation engine

## Installation

Robotodo may be installed from:
- [PyPI](https://pypi.org): 
    Package has not been published yet.
    Effort to publish to [PyPI](https://pypi.org) is underway.
- Source:
    ```sh
    pip install robotodo-isaac @ git+{url}
    ```
    Replace `{url}` with the Git URL of the repository.
    ([Details](https://pip.pypa.io/en/stable/cli/pip_install/#examples.))

## Usage

Directory structure:
- `packages`: Package source code.
- `docs`: Documentation and tutorials.

Details such as API references
can be found in the `docs` directory.

## Development

- Clone this repository and `cd` into it.
- Run `pip install --editable .`.
- (Optional) Start unit testing: 
    - Run `pip install --editable .[test]`.
    - Run `pytest packages/`.
- (Optional) Build the documentation:
    - Run `pip install --editable .[docs]`.
    - Run `jupyter-book build docs/`.

Details such as developer guidelines and roadmaps
can be found in the `docs` directory.