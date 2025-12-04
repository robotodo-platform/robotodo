Installation
============

Before you begin, complete the following check list:

- **Internet**
  The installation requires internet to complete.
  Offline installation is possible but is out of scope for this document.

- **Python**

  - Ensure **Python 3.11** [#ref-isaacsim_pip]_ is installed and available as ``python3``. 
  Run ``python3 --version`` to check.

  - Ensure Python package manager pip is installed and available as ``python3 -m pip``.
  Run ``python3 -m ensurepip`` to make sure.

- **Operating system**

  - **Linux**: Check your GLIBC version with ``ldd --version``.
    Upgrade your distribution if GLIBC is below **2.35**. 
    Find supported versions of common distributions 
    `here <https://gist.github.com/richardlau/6a01d7829cc33ddab35269dacc127680>`_. [#ref-isaacsim_pip]_

  - **Windows**: You may need to enable `long path support <https://pip.pypa.io/warnings/enable-long-paths>`_ 
    to avoid installation errors caused by path length limits. [#ref-isaacsim_pip]_


Getting Started
---------------

The easiest way to install ``robotodo-isaac`` is through ``pip``:

.. code-block:: shell

  python3 -m pip install 'robotodo-isaac @ git+https://github.com/robotodo-platform/robotodo-isaac.git'


.. note:: 
  Effort to publish to `PyPI <https://pypi.org>`_ is underway.


References
----------

.. [#ref-isaacsim_pip] https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html#install-isaac-sim-using-pip

