import os
import contextlib
import asyncio

import nest_asyncio


class Kernel:
    """
    TODO doc
    
    """

    def __init__(
        self, 
        extra_argv: list[str] = [],
        kit_path: str | None = None, 
        loop: asyncio.AbstractEventLoop = None,
    ):
        """
        Create an Isaac Sim "kernel".
        At its core, this starts an Isaac Sim Omniverse App in the background.
        
        :param extra_argv:
            Extra CLI arguments to pass to the Isaac Sim Omniverse App.

            Common arguments include:

            * :code:`--help`: Show help message and shut down.
            * :code:`--no-window`: Switch to headless mode (disables window).
            * :code:`--/app/window/hideUi=True`: Hide all UI elements (still opens a window).

            .. seealso::
                * :class:`isaacsim.simulation_app.AppFramework`
                * :class:`isaacsim.simulation_app.SimulationApp`
                * https://docs.omniverse.nvidia.com/kit/docs/kit-manual/108.0.0/guide/configuring.html#kit-kernel-settings

        :param kit_path: 
            Path to the kit file (ends in `.kit`) that defines an Omniverse App.
            If not provided, uses Isaac Sim's default full app when possible.
            
            .. seealso::
                * https://docs.omniverse.nvidia.com/kit/docs/kit-manual/108.0.0/guide/creating_kit_apps.html

        :param loop:
            The `asyncio` loop to use for all async operations.
            Defaults to the current running loop.
        """

        # NOTE this ensures a loop is present beforehand 
        # to prevent isaacsim from "stealing" the default loop
        self._loop = loop
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        nest_asyncio.apply()

        @contextlib.contextmanager
        def _undo_isaacsim_monkeypatching():
            """
            Undo the unnecessary monkey-patching 
            of builtin Python modules done by `omni.kit.app`.

            .. seealso::
                * :func:`_startup_kit_scripting` in 
                `site-packages/omni/kernel/py/omni/kit/app/_impl/__init__.py`
            """

            import logging

            logging_handlers_orig = list(logging.root.handlers)
            logging_level_orig = int(logging.root.level)

            yield

            logging.root.handlers = logging_handlers_orig
            logging.root.level = logging_level_orig

        with _undo_isaacsim_monkeypatching():
            import isaacsim
            isaacsim.bootstrap_kernel()
            from isaacsim.simulation_app import AppFramework

            # TODO
            if kit_path is None:
                for p in [
                    "isaacsim.exp.full.kit",
                    "omni.isaac.sim.python.kit",
                    "isaacsim.exp.base.python.kit",
                    "isaacsim.exp.base.kit",
                ]:
                    p = os.path.join(os.environ["EXP_PATH"], p)
                    if os.path.isfile(p):
                        kit_path = p
                        break
            exe_path = os.environ["CARB_APP_PATH"]

            self._app_framework = AppFramework(argv=[
                os.path.abspath(kit_path),
                # run as portable to prevent writing extra files to user directory
                "--portable",
                # extensions
                # extensions: adding to json doesn't work
                "--ext-folder", os.path.abspath(os.path.join(os.environ["ISAAC_PATH"], "exts")),
                # extensions: so we can reference other kit files  
                "--ext-folder", os.path.abspath(os.path.join(os.environ["ISAAC_PATH"], "apps")),      
                # ...      
                # this is needed so dlss lib is found
                f"--/app/tokens/exe-path={os.path.abspath(exe_path)}",
                # "--/app/fastShutdown=true",
                "--/app/installSignalHandlers=false",
                "--/app/python/interceptSysStdOutput=false",
                "--/app/python/interceptSysExit=false",
                "--/app/python/logSysStdOutput=false",
                "--/app/python/enableGCProfiling=false",
                "--/app/python/disableGCDuringStartup=false",
                # TODO
                # "--/app/extensions/fastImporter/enabled=false",
                # logging
                # logging: disable default log file creation
                "--/log/file=",
                # "--/log/enabled=false",
                *extra_argv,
            ])

            import isaacsim
            import omni
            import pxr

            self._isaacsim = isaacsim
            self._omni = omni
            self._pxr = pxr

            # TODO load extensions

        # TODO !!!!!
        self._should_run_app_loop = False

    @property
    def isaacsim(self):
        # TODO !!!!
        return self._isaacsim

    @property
    def omni(self):
        # TODO !!!!
        return self._omni

    @property
    def pxr(self):
        # TODO !!!!
        return self._pxr
    
    def start_app_loop_soon(self):
        def f():
            if not self._should_run_app_loop:
                return
            self._app_framework.update()
            self._loop.call_soon(f)
        
        self._should_run_app_loop = True
        self._loop.call_soon(f)

    def stop_app_loop_soon(self):
        self._should_run_app_loop = False

    def step_app_loop(self):
        self._app_framework.update()