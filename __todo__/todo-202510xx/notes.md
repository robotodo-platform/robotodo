```
    
    # TODO rm
    # TODO NOTE mandatory warmup
    def hacky_ensure_render(self):
        # TODO call on every stage opening event
        settings = self.get_settings()
        timeline = self.omni.timeline.acquire_timeline_interface()
        play_simulations_orig = settings.get("/app/player/playSimulations")
        settings.set("/app/player/playSimulations", False)
        timeline.forward_one_frame()
        timeline.rewind_one_frame()
        settings.set("/app/player/playSimulations", play_simulations_orig)

    # TODO
    def render(self, num_passes: int = 1):
        settings = self.get_settings()
        play_simulations_orig = settings.get("/app/player/playSimulations")
        settings.set("/app/player/playSimulations", False)
        for _ in range(num_passes):
            self._app_framework.update()
        settings.set("/app/player/playSimulations", play_simulations_orig)

```