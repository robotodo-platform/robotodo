- user selection: equiv to $0 in devtools:
```
import omni.usd

# Get the Selection singleton
selection = omni.usd.get_context().get_selection()

# Returns a list of selected prim paths (strings like "/World/MyCube")
paths = selection.get_selected_prim_paths()

print("Selected prim paths:", paths)
```

- debug draw:
```
from isaacsim.replicator.grasping.ui import grasping_ui_utils

import pxr

# TODO
grasping_ui_utils.clear_debug_draw()
grasping_ui_utils.draw_grasp_samples_as_lines?

```

