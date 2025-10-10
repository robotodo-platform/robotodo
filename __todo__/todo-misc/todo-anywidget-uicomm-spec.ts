
interface ViewerEvent {
    id: string;
}

interface WindowConfigureEvent extends ViewerEvent {
    id: "win:conf";
    size?: [x: number, y: number];
    scale?: number; // scale applied to size
}

interface WindowFocusEvent extends ViewerEvent {
    id: "win:focus";
}


interface VideoConfigureEvent extends ViewerEvent {
    id: "vid:conf";
    codec: string;
    description: Buffer;
}

interface VideoDecodeEvent extends ViewerEvent {
    id: "vid:dec";
    type: "key" | "delta";
    timestamp: number; // TODO in us
    data: Buffer;
}


interface ClipboardInputEvent extends ViewerEvent {
    id: "inp:clb";
    items: {[mime: string]: Buffer};
}

interface KeyboardInputEvent extends ViewerEvent {
    id: "inp:kb";
    action?: "up" | "down" | null;
    key: string;
}

interface PointerInputEvent extends ViewerEvent {
    id: "inp:pt";
    action?: "up" | "down" | null;
    button?: number | null;
    pos?: [x: number, y: number];
}

interface WheelInputEvent extends ViewerEvent {
    id: "inp:whl";
    deltaPos: [x: number, y: number, z: number];
}


interface ViewerCommunication {
    // TODO ack?
    emit?: (event: ViewerEvent) => void;
    // TODO async??
    on?: (listener: (event: ViewerEvent) => void) => void;
    off?: (listener: (event: ViewerEvent) => void) => void;
}
