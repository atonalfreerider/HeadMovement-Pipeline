// AudioWorklet: records channel 0 of its input as a gap-free sample stream aligned with the AudioContext clock.
// Recording starts at the first render quantum the node processes (frame0, posted with the first batch); every later
// quantum is written - as silence while the microphone has not delivered anything yet (inputs[0] is empty) - and a
// jump in currentFrame is padded with zeros. So recorded sample k is context frame frame0 + k exactly, which is what
// the page's player log (context times) is mapped against.
class PcmRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.on = true;
    this.buf = new Float32Array(4096);
    this.n = 0;
    this.next = null;   // context frame of the next sample to record
    this.batch0 = 0;    // context frame of buf[0]
    this.port.onmessage = (e) => {
      if (e.data === 'stop') {
        this.flush();
        this.on = false;
        this.port.postMessage({ done: true });
      }
    };
  }
  flush() {
    if (this.n > 0) {
      this.port.postMessage({ frame: this.batch0, data: this.buf.slice(0, this.n) });
      this.n = 0;
    }
  }
  push(x) {
    if (this.n === 0) this.batch0 = this.next;
    this.buf[this.n++] = x;
    this.next++;
    if (this.n === this.buf.length) this.flush();
  }
  process(inputs) {
    if (!this.on) return false;
    const f = currentFrame;
    if (this.next === null) this.next = f;
    while (this.next < f) this.push(0);           // frames we were not called for: silence
    const ch = inputs[0] && inputs[0][0];
    const len = ch ? ch.length : 128;
    for (let i = 0; i < len; i++) this.push(ch ? ch[i] : 0);
    return true;
  }
}
registerProcessor('pcm-recorder', PcmRecorder);
