import { initIdentity } from './identity.js';

const socket = new WebSocket('ws://localhost:8000/ws');

socket.onopen = () => {
  console.log('[identity] ws connected');
};

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);

  // 你现在 live-stage 已经在发这些
  const {
    z_A,
    z_B,
    z_mix,
    alpha,
    t
  } = data;

  initIdentity({ z_mix, alpha, t });
};
