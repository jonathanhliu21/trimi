
1. Home the leader.

~/TriMI-Copilot/.venv/bin/trimi-live home \
  --hand-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO9P0-if00-port0 \
  --out ~/.config/trimi-copilot/hand_leader.yaml




2. Home the follower 

~/TriMI-Copilot/.venv/bin/trimi-live home \
  --hand-config ~/.config/trimi-copilot/hand_leader.yaml \
  --hand-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO3G5-if00-port0 \
  --out ~/.config/trimi-copilot/hand_follower.yaml



python3 scripts/teleop/trimi_teleop.py --leader-port $L --follower-port $F   --force-feedback --grasp-only --plot --print --log $(date +%m%d%H%M)