import type { DecoySlots } from "../../api/types";

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/**
 * Position of a subnet's decoy slot. Mirrors
 * cyberwheel/visualization/layout.py::decoy_slot_position — keep the two in
 * sync so a decoy renders where the writer reserved space for it.
 */
export function decoySlotPosition(slots: DecoySlots, slotIndex: number): [number, number] {
  const index = slots.base + slotIndex;
  const radius = slots.spacing * Math.sqrt(index + 1);
  const theta = index * GOLDEN_ANGLE + slots.rot;
  return [slots.cx + radius * Math.cos(theta), slots.cy + radius * Math.sin(theta)];
}
