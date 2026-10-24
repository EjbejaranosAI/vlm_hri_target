# Next steps — vlm_hri (sesión 2026-09-19)

Contexto: working tree tiene un WIP grande sin commitear sobre el último commit
que funcionaba en TIAGo Pro (`9f14aec`) — separación pose(geometría)/action(VLM),
confidence/interrupt_cost, TargetStability. En esta sesión se depuró con la
webcam local (`python main.py stream --camera 0 --display`, `.venv`, robot
apagado). Ver `git diff` para el estado completo sin commitear.

## Pendiente

- [ ] **Confirmar el fix de `walking` con torso solo (sin piernas visibles).**
  Se quitó la señal experimental sin calibrar `chunk_depth_motion_by_pid` de
  `raw_moving` en `vlm_hri/runners/pose_session.py::_pose_vlm_worker`, y se
  filtró `bbox_confirmed` por `legs_visible` antes de forzar `walking` en
  `posture_hint` (`vlm_hri/pose/gait.py::refine_labels_pose`). Falta
  confirmar en vivo que ya no aparece `walking` estando parado cerca de la
  cámara (solo torso en el frame).
- [ ] **`"s": "sitting"` mal formado al arrancar.** En los primeros 2 trozos
  del último test el VLM devolvió `{"a": "idle", "s": "sitting", "c": 2}`
  (confundió el campo de intención `s` con una palabra de postura) y cayó a
  `unknown`/`AVAILABLE` por defecto. Se autocorrigió solo después de esos 2
  trozos y no volvió a pasar en los siguientes 18. Vigilar si se repite de
  forma consistente (no solo en el arranque/warmup) — si es así, reforzar el
  prompt (`vlm_hri/vlm/prompt_templates/compact.json`, campo `'s'`) para que
  no admita palabras de postura ahí tampoco.
- [x] ~~`chunk_motion_by_pid` confundía balanceo/jitter con caminar~~ —
  recalibrado 2026-10-02 (avance neto + sin saltos de tracking, ver
  docstring). Queda abierto: caminar HACIA la cámara apenas desplaza el
  centro de la caja y las piernas suelen quedar fuera de cuadro
  (empresarios.mp4: 1/30 trozos detectados como MOVING) — haría falta una
  señal de crecimiento de caja calibrada (`chunk_depth_motion_by_pid` sigue
  experimental).
- [ ] **Decidir si el resto del WIP se commitea.** El working tree tiene
  cambios grandes sin commitear (confidence/interrupt_cost, TargetStability,
  fps_calibration en el nodo ROS, `/vlm_hri/pose_skeleton`, etc.) — revisar
  con `git status`/`git diff` y decidir si se prueban en el robot real antes
  de commitear, o si se commitea por partes.
- [ ] **Volver a probar en el robot (TIAGo Pro) cuando esté encendido**, ya
  que todo lo de esta sesión se validó con la webcam local (`.venv`), no con
  `/head_front_camera` + `dynamic_tracking` (`.venv_ros` + `ros2 launch`).

## Cómo retomar

```bash
cd /home/edison/Desktop/PhD/RAL2026/vlm_hri_target
source .venv/bin/activate
python main.py stream --camera 0 --display
```

Para el robot (cuando esté encendido): ver `run_vlm_hri_node_tiago_pro.txt`
en `/home/edison/Desktop/PhD/RAL2026/`.
