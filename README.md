# vlm_hri_target

Detección de acciones e intenciones de personas para interacción humano-robot
(HRI): a partir de video (archivo o cámara en vivo), detecta personas, sigue
su identidad entre frames, y usa un VLM (Qwen2-VL-2B) para clasificar su
acción (parado, caminando, sentado, hablando) y su estado social —
**AVAILABLE**, **ATTENTIVE**, **BUSY**, **ENGAGED**, **MOVING** — para que un
robot pueda decidir **a quién acercarse** para iniciar una interacción.

## Cómo funciona

1. **Detección + seguimiento** (`yolo11n.pt`): encuentra personas y les asigna
   un ID estable entre frames.
2. **Pose** (`yolo26n-pose.pt`, 17 puntos COCO): se pega a cada caja ya
   trackeada — se usa un modelo aparte solo para los keypoints porque su caja
   propia es menos estable para el seguimiento.
3. **Señal de movimiento independiente del VLM**: marcha real de piernas
   (alternancia del ángulo rodilla-cadera-tobillo) + cambio de profundidad de
   la caja + histéresis temporal, con compensación de movimiento de cámara
   (si la cámara tiembla/se mueve, no confunde a todo el mundo con caminando).
4. **VLM (Qwen2-VL-2B-Instruct, 4-bit)**: ve el clip de ~1s con esa señal
   como pista visual (caja azul=movimiento detectado, roja=no) y decide la
   acción/estado final combinando ambas fuentes.
5. **Correcciones de sesgo del VLM**: el VLM tiende a pegarle la misma
   etiqueta a todo un grupo ("todos están hablando", "todos están
   caminando") aunque no sea cierto — se corrige exigiendo evidencia
   cinemática independiente antes de aceptar una etiqueta grupal.
6. **Selección de target**: de las personas libres (ATTENTIVE > AVAILABLE,
   nunca alguien ENGAGED/BUSY/MOVING), elige la más cercana a la cámara como
   candidata para que el robot inicie la interacción.

Todo el video/cámara final muestra: caja de color por estado + ID, un panel
lateral con el detalle de cada persona, y el candidato a target resaltado.

## Estructura del código

```
main.py              # entrypoint CLI (video / videos / stream)
vlm_hri/              # paquete: detección, VLM, social-state, pose, runners
  config.py            # constantes de entorno
  detection.py          # YOLO + tracking
  clustering.py          # agrupación por proximidad real (LiDAR, vía ROS2)
  vlm/                  # carga del modelo, prompts, parseo, inferencia
  pose/gait.py           # marcha por piernas + señales de pose
  runners/              # video.py y stream.py (run(..., use_pose=True));
                        # pose_session.py: sesión de streaming reutilizable
                        # entre el CLI y el nodo ROS2
  ros/                  # nodo ROS2 (ver sección "ROS2" más abajo)
models/               # pesos de YOLO (se descargan solos, no versionados)
package.xml, setup.py, launch/, config/  # paquete ROS2 (ament_python)
```

## Requisitos

- Linux, Python 3.10+
- GPU NVIDIA con CUDA (probado en 8GB VRAM — el VLM corre en 4-bit)
- `ffmpeg` instalado en el sistema (recodifica el video final a H.264):
  `sudo apt install ffmpeg`

## Instalación

```bash
./setup.sh
source .venv/bin/activate
```

Los pesos de YOLO y el VLM se descargan solos la primera vez que corres algo
(no están en este repo). Si un modelo de HuggingFace te pide autenticación,
exporta tu propio token: `export HF_TOKEN=tu_token`.

## Uso

Coloca tus videos en `input_videos/` (no se versionan en git).

**Un video:**
```bash
python main.py video -i input_videos/mi_video.mp4
```

**Todos los videos de una carpeta:**
```bash
python main.py videos --input-dir input_videos
```

**Cámara en vivo:**
```bash
python main.py stream --camera 0 --display     # ventana en pantalla
python main.py stream --camera 0 --preview      # mpv/ffplay o navegador
```

**Video simulando cámara (para probar el modo streaming sin cámara física):**
```bash
python main.py stream -i input_videos/mi_video.mp4 --realtime --preview
```

Por defecto los tres modos usan el detector de pose (marcha por piernas).
Con `--no-pose` corren solo con detección (yolo11n), sin esa señal.

Los resultados quedan en `output/<nombre_del_video_o_camera_N>/`:
- `annotated_actions.mp4` / `annotated_stream.mp4` — video final con las cajas, panel y target.
- `vlm_input.mp4` — lo que realmente ve el VLM (con el esqueleto y la pista de color).
- `actions.json` — acciones/estados por persona, por trozo y el resumen final.
- `summary.json` — tiempos (YOLO, VLM, latencia media, factor de tiempo real).
- `detections_per_frame.csv` — cajas crudas por frame (modo video, no streaming).

## ROS2 (TIAGO / Gazebo)

`vlm_hri_node` sustituye la cámara/detección/tracking propios por los de
[`dynamic-tracking`](../dynamic-tracking) (LiDAR 360° + YOLO auxiliar, ya con
tracking Kalman + ID estable + posición real en el mundo): se suscribe a la
imagen cruda del robot y a `/detections/tracked`
(`vision_msgs/Detection2DArray` — bbox, track_id embebido en `class_id` como
`"person#7"`, y posición real vía `results[0].pose`), corre localmente solo
el modelo de pose (para marcha/gait — dynamic_tracking no hace pose), y
alimenta la misma sesión de streaming que usa `python main.py stream`.

**Requiere un entorno con `numpy<2`.** `cv_bridge` (parte de ROS2) está
compilado contra NumPy 1.x; el `.venv` de este repo usa NumPy 2.x para
torch/ultralytics, y con eso `cv_bridge` falla al importar. Mismo problema
(y misma solución) que ya documenta `dynamic-tracking/README.md`: usa un
venv aparte con `numpy<2` para correr el nodo, o instala las dependencias de
`requirements.txt` con `numpy<2` en un entorno dedicado a ROS2. El `.venv`
normal (numpy 2.x) sigue sirviendo para `main.py` (CLI) sin tocar nada.

```bash
# en un venv/entorno con numpy<2 y acceso a los paquetes de ROS2 (rclpy,
# cv_bridge, vision_msgs, message_filters, ament_index_python, launch)
cd ~/ros_ws   # workspace con dynamic_tracking y este repo en src/
colcon build --packages-select vlm_hri_target
source install/setup.bash

ros2 launch vlm_hri_target vlm_hri.launch.py profile:=sim   # Gazebo
ros2 launch vlm_hri_target vlm_hri.launch.py profile:=robot # TIAGO real
```

Tópicos publicados (`config/vlm_hri_params_{sim,robot}.yaml` fija
`frame_id: map` en sim / `odom` en robot, igual que `tracking_frame` en
dynamic_tracking):

| Tópico | Tipo | Contenido |
|---|---|---|
| `/vlm_hri/image_annotated` | `sensor_msgs/Image` | Mismo video anotado que el CLI (cajas, panel, banner) |
| `/vlm_hri/social_states` | `std_msgs/String` (JSON) | `[{id, action, social_state, map_x, map_y, cluster_id, is_target}, ...]` |
| `/vlm_hri/target_pose` | `geometry_msgs/PoseStamped` | Posición real del target elegido |

`cluster_id` viene de agrupar personas por distancia real (LiDAR,
`ENGAGED_PROXIMITY_MAX_M`, default 2.0m) — además de exponerse, se usa para
corregir ENGAGED: si el VLM dice que dos personas están hablando pero su
posición real las tiene a más de esa distancia (o nadie más está ENGAGED),
se corrige a ATTENTIVE (`STRICT_ENGAGED_PROXIMITY=0` para desactivar).

## Configuración (variables de entorno)

| Variable | Default | Qué hace |
|---|---|---|
| `VIDEO_VLM_CHUNK_SEC` | `1` | Duración de cada trozo analizado por el VLM (segundos). |
| `MAX_VLM_PEOPLE` | `10` | Máximo de personas por trozo que se le mandan al VLM (prioriza las más cercanas a la cámara). |
| `VLM_PROFILE` | `balanced` | Resolución/tokens del VLM: `stream`\|`fast`\|`balanced`\|`quality`. |
| `SOCIAL_STATE_MODE` | `vlm` | `vlm` = el VLM decide el estado social; `map` = se deriva solo de la acción. |
| `STRICT_ENGAGED_PROXIMITY` | `1` | Solo ROS2: corrige ENGAGED sin corroboración de proximidad LiDAR a ATTENTIVE. |
| `ENGAGED_PROXIMITY_MAX_M` | `2.0` | Distancia real (metros) máxima para considerar dos personas "en el mismo grupo". |
| `HF_TOKEN` | — | Tu token de HuggingFace, si algún modelo lo requiere. |

## Rendimiento (medido)

Con 2-4 personas en cuadro, cada trozo de 1s del VLM tarda ~0.35-0.45s — corre
en tiempo real de sobra. Con 5+ personas, la latencia sube a ~0.6-0.8s por
trozo, al filo del presupuesto de 1 segundo. En cámara en vivo la latencia es
un poco mayor que en modo offline (~0.6-1.0s con 1-2 personas) porque la
captura y el VLM corren al mismo tiempo en la misma GPU — es un trade-off
deliberado para que la cámara no se congele mientras el VLM piensa.

## Limitaciones conocidas

- La marcha por piernas se detecta mejor de perfil que caminando derecho
  hacia/desde la cámara (el balanceo lateral es más sutil desde ese ángulo);
  se compensa con una señal de cambio de profundidad de la caja + histéresis,
  pero no es perfecto.
- Alguien de pie balanceándose lentamente (hablando, gesticulando) puede
  ocasionalmente leerse como movimiento si su vaivén dura más que la ventana
  de 1 segundo del trozo.
- "ATTENTIVE" se basa en si se ven ambos ojos con buena confianza (cara de
  frente a la cámara), no en dirección de mirada real.
