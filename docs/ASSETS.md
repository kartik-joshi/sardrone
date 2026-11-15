# Third-party assets

Nothing in this list is committed to the repository. `make assets` downloads
the models into `assets/fuel/` (git-ignored), and the world generator makes
recoloured copies in `assets/variants/` (also git-ignored).

## 3D models from Gazebo Fuel

Licences and sizes are as reported by the Fuel API
(`https://fuel.gazebosim.org/1.0/models?q=...`) on 2026-11-14. All are owned
by `OpenRobotics` on Fuel.

| Model on Fuel | Used for | Licence | Download |
| --- | --- | --- | --- |
| Standing person | person, standing | CC0 1.0 | 26.1 MB |
| Rescue Randy Sitting | person, sitting | CC0 1.0 | 3.0 MB |
| Rescue Randy | person, lying (the standing mesh laid flat) | CC BY 4.0 | 11.0 MB |
| Oak tree | trees | CC0 1.0 | 0.6 MB |
| Pine Tree | trees | CC0 1.0 | 0.9 MB |
| Backpack | not placed yet (a later detection class) | CC0 1.0 | 1.8 MB |

**Attribution (CC BY 4.0):** "Rescue Randy" by Open Robotics, from Gazebo
Fuel, https://fuel.gazebosim.org/1.0/OpenRobotics/models/Rescue%20Randy.
This project recolours the jacket in its texture.

### How the models are changed

- Only the mesh and texture files are used. The models' own `model.sdf`
  files are ignored: they use material scripts that Gazebo Harmonic does not
  support, and "Rescue Randy Sitting" would download a further model.
- Clothing colour: the T-shirt texture of "Standing person" is tinted, and
  the yellow-green jacket in the Rescue Randy textures is replaced with
  another colour. Skin, trousers and boots are left alone.
- "Rescue Randy" is rotated 90 degrees to lie on the ground.

## From PX4-Autopilot (BSD 3-Clause)

The drone is PX4's `x500` model with PX4's `mono_cam` camera. Both are
referenced from the PX4 checkout, not copied; `x500_sar_cam` (generated per
world) only adds the camera's pitched mounting pose.

## Made by this project

The ground texture is generated from seeded random noise by
`sardrone.worlds.assets.ground_texture`. No external image is used.
