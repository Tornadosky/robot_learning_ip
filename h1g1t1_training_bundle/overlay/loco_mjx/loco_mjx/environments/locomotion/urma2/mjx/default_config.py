from ml_collections import config_dict


def get_config(environment_name):
    config = {
        "name": environment_name,
        "nr_envs": 4096,
        "nr_envs_mode": "per_device",
        "nr_eval_envs": 256,
        "seed": 1,
        "render": False,
        "keep_render_assets": False,
        "device": "gpu",
        "train_robots": ("unitree_a1",),  # Use ("all",) to train on every available robot.
        "train_robots_all_excluding": (),  # When using ("all",) for train_robots, this can be used to exclude certain robots from training
        "eval_robots": (),
        "fixed_observation_size": -1,
        "fixed_action_size": -1,
        "control_type": "pd",
        # Overnight 18-08 ablations (S3, S2): PD decision rate as a config key
        # (was a PDControl.__init__ default), and a uniform multiplier on the
        # per-actuator action scaling_factors (x4 ~ the loco-mujoco full-joint-
        # range action convention).
        "control_frequency_hz": 50,
        "scaling_factor_multiplier": 1.0,
        "command": {
            "type": "random_flyer_altitude_target",
            "sampling_type": "step_probability_and_reset",
            "sampling_probability": 0.002,
            "expected_tracked_frame_displacement_per_robot_size": 0.5,
            "max_velocity_per_m_factor": 2.0,
            "clip_max_velocity": 1.0,
            "zero_clip_threshold_percentage": 0.1,
            "all_zero_chance": 0.04,
            "single_zero_chance": 0.005,
            "actuator_joint_keep_nominal_flip_chance": 0.001,
            "flyer_altitude_target_min_ratio": 0.5,
            "flyer_altitude_target_max_ratio": 1.5,
            "flyer_altitude_velocity_time_constant_s": 1.0,
            "flyer_altitude_termination_margin_ratio": 0.05,
            "log_signed_z_tracking": True,
            # command.type="tracking" only. A joint-angle reference, so these
            # are morphology-invariant by construction and are NOT rescaled
            # when the body is randomized.
            "tracking_gait_period_s": 0.8,
            "tracking_gait_amplitude_rad": 0.35,
            "tracking_gait_period_jitter": 0.25,
            "tracking_amplitude_jitter": 0.25,
            # command.type="tracking_clip" only: a REAL retargeted motion instead
            # of the procedural gait. One offline retarget per family, because
            # H1's 19-vector and G1's 29-vector are different spaces -- the map
            # below is what keeps a family from being fed another family's angles.
            # Default is the LAFAN1 WALK clip, not the dance: 261 s at 40 Hz,
            # 178 m of path at 0.68 m/s with real leg swing (0.30 rad mean joint
            # std). The dance clip was measured untrackable at every coefficient
            # tried (30/100/300 all sat exactly at the ignore baseline), and a
            # dance is not dynamically consistent with balance the way a gait is.
            "tracking_clip_dir": "/mnt/c/Users/smirn/Desktop/robot_learning_ip/external_data/amass_converted/LAFAN1",
            "tracking_clip_file": "walk1_subject1.npz",
            # Keyed by BOTH the full robot name and the short name, because
            # robot_config resolves to the short form ("h1") for some robots and
            # the full form for others.
            "tracking_clip_robot_map": '{"unitree_h1": "UnitreeH1", "unitree_g1": "UnitreeG1", "h1": "UnitreeH1", "g1": "UnitreeG1", "booster_t1": "BoosterT1", "t1": "BoosterT1", "atlas": "Atlas", "talos": "Talos", "toddlerbot": "ToddlerBot", "unitree_h1v2": "UnitreeH1v2", "h1v2": "UnitreeH1v2"}',
            # 0 = take the clip's own `frequency` field (LAFAN1 walk is 40 Hz).
            "tracking_clip_fps": 0.0,
            # Short extracted gait clips can be looped smoothly. This enables
            # seam-aware interpolation and finite-difference velocities.
            "tracking_clip_cyclic": False,
            # Scale the clip about its own mean pose without changing its phase.
            # A separate root-velocity scale keeps the commanded translation
            # consistent when a retarget's swing amplitude is too large.
            "tracking_clip_amplitude_scale": 1.0,
            "tracking_clip_velocity_scale": 1.0,
            # Desired velocity disambiguates the two times a gait visits the
            # same pose while moving in opposite directions.
            "tracking_clip_observe_velocity": False,
            # FSQ z-conditioning (the tracking_latent experiment): publish the
            # per-joint quantized latent stream (<clip>_zq.npz next to the clip)
            # as extra per-joint observation channels. With
            # ..._replaces_reference the explicit reference-delta channel is
            # DROPPED, so the token stream is the policy's only pose-level view
            # of the motion (pair with REFBIAS=0 or the action path leaks the
            # reference anyway). All three keys are gated so existing
            # checkpoints keep their observation layout.
            "tracking_clip_latent_obs": False,
            "tracking_clip_latent_dim": 32,
            "tracking_clip_latent_replaces_reference": True,
            # Token RATE. The stream is one code per clip frame, and the
            # tokenizer's own report shows 88% of frame-joint slots carry a
            # UNIQUE code (756746 / 857375 on super5dance/H1) -- that is a
            # continuous latent, not a vocabulary. Holding the code for K frames
            # turns the interface into a genuine low-rate motion code (K=5 at
            # 40 fps = 8 tokens/s) without retraining the tokenizer, and asks
            # the only question that matters for the token claim: how much of
            # the stream does the policy actually need? 1 = every frame (old
            # behaviour, bit-identical).
            "tracking_clip_latent_hold": 1,
            "tracking_clip_reference_hold": 1,
            "tracking_clip_root_height_from_pose": False,
            # FEETFIX. With ..._from_pose alone the reference root is placed so
            # the lowest reference foot touches z=0 EVERY frame, which deletes
            # the clip's vertical root motion (a hop becomes a floor-skim). With
            # this on, the pose-derived height acts as a FLOOR under the clip's
            # own root height instead of replacing it: grounded where the clip
            # is grounded, airborne where the clip is airborne, never
            # underground. Requires ..._from_pose=True; False = old behaviour.
            "tracking_clip_root_height_pose_as_floor": False,
            # ROOTFIX step 4. Publish the reference HEADING DELTA as a general
            # observation channel. Steps 1-3 put the clip's root motion into the
            # reference and the reward; without this the policy is scored on a
            # quantity it cannot see, and the resulting null would be misread as
            # "root tracking does not help". Off by default: it changes the
            # observation size and so invalidates old checkpoints.
            "tracking_clip_observe_root_heading": False,
            # Center position-actuator actions on the reference pose, turning
            # the learned action into a feedback residual. Zero preserves the
            # standard nominal-centered controller and old checkpoints.
            "tracking_reference_action_bias": 0.0,
            # Derive the velocity command from the clip's own root motion instead
            # of sampling it at random. The env curriculum is a feedback
            # regulator keyed on velocity-command error, so with a random command
            # "reproduce this walk" and "travel at this sampled velocity" are
            # competing objectives and a policy that tracks is scored as failing.
            # Measured 2026-08-05: ratio 0.229 with curriculum stuck at 0.0016,
            # and a 6x coefficient cut moved the curriculum only 1.75x.
            # Off by default -- it changes the task definition.
            "tracking_clip_command_cap_scale": 1.0,
            # DANCEPROOF Check 30. Multiplies max_command_velocity for
            # CLIP-DERIVED commands only. The cap keeps a randomly SAMPLED
            # command reachable; a command measured off a real motion is
            # reachable by construction. Measured on dance2_subject1: a cap
            # of 1.0 rad/s against a yaw rate whose MEDIAN is 1.029 and
            # whose p95 is 7.641, saturating 36.27% of frames. 1.0 = off.
            "tracking_clip_velocity_command": False,
            # M1 -- fit the JOINT reference to the CURRENT body's limits, online.
            #
            # `_fit_offsets_to_limits` fits the clip ONCE, offline, against the
            # NOMINAL joint limits, so the fit degrades as bodies randomize away
            # from nominal. Measured 2026-08-05 on H1, worst body of 48:
            # curriculum 0.00 -> 0.19% of frames clipped, 0.10 -> 8.8%,
            # 0.30 -> 17.0%, 1.00 -> 88.3%. At the curriculum any serious
            # randomization experiment needs, ~17% of frames on the worst body
            # are silently rewritten by the limit clamp, so the reference is a
            # DIFFERENT motion per body and every cross-morphology number is
            # confounded.
            #
            # With this on, the same translate-and-only-if-forced-scale fit is
            # recomputed per environment from that body's own
            # internal_state["joint_position_limits"], exactly as the FK site
            # targets already are. Off by default so the joint-only results
            # measured against the offline fit stay reproducible.
            "tracking_clip_fit_per_variant": False,
            # What the clip offsets are anchored to. "centered" (historical):
            # each joint's series minus its own clip mean, which re-centers the
            # motion about the home pose -- measured 2026-08-22 this pushes
            # 10/19 H1 joints out of their limits and the limit fit then shrinks
            # them on EVERY frame (ankles x0.74, shoulders x0.43; targets 0.318
            # rad from the true dance). "absolute": series minus the actuator's
            # nominal angle, so nominal + offset IS the sign-corrected clip
            # angle (0.009 rad from raw on H1, exact on G1); all offline/online
            # rescaling is skipped and the per-step limit clip handles the rare
            # peaks. Morphology randomization never mutates joint limits, so
            # absolute targets stay legal on every random body. Requires
            # tracking_clip_fit_per_variant=False. Default preserves every
            # existing checkpoint.
            "tracking_clip_anchor": "centered",
        },
        "env_curriculum_nr_levels": 100,
        # N2 -- a CEILING on the velocity-error regulator, separate from the
        # morphology magnitude.
        #
        # env_curriculum_coeff scales observation noise, perturbation strength,
        # action delay and actuator dropout as well as the 25 gait reward terms.
        # Measured 2026-08-05: at curriculum 0.84 unitree_g1's tracking error
        # went intermittently NaN (16 of 640 logged points, from step 40.2M) and
        # its finite values degraded to ratio 1.61 -- i.e. WORSE than ignoring
        # the reference. That is a ceiling on the *perturbation* schedule, and it
        # is not the same quantity as the body-randomization level, which is why
        # capping it here rather than lowering morphology_coeff is the right fix.
        #
        # 1.0 reproduces the old behaviour exactly.
        "env_curriculum_coeff_max": 1.0,
        # M2 -- let the curriculum criterion SEE tracking. It advances on
        # velocity-command error alone (environment.py:_reset), so a policy that
        # reproduces the reference exactly is still graded as failing whenever
        # the randomly sampled command disagrees with the clip. An episode whose
        # mean tracking error is below this RATIO of the clip's own ignore
        # baseline counts as a success too. 0.0 = off (velocity-only).
        "env_curriculum_level_success_tracking_ratio": 0.0,
        "env_curriculum_level_success_normalized_velocity_diff": 0.4,
        "env_curriculum_level_success_episode_length": 500,
        "env_curriculum_worst_active_axis_weight_start": 0.0,
        "domain_randomization": {
            "sampling_type": "step_probability_and_reset",
            "sampling_probability": 0.002,
            "action_delay": {
                "type": "default",
                "min_delay_s": 0.0,
                "max_delay_s": 0.02,
            },
            "initial_state": {
                "type": "random",
                "roll_angle_pi_factor": 0.05,
                "pitch_angle_pi_factor": 0.05,
                "yaw_angle_pi_factor": 1.0,
                "actuator_joint_position_offset_to_nominal": 0.01,
                "actuator_joint_nominal_position_factor": 0.5,
                "joint_velocity_max_factor": 0.5,
                "trunk_velocity_clip_mass_factor": 0.1,
                "trunk_velocity_clip_limit": 0.5,
            },
            "joint_dropout": {
                "type": "default",
                "dropout_open_chance": 0.001,
                "dropout_lock_chance": 0.001,
            },
            "mujoco_model": {
                "type": "default",
                "friction_tangential_factor": 1.0,
                "friction_torsional_factor": 1.0,
                "friction_rolling_factor": 1.0,
                "timeconst_log_range": 1.8,
                "dampratio_factor": 0.6,
                "foot_solimp_factor": 0.8,
                "add_impratio": 1.0,
                "xy_gravity": 0.5,
                "z_gravity_factor": 0.1,
                "density_factor": 0.1,
                "viscosity_factor": 0.1,
            },
            "observation_noise": {
                "type": "default",
                "joint_position": 0.03,
                "joint_velocity": 1.5,
                "linear_velocity_estimate": 0.2,
                "imu_angular_velocity": 0.2,
                "gravity_vector": 0.05,
                "exteroception": 0.03,
            },
            "perturbation": {
                "sampling_type": "step_probability",
                "sampling_probability": 0.002,
                "type": "default",
                "trunk_velocity_clip_mass_factor": 0.1,
                "trunk_velocity_clip_limit": 1.0,
                "trunk_velocity_add_chance": 0.5,
                "max_joint_velocity": 0.5,
                "max_joint_position": 0.01,
            },
            "seen_robot": {
                "type": "default",
                # M0 -- split the overloaded env_curriculum_coeff.
                #
                # One scalar used to control three unrelated things: how
                # randomized the BODY is (this sampler), how hard the 25
                # gait/regularization reward terms are scaled, and whether the
                # tracking term is gated. It was regulated by a feedback
                # criterion that measures only VELOCITY-COMMAND error, so every
                # tracking arm measured on 2026-08-04/05 sat at curriculum
                # 0.014-0.047, i.e. bodies that varied +-2%: "tracking fights
                # gait" and "bodies stay nominal" were the same bug.
                #
                # This coefficient controls ONLY the seen-robot sampler, i.e.
                # the morphology. "curriculum" reproduces the old behaviour
                # exactly (morphology_coeff == env_curriculum_coeff), so nothing
                # changes unless asked; "fixed" pins it at morphology_coeff_value
                # regardless of what the velocity-error regulator is doing.
                # N2 -- "schedule" ramps morphology_coeff_start ->
                # morphology_coeff_value linearly over morphology_coeff_ramp_steps
                # GLOBAL environment steps (the same unit as
                # algorithm.total_timesteps), then holds at the value.
                #
                # Why a ramp: a fixed high coefficient makes the earliest, most
                # fragile part of training needlessly hard, and the curriculum
                # regulator cannot compensate because M0 split it away from the
                # body. Measured 2026-08-05: morphology 0.7 reaches the same
                # gates as 0.4 but takes 26.2M steps instead of 15.7M -- the cost
                # is paid entirely up front, which is exactly what a ramp removes.
                "morphology_coeff_mode": "curriculum",
                "morphology_coeff_value": 0.0,
                "morphology_coeff_start": 0.0,
                "morphology_coeff_ramp_steps": 0,
                # M4b -- evaluate on a FIXED, NAMED held-out body from
                # generated_variants/ (extreme_tall_light, short_legs, big_feet,
                # ...). Empty = off. See _load_named_variant in
                # seen_robot_functions/default.py for why this is a per-body
                # ratio applied inside the sampler rather than a model swap.
                "named_variant_name": "",
                "named_variant_family": "",
                "named_variant_json": "",
                # DANCEPROOF -- train on a FIXED, FINITE set of randomized
                # bodies instead of a fresh one every reset. 0 = off (the
                # normal continuous distribution). N > 0 quantizes the sampler
                # key to one of N keys, so the run sees exactly N bodies and
                # sees each of them many times.
                #
                # This is a quantization of the key rather than a pool of
                # models because EVERY morphology in `sample()` below is a
                # deterministic function of that one key -- same key, same
                # body, down to the actuator gains. A pool of pre-built models
                # would need its own code path for the FK-derived quantities
                # (nominal standing height, foot tilt, collision counts) that
                # the tracking reward reads; this reuses the existing one.
                "body_pool_size": 0,
                "body_pool_seed": 0,
                "exact_inertia_rescale": False,
                # 1.0 = upstream. 4.0 = physically consistent (mass s^3 x lever s^1).
                "torque_scaling_exponent": 1.0,
                "robot_size_scaling_factor": 0.5,
                "coupled_mass_inertia_factor": 0.25,
                "decoupled_mass_inertia_factor": 0.25,
                "add_com_displacement": 0.01,
                "add_inertia_orientation_rad": 0.2,
                "add_body_position": 0.001,
                "add_body_orientation_rad": 0.01,
                "add_imu_position": 0.1,
                "foot_size_factor": 0.75,
                "joint_axis_angle_rad": 0.1,
                "torque_limit_factor": 0.75,
                "add_actuator_joint_nominal_position": 0.1,
                "joint_velocity_max_factor": 0.75,
                "add_joint_range": 0.4,
                "soft_joint_position_limit": 0.9,
                "joint_damping_factor": 1.0,
                "add_joint_damping": 0.1,
                "joint_armature_factor": 1.0,
                "add_joint_armature": 0.001,
                "joint_stiffness_factor": 0.1,
                "add_joint_stiffness": 0.1,
                "joint_friction_loss_factor": 1.0,
                "add_joint_friction_loss": 0.00001,
                "p_gain_factor": 0.5,
                "d_gain_factor": 0.5,
                "force_gain_factor": 0.5,
                "scaling_factor_factor": 0.5,
            },
            "unseen_robot": {
                "type": "default",
                "mass_inertia_factor": 0.3,
                "com_factor": 0.15,
                "body_position_factor": 0.01,
                "joint_damping_factor": 0.5,
                "joint_armature_factor": 0.5,
                "joint_stiffness_factor": 0.5,
                "joint_friction_loss_factor": 0.5,
                "p_gain_factor": 0.2,
                "d_gain_factor": 0.2,
                "force_gain_factor": 0.2,
                "position_offset": 0.02,
            },
        },
        "policy_exteroceptive_observation_type": "none",
        "critic_exteroceptive_observation_type": "height_over_ground",
        "reward": {
            "type": "default",
            # L1-A (RealWalk, 2026-08-06): a FLOOR under the 25 gait/regularization
            # terms, split out of env_curriculum_coeff exactly as M0 split
            # morphology out of it.
            #
            # Why this exists. Every gait term in DefaultReward is multiplied by
            # env_curriculum_coeff, and that coefficient is not a schedule -- it
            # is a feedback regulator that only rises when the mean normalized
            # velocity error drops below 0.4. Under a tracking reward at coeff 30
            # against a velocity-command reward at coeff 2, the policy rationally
            # ignores the velocity command, the error never falls, and the
            # regulator stays pinned. Measured over the 200M-step GaitProof run:
            # it ended at 0.0012, which put foot_slip at an effective 0.00012,
            # ground_penetration at 0.012 and foot_air_time at 0.0036 -- i.e. off.
            # The resulting policy jiggled, skated and clipped the floor, which is
            # exactly what that reward asked for.
            #
            # "curriculum" is the identity and the default, so an unchanged config
            # is bit-for-bit the old behaviour.
            #   curriculum  gait_coeff == env_curriculum_coeff  (pre-L1-A)
            #   fixed       gait_coeff == gait_coeff_value, curriculum ignored
            #   floor       gait_coeff == max(env_curriculum_coeff, gait_coeff_value)
            #
            # "floor" is the intended setting: gait shaping can never be switched
            # off by a stuck regulator, but a curriculum that DOES rise still
            # takes the terms above the floor.
            "gait_coeff_mode": "curriculum",
            "gait_coeff_value": 0.0,
            "tracking_xy_velocity_command_coeff": 2.0,
            "tracking_xy_temperature": 0.25,
            "tracking_yaw_velocity_command_coeff": 1.0,
            "tracking_yaw_temperature": 0.25,
            "tracking_z_velocity_command_coeff": 1.0,
            "tracking_z_temperature": 0.25,
            "tracked_frame_command_backtracking_coeff": 1.0,
            "alive_clipped_coeff": 0.05,
            "alive_unclipped_coeff": 0.05,
            "imu_acceleration_coeff": 1e-4,
            "roll_pitch_vel_coeff": 0.05,
            "roll_pitch_pos_coeff": 10.0,
            "actuator_joint_nominal_diff_coeff": 100.0,
            # "nominal" | "reference". What the keep-still regularizer pulls
            # toward. Set to "reference" for tracking runs, where pulling to the
            # nominal pose means rewarding the ignore-the-reference baseline.
            "nominal_diff_target": "nominal",
            "joint_position_limit_coeff": 40.0,
            "actuator_joint_velocity_limit_coeff": 5.0,
            "soft_actuator_joint_velocity_limit": 0.9,
            "joint_velocity_coeff": 4e-4,
            "joint_acceleration_coeff": 5e-6,
            "joint_torque_frac_coeff": 0.03,
            "power_draw_frac_penalty_coeff": 0.03,
            "action_rate_coeff": 3.0,
            "action_smoothness_coeff": 0.1,
            "collision_coeff": 2.0,
            "ground_penetration_coeff": 10.0,
            "base_height_coeff": 30.0,
            "foot_air_time_coeff": 3.0,
            "foot_air_time_per_robot_size_m": 0.4,
            "symmetry_air_coeff": 1.0,
            "feet_lateral_min_distance_ratio": 0.75,
            "feet_lateral_min_distance_coeff": 2.0,
            "foot_slip_coeff": 0.1,
            "foot_z_velocity_coeff": 0.2,
            "foot_flat_contact_coeff": 0.01,
            "feet_orientation_coeff": 1.0,
            # reward.type="tracking" only. Errors are normalized per actuator
            # (and, for height, by body size) so H1 and G1 -- and a 0.5x and a
            # 1.5x body -- are held to the same standard in a shared policy.
            "joint_tracking_coeff": 3.0,
            "joint_tracking_temperature": 0.25,
            # Scale the tracking term by env_curriculum_coeff, exactly as the 25
            # gait/regularization terms in DefaultReward already are. See the
            # long comment in reward_functions/tracking.py: ungated, a large
            # tracking coefficient starves the velocity command, freezes the
            # curriculum, and thereby switches OFF every reward that shapes how
            # the robot steps -- producing a shuffling in-place gait.
            "tracking_curriculum_gated": False,
            "root_height_tracking_coeff": 1.0,
            "root_height_tracking_temperature": 0.01,
            # DeepMimic term structure, ported from loco-mujoco's MimicReward
            # (loco_mujoco/core/reward/trajectory_based.py:174-183). Off by
            # default so the existing joint-only results stay reproducible.
            #
            # loco-mujoco mixes with w_sum {qpos 0.4, qvel 0.2, rpos 0.5,
            # rquat 0.3} and sharpens with a SEPARATE w_exp {qpos 10, qvel 2,
            # rpos 100, rquat 10}; urma2's form is coeff*exp(-dist/temperature),
            # so temperature = 1/w_exp. The weights below are ratios against
            # joint_tracking_coeff, which is the only coefficient on this stack
            # that has been measured, so anchoring to it keeps that calibration.
            # ROOTFIX step 2. An explicit HEADING term against the clip's own
            # root yaw. Heading is an angle, so unlike every other reference
            # quantity on this stack it is body-independent and needs no
            # per-body fit -- a 0.5x and a 1.5x H1 must turn through the same
            # number of degrees. Ratio-anchored to joint_tracking_coeff for the
            # same reason the DeepMimic terms are.
            #
            # 0.0 = off = current behaviour. deepmimic_heading_free removes the
            # heading anchor that was smuggled into the POSE terms (Check 28);
            # this is where heading is supposed to be scored instead, and until
            # this term is on, nothing in the reward asks the robot to turn.
            "root_heading_tracking_weight_ratio": 0.0,
            # exp(-err^2 / T) with err in radians. 0.25 puts a 30 deg heading
            # error at exp(-0.27/0.25) = 0.34 of the term, and 90 deg at 0.008.
            "root_heading_tracking_temperature": 0.25,
            # Kernel used by the explicit heading term. ``exponential`` preserves
            # historical checkpoints/configurations. ``cosine`` keeps a useful
            # recovery gradient over the full wrapped [-pi, pi] range and is used
            # by the H1+G1+T1 probe.
            "root_heading_tracking_kernel": "exponential",
            "deepmimic_heading_free": True,
            # DANCEPROOF Check 28. The DeepMimic body terms compare WORLD-frame
            # quantities against a reference built at the NOMINAL heading, so a
            # robot whose joints match the reference exactly is still charged
            # pose error purely for facing another way. Measured on a robot
            # placed in exactly the reference joint configuration and rotated:
            #     yaw   0 deg -> rpos 0.000000  rquat 0.000000
            #     yaw  90 deg -> rpos 0.038388  rquat 0.444444
            #     yaw 180 deg -> rpos 0.076775  rquat 0.888889
            # True evaluates the reference at the robot's own yaw, so the terms
            # measure only what the JOINTS did. False restores the old anchored
            # behaviour, so the two can be compared at matched budget.
            "deepmimic_enabled": False,
            "deepmimic_site_target_mode": "fk",   # "fk" | "scaled"
            "deepmimic_reference_adapt_rate": 0.0,   # ReActor-style; 0 = off
            "deepmimic_reference_adapt_bound": 0.1,  # fraction of body size
            "deepmimic_qvel_weight_ratio": 0.5,     # 0.2 / 0.4
            "deepmimic_rpos_weight_ratio": 1.25,    # 0.5 / 0.4
            "deepmimic_rquat_weight_ratio": 0.75,   # 0.3 / 0.4
            # The qvel error is normalized by the REFERENCE's own mean square
            # velocity (see reward_functions/tracking.py), so it is
            # dimensionless: 1.0 means "off by as much as the motion moves".
            # loco-mujoco's 1/w_exp = 0.5 is kept, but it now acts on a
            # dimensionless quantity rather than on raw rad^2/s^2, which is what
            # made the term dead on H1.
            "deepmimic_qvel_temperature": 0.5,      # 1 / 2.0
            "deepmimic_rpos_temperature": 0.01,     # 1 / 100.0
            "deepmimic_rquat_temperature": 0.1,     # 1 / 10.0
            # FEETFIX. The one thing no term on this stack has ever scored: how
            # high the FEET are above the floor. rpos is root-relative and
            # body-size normalized, so a foot 4 cm under the floor and a foot
            # 4 cm above it are worth exactly the same reward -- which is why
            # trained policies float and clip through the ground while their
            # joint tracking looks fine.
            #
            # The target is the reference pose's own foot bottoms, computed by
            # FK on the CURRENT randomized model and placed at the reference
            # root height, i.e. the same construction as every other DeepMimic
            # term here: morphology-correct and topology-agnostic, no clip-side
            # foot correspondence required. Ratio-anchored to the joint term so
            # "0.5" means the same thing it means for qvel/rpos/rquat.
            # 0.0 = off, and every existing arm is bit-identical with it off.
            "deepmimic_foot_height_weight_ratio": 0.0,
            # Error is (height difference / body size)^2 averaged over feet.
            # 0.01 == rpos's temperature: a 10% -of-body-size foot height error
            # costs the term ~63% of its value.
            "deepmimic_foot_height_temperature": 0.01,
            # DefaultReward clips its regularized locomotion component before
            # TrackingReward adds positive imitation bonuses. Opting in reapplies
            # contact terms after those bonuses, so physical contact can matter.
            "tracking_post_contact_penalties": False,
            "log_info": False,
        },
        "termination": {
            "type": "below_height",
            "height_percentage_threshold": 0.8,
            # M3 -- DeepMimic's other core trick: end the episode once the robot
            # has drifted too far from the reference, so samples are not spent in
            # states the reference never visits. A RATIO of the clip's own ignore
            # baseline (H1 0.27551, G1 0.12018 rad^2), so one number means the
            # same thing on both families. 0.0 = off.
            "tracking_deviation_ratio": 0.0,
        },
        "terrain": {
            "type": "hfield_diverse",
            "wave_fn_min": 0,
            "wave_fn_max": 2,
            "wave_height_max_per_m_factor": 0.3,
            "random_height_max_per_m_factor": 0.04,
            "block_probability": 0.5,
            "block_length_in_meters": 0.5,
            "block_height_max_per_m_factor": 0.1,
            # RealWalk R-contact: > 0.0 stiffens the foot/floor contact by setting
            # solref timeconst to this value (clamped to 2*timestep) and solimp
            # [dmin, dmax] to [0.95, 0.99] on the floor and foot geoms. 0.0 keeps
            # the XML values, bit-identical to every run before this flag existed.
            # Measured basis: under solref [0.02, 1] quiet standing equilibrates
            # ~5 mm deep, so the < 5 mm penetration gate is unattainable and fast
            # loaded steps reach 80-150 mm (l2p_contact_probe / l2p2_pen_dynamics).
            "contact_solref_timeconst": 0.0,
        },
        "add_goal_arrow": False,
        "timestep": 0.005,
        "episode_length_in_seconds": 20,
    }

    return config_dict.ConfigDict(config)
