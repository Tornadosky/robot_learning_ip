import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from loco_mjx.environments.locomotion.urma2.mjx.reward_functions.default import DefaultReward


class TrackingReward(DefaultReward):
    """DefaultReward plus a joint-space reference-tracking term.

    Deliberately additive rather than a replacement. The default reward carries
    all the regularization that keeps a randomized body physically sane (torque
    and velocity limits, collisions, ground penetration, foot behaviour), and
    -- importantly -- the velocity-tracking term the env curriculum reads to
    decide when to ramp morphology randomization in. Dropping it would freeze
    the curriculum at zero and the policy would never see a randomized body.

    The tracking term is normalized per actuator, so H1 (19) and G1 (29) get
    comparable reward magnitudes rather than the higher-DOF robot dominating
    the shared policy's gradient.
    """

    def __init__(self, env):
        super().__init__(env)

        reward_config = env.env_config["reward"]
        self.joint_tracking_coeff = reward_config["joint_tracking_coeff"] * env.dt
        self.joint_tracking_temperature = reward_config["joint_tracking_temperature"]
        self.root_height_tracking_coeff = reward_config["root_height_tracking_coeff"] * env.dt
        self.root_height_tracking_temperature = reward_config["root_height_tracking_temperature"]
        self.tracking_curriculum_gated = reward_config["tracking_curriculum_gated"]
        self.root_height_from_reference_pose = bool(
            env.env_config["command"].get("tracking_clip_root_height_from_pose", False)
        )
        self.root_height_pose_as_floor = bool(
            env.env_config["command"].get("tracking_clip_root_height_pose_as_floor", False)
        )
        self.post_contact_penalties = bool(
            reward_config.get("tracking_post_contact_penalties", False)
        )

        # ---- DeepMimic term structure -------------------------------------
        # loco-mujoco's MimicReward is sum_i w_sum[i] * exp(-w_exp[i] * dist_i)
        # with w_sum = {qpos 0.4, qvel 0.2, rpos 0.5, rquat 0.3} and a SEPARATE
        # w_exp = {qpos 10.0, qvel 2.0, rpos 100.0, rquat 10.0}. urma2 writes the
        # same shape as coeff * exp(-dist / temperature), so temperature = 1/w_exp.
        #
        # The new terms are expressed as RATIOS of the existing joint term rather
        # than as absolute loco-mujoco weights. The joint term's coefficient is
        # the one quantity here that has actually been measured on this stack
        # (the coeff 3/30/100/300 Pareto), so anchoring to it preserves that
        # calibration and adds the others in loco-mujoco's proportion:
        #   qvel  0.2/0.4 = 0.50      rpos 0.5/0.4 = 1.25      rquat 0.3/0.4 = 0.75
        # ROOTFIX step 2. Ratio-anchored to the joint term and multiplied by dt
        # exactly like root_height_tracking_coeff above, so "0.75" means the
        # same thing here as the DeepMimic ratios mean and the term does not
        # silently scale with the control rate.
        self.root_heading_tracking_coeff = (
            reward_config["root_heading_tracking_weight_ratio"]
            * reward_config["joint_tracking_coeff"] * env.dt
        )
        self.root_heading_tracking_temperature = reward_config[
            "root_heading_tracking_temperature"]
        self.root_heading_tracking_kernel = str(
            reward_config.get("root_heading_tracking_kernel", "exponential")
        )
        if self.root_heading_tracking_kernel not in ("exponential", "cosine"):
            raise ValueError(
                "root_heading_tracking_kernel must be 'exponential' or 'cosine', "
                f"got {self.root_heading_tracking_kernel!r}"
            )
        self.deepmimic_enabled = reward_config["deepmimic_enabled"]
        self.deepmimic_heading_free = reward_config["deepmimic_heading_free"]
        # "fk" (default, unchanged behaviour) or "scaled" -- see the block that
        # builds `reference_relative` for what each one means.
        # ReActor-style reference adaptation (arm C). See the block that applies
        # it for the mechanism and for what stops it degenerating.
        self.reference_adapt_rate = float(
            reward_config.get("deepmimic_reference_adapt_rate", 0.0))
        self.reference_adapt_bound = float(
            reward_config.get("deepmimic_reference_adapt_bound", 0.1))
        self.site_target_mode = reward_config.get("deepmimic_site_target_mode", "fk")
        if self.site_target_mode not in ("fk", "scaled"):
            raise ValueError(
                f"deepmimic_site_target_mode must be 'fk' or 'scaled', "
                f"got {self.site_target_mode!r}")
        self.qvel_weight_ratio = reward_config["deepmimic_qvel_weight_ratio"]
        self.rpos_weight_ratio = reward_config["deepmimic_rpos_weight_ratio"]
        self.rquat_weight_ratio = reward_config["deepmimic_rquat_weight_ratio"]
        self.qvel_temperature = reward_config["deepmimic_qvel_temperature"]
        self.rpos_temperature = reward_config["deepmimic_rpos_temperature"]
        self.rquat_temperature = reward_config["deepmimic_rquat_temperature"]
        # FEETFIX. See default_config for why a foot HEIGHT term is not already
        # covered by rpos. Independent of deepmimic_enabled so it can be run as
        # a single delta, but it needs the same reference FK, so the block that
        # builds `reference_data` below is gated on this too.
        self.foot_height_weight_ratio = float(
            reward_config.get("deepmimic_foot_height_weight_ratio", 0.0))
        self.foot_height_temperature = float(
            reward_config.get("deepmimic_foot_height_temperature", 0.01))
        self.foot_height_active = (
            self.foot_height_weight_ratio > 0.0 and env.nr_feet > 0 and env.has_free_base
        )
        self.foot_height_coeff = (
            self.foot_height_weight_ratio * reward_config["joint_tracking_coeff"] * env.dt
        )
        # Diagnostics need the reference's own feet whenever there IS a
        # reference, term or no term -- the same rule the heading error follows,
        # and for the same reason: it makes "was the REFERENCE underground?"
        # answerable from the control arms instead of costing a slot.
        # Restricted to arms that already build `reference_data` for another
        # reason, so no arm pays for an extra mjx.kinematics it did not before.
        self.foot_reference_metrics = (
            env.nr_feet > 0 and env.has_free_base
            and (self.deepmimic_enabled or self.root_height_from_reference_pose
                 or self.foot_height_active)
        )

        # Every body except world. Deliberately NOT the clip's body list: the
        # site targets are computed by forward kinematics from the reference
        # ANGLES on the current randomized model, so the robot is compared
        # against its own FK reference and no name correspondence with the clip
        # is required. That is what keeps this term topology-agnostic across
        # H1's 21 bodies and G1's 31.
        self.tracked_body_ids = jnp.array(np.arange(1, env.initial_mjx_model.nbody))


    def init(self, internal_state):
        super().init(internal_state)
        # ReActor-style adapted reference (arm C). One bounded position offset
        # per tracked body, in the trunk frame. Zero unless the adapt rate is
        # positive, so every other arm is bit-identical.
        if self.reference_adapt_rate > 0.0:
            internal_state["reference_site_offset"] = jnp.zeros(
                (self.tracked_body_ids.shape[0], 3))

    def reward_and_info(self, data, mjx_model, internal_state, action, info, feet_floor_contacts):
        reward = super().reward_and_info(
            data, mjx_model, internal_state, action, info, feet_floor_contacts=feet_floor_contacts
        )

        joint_positions = data.qpos[self.env.actuator_joint_mask_qpos]
        joint_errors = joint_positions - internal_state["reference_joint_targets"]
        joint_errors = joint_errors * self.env.actuator_is_joint_transmission
        trackable_count = jnp.maximum(jnp.sum(self.env.actuator_is_joint_transmission), 1.0)
        mean_squared_joint_error = jnp.sum(jnp.square(joint_errors)) / trackable_count
        # Published for M3's early termination, which needs the same error the
        # reward is computed from. environment.py evaluates the reward (line 872)
        # before the termination check (line 891), so this is always the CURRENT
        # step's error, not the previous one's.
        internal_state["joint_tracking_error"] = mean_squared_joint_error
        # Optionally scale the tracking term by the SAME curriculum coefficient
        # that already scales the 25 gait/regularization terms in DefaultReward
        # (foot air time, foot slip, symmetry, torque, action smoothness, ...).
        #
        # Leaving it ungated created a self-reinforcing failure: a large tracking
        # coefficient pulled the policy off the velocity command -> the curriculum
        # criterion stopped being met -> the curriculum coefficient collapsed to
        # ~0.005 -> every gait-shaping reward was multiplied by ~0 -> the policy
        # had no incentive left to lift its feet and learned to shuffle in place,
        # which kept the velocity error high and the curriculum pinned. Measured
        # 2026-08-04: coeff 100 gave curriculum 0.005 and foot_air_time reward
        # -0.0000, versus 0.409 / -0.0070 for the stock reward.
        #
        # Gating inverts the ordering: at curriculum 0 the policy learns to WALK
        # with gait rewards at full relative weight and no tracking distraction;
        # as the curriculum rises, morphology randomization and tracking fade in
        # together. Tracking becomes a bonus on competent locomotion rather than
        # a competitor to it.
        tracking_gate = internal_state["env_curriculum_coeff"] if self.tracking_curriculum_gated else 1.0
        joint_tracking_reward = tracking_gate * self.joint_tracking_coeff * jnp.exp(
            -mean_squared_joint_error / self.joint_tracking_temperature
        )

        reference_data = None
        reference_qpos = None
        if (self.root_height_from_reference_pose or self.deepmimic_enabled
                or self.foot_height_active or self.foot_reference_metrics):
            reference_qpos = self.env.initial_qpos.at[
                self.env.joint_actuator_qpos_indices
            ].set(
                internal_state["reference_joint_targets"][self.env.joint_actuator_indices]
            )
            # DANCEPROOF Check 28: the reference pose must be evaluated at the
            # ROBOT'S OWN root orientation, not at initial_qpos's.
            #
            # Both body terms below compare world-frame quantities: rquat reads
            # xmat directly, and rpos subtracts the trunk POSITION without
            # rotating into the trunk FRAME. So if the reference is built at the
            # nominal heading while the robot has yawed, both report pose error
            # for a robot whose joints match the reference exactly. Measured, on
            # a robot placed in precisely the reference joint configuration and
            # then rotated about z:
            #     yaw    0 deg -> rpos 0.000000   rquat 0.000000
            #     yaw   90 deg -> rpos 0.038388   rquat 0.444444
            #     yaw  180 deg -> rpos 0.076775   rquat 0.888889
            # That is a HEADING ANCHOR wired into the pose reward. It is nearly
            # free on a straight walk (walk_cycle turns 8.5 deg in total) and it
            # is ruinous on dance2_subject1, which turns 2262 deg -- every
            # degree charged as pose error.
            #
            # Copying the live root makes both terms measure only what the
            # JOINTS did, which is what a pose term is for. Heading is a
            # separate quantity and belongs in its own explicit term against the
            # clip's own root yaw, not smuggled into the limb comparison.
            # YAW ONLY, deliberately. Copying the whole root would break two
            # things below: reference_root_height reads reference_qpos[2] and
            # would then track the robot's own height (a self-fulfilling
            # target), and copying roll/pitch would stop the reference demanding
            # an upright torso, which IS genuine pose information. Yaw is the
            # only component that carries no pose meaning while dominating both
            # world-frame comparisons.
            if self.env.has_free_base and self.deepmimic_heading_free:
                w, x, y, z = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
                yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                reference_qpos = reference_qpos.at[3:7].set(
                    jnp.array([jnp.cos(0.5 * yaw), 0.0, 0.0, jnp.sin(0.5 * yaw)])
                )
            reference_data = self.env.mjx_data.replace(
                qpos=reference_qpos,
                qvel=jnp.zeros(self.env.initial_mjx_model.nv),
                ctrl=jnp.zeros(self.env.nr_actuators),
            )
            reference_data = mjx.kinematics(mjx_model, reference_data)

        # The reference's only length-carrying component. Normalizing the error
        # by the current body size keeps the term scale-free, so a 0.5x and a
        # 1.5x body are held to the same relative standard.
        if self.env.has_free_base:
            reference_root_height = internal_state["reference_root_height"]
            if self.root_height_from_reference_pose and self.env.nr_feet > 0:
                foot_centers = reference_data.geom_xpos[self.env.foot_geom_indices, 2]
                foot_bottoms = foot_centers - self.env.feet_bottom_extent(reference_data, mjx_model)
                grounded_root_height = reference_qpos[2] - jnp.min(foot_bottoms)
                # FEETFIX. The plain pose-derived height puts the LOWEST
                # reference foot exactly on the floor on EVERY frame, which
                # silently deletes the clip's vertical root motion: on a dance
                # with a hop the target says "come back down" precisely when the
                # motion leaves the ground, and the policy is rewarded for
                # skimming the floor through a jump.
                #
                # "pose as floor" keeps the clip's own root height and only
                # raises it when it would drive a foot underground. So the
                # reference is grounded where the clip is grounded, airborne
                # where the clip is airborne, and never underground -- which is
                # the one property the leg-length mismatch between a human clip
                # and a randomized robot breaks.
                if self.root_height_pose_as_floor:
                    reference_root_height = jnp.maximum(
                        reference_root_height, grounded_root_height)
                else:
                    reference_root_height = grounded_root_height
                internal_state["reference_root_height"] = reference_root_height
            height_error = data.qpos[2] - (
                reference_root_height + internal_state["center_height"]
            )
            normalized_height_error = height_error / jnp.maximum(internal_state["robot_dimensions_mean"], 1e-6)
            root_height_reward = self.root_height_tracking_coeff * jnp.exp(
                -jnp.square(normalized_height_error) / self.root_height_tracking_temperature
            )
        else:
            root_height_reward = 0.0

        # ---- ROOTFIX step 2: the HEADING term ------------------------------
        #
        # The one reference quantity that needs no per-body fit. Every other
        # term on this stack is normalized by body size or by joint limits
        # because a 0.5x and a 1.5x H1 cannot reach the same positions; both
        # must turn through exactly the same number of degrees.
        #
        # Scored as an absolute heading, not a rate: the rate is already
        # commanded through goal_velocities, and Check 30 measured that channel
        # pinned to its cap for 53.36% of the clip. Integrating a saturated rate
        # loses the turn; comparing headings does not.
        # The ERROR is computed whenever there is a free base, even with the term
        # switched off, so every arm logs it. The REWARD is what the flag gates.
        # A heading error series on the control arms is the baseline this
        # experiment is measured against, and computing it costs one arctan2.
        root_heading_reward = 0.0
        heading_error = 0.0
        if self.env.has_free_base:
            w, x, y, z = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
            robot_yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            # Wrapped into (-pi, pi]. The clip's yaw is UNWRAPPED and reaches
            # 2262 deg, and the robot's is not; comparing them without wrapping
            # would charge a robot that is facing exactly the right way with
            # every full revolution the clip has made since the episode began.
            heading_error = (
                robot_yaw
                - internal_state["root_yaw_origin"]
                - internal_state["reference_root_yaw_delta"]
            )
            heading_error = jnp.arctan2(jnp.sin(heading_error), jnp.cos(heading_error))
            if self.root_heading_tracking_coeff > 0.0:
                if self.root_heading_tracking_kernel == "cosine":
                    heading_score = 0.5 * (1.0 + jnp.cos(heading_error))
                else:
                    heading_score = jnp.exp(
                        -jnp.square(heading_error) / self.root_heading_tracking_temperature
                    )
                root_heading_reward = self.root_heading_tracking_coeff * heading_score

        # ---- DeepMimic terms ----------------------------------------------
        qvel_reward = 0.0
        rpos_reward = 0.0
        rquat_reward = 0.0
        if self.deepmimic_enabled:
            # (1) joint VELOCITY tracking.
            joint_velocities = data.qvel[self.env.actuator_joint_mask_qvel]
            velocity_errors = (
                joint_velocities - internal_state["reference_joint_velocities"]
            ) * self.env.actuator_is_joint_transmission
            # Normalized by the REFERENCE's own mean square velocity, not by
            # body size. Body size is a length and the quantity here is a rate,
            # so dividing by it left the error scale robot-dependent: measured
            # H1 13.2 vs G1 2.3 for the same motion, and exp(-13.2/0.5) ~ 2e-12
            # made the term completely dead on H1 while alive on G1. Against the
            # reference's own velocity the error is dimensionless -- 1.0 means
            # "off by as much as the motion itself moves" -- so one temperature
            # is meaningful across both families.
            reference_velocity_scale = getattr(
                self.env.command_function, "reference_velocity_mean_square", 1.0
            )
            mean_squared_velocity_error = jnp.sum(
                jnp.square(velocity_errors)
            ) / trackable_count / reference_velocity_scale
            qvel_reward = self.qvel_weight_ratio * self.joint_tracking_coeff * jnp.exp(
                -mean_squared_velocity_error / self.qvel_temperature
            )

            # (2)+(3) end-effector position and orientation, PER VARIANT.
            #
            # This is the L2 capability wired into the reward: push the reference
            # joint angles through forward kinematics on the CURRENT randomized
            # mjx_model to get body targets for THIS body. A 1.4x leg gets a foot
            # target 1.4x further out at the same joint angles -- measured ratio
            # exactly 2.000 between a 0.7x and a 1.4x body.
            # Root-relative on both sides: the world-frame position is dominated
            # by where the free base happens to be, which carries no pose
            # information. loco-mujoco compares sites relative to the root body
            # for the same reason.
            reference_relative = (
                reference_data.xpos[self.tracked_body_ids]
                - reference_data.xpos[self.env.trunk_body_id]
            )

            # `deepmimic_site_target_mode`: what "the same motion" means for a
            # body with different proportions. Two answers, and it is a
            # SPECIFICATION rather than something an experiment can settle.
            #
            #   fk      the target is FK of the reference angles on THIS body.
            #           Zero joint error gives zero site error on every body
            #           (measured: 7.573 / 2.272 / 0.757 / 0.000 cm at 0.10 /
            #           0.03 / 0.01 / 0.00 rad), so this term re-weights joint
            #           error through FK and carries no body-specific target.
            #           A shorter arm makes a proportionally smaller gesture.
            #
            #   scaled  keep the NOMINAL body's direction for each limb, and the
            #           current body's own reach along it. "Make the same shape,
            #           sized for you." This is the only construction that asks a
            #           randomized body for something its joint angles do not
            #           already encode.
            #
            # Reachability is deliberately NOT required here. Solving all eight
            # sites simultaneously as an IK problem is over-specified -- measured
            # 4.90 cm residual, 64.8% of sites missed (phase2_similarity_
            # reference.py). A REWARD target has no such constraint: it supplies
            # a gradient toward the scaled shape, and the policy trades it off
            # against the other terms. That distinction is why this arm is worth
            # running even though the IK version is not.
            if self.site_target_mode == "scaled":
                nominal_data = mjx.kinematics(
                    self.env.initial_mjx_model, reference_data)
                nominal_relative = (
                    nominal_data.xpos[self.tracked_body_ids]
                    - nominal_data.xpos[self.env.trunk_body_id]
                )
                nominal_norm = jnp.linalg.norm(nominal_relative, axis=-1, keepdims=True)
                reference_norm = jnp.linalg.norm(reference_relative, axis=-1, keepdims=True)
                # A site sitting on the trunk has no direction to preserve; leave
                # those at the FK target rather than dividing by ~0.
                usable = nominal_norm > 1e-6
                scaled_relative = jnp.where(
                    usable,
                    nominal_relative / jnp.maximum(nominal_norm, 1e-6) * reference_norm,
                    reference_relative,
                )
                # Under a UNIFORM scale s, FK gives exactly s x the nominal
                # vector, so direction is unchanged and this reduces to the fk
                # target identically -- the two modes only diverge where the body
                # is anisotropic, which is the intended behaviour.
                reference_relative = scaled_relative
            current_relative = (
                data.xpos[self.tracked_body_ids] - data.xpos[self.env.trunk_body_id]
            )

            # ---- ARM C: ReActor-style adapted reference ---------------------
            #
            # ReActor (Disney, arXiv 2605.06593) treats retargeting as the OUTER
            # loop of a bilevel problem: bounded per-body offsets are adjusted
            # during RL so the reference becomes something this body can actually
            # achieve, while the policy learns to track it. That removes the
            # assumption that an offline retarget is feasible on a body it never
            # saw -- which is exactly the assumption a +-50% randomized H1 breaks.
            #
            # SCOPE, stated honestly: this is a within-episode EMA, not ReActor's
            # two-timescale outer optimizer. The offsets are carried in
            # internal_state and reset with the episode, so they adapt to THIS
            # body during THIS episode rather than being optimized globally across
            # training. That is a weaker instrument, and it is what can be built
            # without modifying RL-X's update loop.
            #
            # WHAT STOPS IT DEGENERATING: with an unbounded offset the target
            # would converge to wherever the robot already is, tracking error
            # would go to zero, and the reference would carry no information at
            # all -- a reward that is always satisfied. Two guards: the rate is
            # slow, and the offset is hard-clipped to a fraction of body size.
            # The clipped fraction is logged so the arm cannot silently become
            # "no reference".
            if self.reference_adapt_rate > 0.0:
                body_scale = jnp.maximum(internal_state["robot_dimensions_mean"], 1e-6)
                bound = self.reference_adapt_bound * body_scale
                offset = internal_state["reference_site_offset"]
                # Move the target toward what the body achieved, slowly.
                residual = current_relative - (reference_relative + offset)
                offset = offset + self.reference_adapt_rate * residual
                offset = jnp.clip(offset, -bound, bound)
                internal_state["reference_site_offset"] = offset
                reference_relative = reference_relative + offset
                info[f"env_info/reference_adapt_saturated/{self.env.robot_config['short_name']}"] = (
                    jnp.mean((jnp.abs(offset) >= bound * 0.999).astype(jnp.float32)))
                info[f"env_info/reference_adapt_magnitude/{self.env.robot_config['short_name']}"] = (
                    jnp.mean(jnp.linalg.norm(offset, axis=-1)) / body_scale)
            # Normalized by body size so the term is scale-free: without this a
            # large body would be charged more reward for the same RELATIVE
            # pose error purely because its limbs are longer.
            # Positions ARE lengths, so body size is the right scale here (unlike
            # the velocity term above, where it was a category error).
            body_length_scale = jnp.maximum(internal_state["robot_dimensions_mean"], 1e-6)
            position_error = (current_relative - reference_relative) / body_length_scale
            rpos_distance = jnp.mean(jnp.square(position_error))
            rpos_reward = self.rpos_weight_ratio * self.joint_tracking_coeff * jnp.exp(
                -rpos_distance / self.rpos_temperature
            )

            # Orientation: compare rotation MATRICES elementwise rather than
            # quaternions. A quaternion and its negation encode the same
            # rotation, so a raw quaternion difference reports a large error for
            # two identical orientations -- the double-cover trap. The matrix
            # form has no such ambiguity and needs no sign canonicalization.
            reference_rotation = reference_data.xmat[self.tracked_body_ids]
            current_rotation = data.xmat[self.tracked_body_ids]
            rquat_distance = jnp.mean(jnp.square(current_rotation - reference_rotation))
            rquat_reward = self.rquat_weight_ratio * self.joint_tracking_coeff * jnp.exp(
                -rquat_distance / self.rquat_temperature
            )

            qvel_reward = tracking_gate * qvel_reward
            rpos_reward = tracking_gate * rpos_reward
            rquat_reward = tracking_gate * rquat_reward

        # ---- FEETFIX: foot HEIGHT against the reference's own feet ----------
        #
        # The gap this closes: rpos compares body positions RELATIVE TO THE
        # TRUNK and divides by body size, so it scores the shape of the pose and
        # is blind to where that pose sits against the floor. Nothing else in
        # the reward looks at foot height at all. A policy that performs the
        # dance 3 cm sunk into the ground, or skimming 3 cm above it, collects
        # the full imitation reward -- which is exactly what the 22->23-08
        # videos show.
        #
        # Target construction, deliberately the same as every other term here:
        # push the reference ANGLES through FK on the CURRENT randomized model,
        # measure that pose's foot bottoms relative to its own root, then place
        # the root at the height the root-height term is already asking for. So
        # the term reads "with your root where the reference wants it, your feet
        # belong where the reference's feet are" -- one consistent target, and
        # it needs no foot correspondence with the clip, which is what keeps it
        # working across topologies and randomized bodies.
        foot_height_reward = 0.0
        if self.foot_reference_metrics:
            reference_root_height_now = internal_state["reference_root_height"]
            reference_feet_bottom = (
                reference_data.geom_xpos[self.env.foot_geom_indices, 2]
                - self.env.feet_bottom_extent(reference_data, mjx_model)
            )
            # Relative to the reference pose's own root, then re-based onto the
            # commanded root height over the ground.
            reference_feet_over_ground = (
                reference_feet_bottom - reference_qpos[2] + reference_root_height_now
            )
            robot_feet_over_ground = (
                data.geom_xpos[self.env.foot_geom_indices, 2]
                - self.env.feet_bottom_extent(data, mjx_model)
                - internal_state["center_height"]
            )
            body_length_scale = jnp.maximum(internal_state["robot_dimensions_mean"], 1e-6)
            foot_height_error = jnp.mean(jnp.square(
                (robot_feet_over_ground - reference_feet_over_ground) / body_length_scale))
            if self.foot_height_active:
                foot_height_reward = tracking_gate * self.foot_height_coeff * jnp.exp(
                    -foot_height_error / self.foot_height_temperature
                )
            if self.log_info:
                short_name = self.env.robot_config["short_name"]
                info[f"env_info/foot_height_error/{short_name}"] = foot_height_error
                info[f"env_info/foot_height_reward/{short_name}"] = foot_height_reward
                # Is the REFERENCE itself underground or airborne? Under
                # tracking_clip_root_height_from_pose=True the first is 0 by
                # construction and the second is 0 every frame -- which is its
                # own finding, because a dance with a hop then has a target that
                # never leaves the floor. Logged on every arm so the question is
                # answerable without a dedicated run.
                info[f"env_info/ref_foot_penetration_m/{short_name}"] = jnp.mean(
                    jnp.maximum(-reference_feet_over_ground, 0.0))
                info[f"env_info/ref_foot_airborne/{short_name}"] = (
                    jnp.min(reference_feet_over_ground) > 0.02).astype(jnp.float32)

        if self.log_info:
            # Suffix with the robot's short name, matching how environment.py
            # writes every other env_info key. Without it MultiEnvironment
            # concatenates H1's and G1's values under one key and the logged
            # scalar is pooled over both bodies -- which cannot answer whether
            # tracking error differs *between* morphologies, the Phase 1
            # question. Per-robot keys give one series per body instead.
            short_name = self.env.robot_config["short_name"]
            info[f"env_info/joint_tracking_error/{short_name}"] = mean_squared_joint_error
            info[f"env_info/joint_tracking_reward/{short_name}"] = joint_tracking_reward
            info[f"env_info/root_height_tracking_reward/{short_name}"] = root_height_reward
            # Logged unconditionally, including when the term is off. A heading
            # ERROR series recorded on every arm is what makes "does the policy
            # turn?" answerable from arms that were never asked to turn, which
            # is the control this experiment otherwise has to buy with a slot.
            info[f"env_info/root_heading_error/{short_name}"] = jnp.abs(heading_error)
            info[f"env_info/root_heading_tracking_reward/{short_name}"] = root_heading_reward
            if self.deepmimic_enabled:
                # Log every term separately: the 2026-08-04 failure was a term
                # BALANCE failure, and a pooled tracking reward cannot show which
                # term is dominating.
                # Whether the qvel normalizer actually engaged: the clip's own
                # mean-square velocity (4.0-5.8 rad^2/s^2 on dance2_subject4)
                # if live, 1.0 if the getattr fallback fired. Settles whether a
                # logged error of ~13 means "3.6x the motion" or "unnormalized".
                info[f"env_info/reference_velocity_scale/{short_name}"] = reference_velocity_scale
                info[f"env_info/qvel_tracking_error/{short_name}"] = mean_squared_velocity_error
                info[f"env_info/qvel_tracking_reward/{short_name}"] = qvel_reward
                info[f"env_info/rpos_tracking_error/{short_name}"] = rpos_distance
                info[f"env_info/rpos_tracking_reward/{short_name}"] = rpos_reward
                info[f"env_info/rquat_tracking_error/{short_name}"] = rquat_distance
                info[f"env_info/rquat_tracking_reward/{short_name}"] = rquat_reward

        reward = (reward + joint_tracking_reward + root_height_reward
                  + root_heading_reward + foot_height_reward
                  + qvel_reward + rpos_reward + rquat_reward)
        if self.post_contact_penalties:
            # Preserve DefaultReward's non-negative convention and always-on
            # alive floor, while allowing physical contact quality to compete
            # with bonuses that were historically added only after clipping.
            alive_floor = (
                self.env.resolve_gait_coeff(internal_state)
                * self.alive_unclipped_coeff
            )
            reward = jnp.maximum(
                reward - alive_floor + internal_state["tracking_post_contact_penalty"],
                0.0,
            ) + alive_floor
            reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
            if self.log_info:
                info[f"reward/total/{self.env.robot_config['short_name']}"] = reward
        return reward
